/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package opennlp.tools.coref;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;
import opennlp.tools.document.NameFinderAnnotator;
import opennlp.tools.util.Span;
import opennlp.tools.util.StringUtil;

/**
 * Deterministic coreference resolution over the document graph: entity mentions and
 * third-person pronouns are linked by precision-ordered sieves and provided as
 * {@link #CHAINS}, one annotation per mention carrying its {@link CorefMention}.
 *
 * <p>Three sieves run in order of decreasing precision. Exact match links entity
 * mentions of the same type with identical normalized text, regardless of how far apart
 * they are. Name containment links entity mentions of the same type where one is a
 * whitespace-delimited prefix or suffix of the other, so a surname finds its full name.
 * Pronoun resolution links each third-person
 * pronoun to the nearest preceding compatible entity mention in the same or the
 * directly preceding sentence: gendered pronouns to person entities, neutral pronouns
 * to the configured non-person types, and plural pronouns to either. First and second
 * person pronouns are not resolved, since they refer to speakers the text alone does
 * not identify.</p>
 *
 * <p>An entity whose type is {@link NameFinderAnnotator#UNTYPED}, the label an untyped
 * name finder's mentions arrive with, has an unknown type rather than a mismatching
 * one: it satisfies every type guard, so it links to identical or contained names of
 * any type and to every pronoun class. The sieves cannot discriminate on a type that
 * carries no information, so they must not silently exclude it.</p>
 *
 * <p>Every mention is reported, including those that found no partner, as singleton
 * chains; consumers interested only in links filter by chain size. Chains are numbered
 * in order of first mention. The resolution is rule-based and needs no model or
 * training data.</p>
 *
 * <p>The annotator holds no per-call state and is safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public class CorefAnnotator implements DocumentAnnotator {

  /**
   * Coreference mentions; each annotation covers one mention and carries its
   * {@link CorefMention}, ordered by text position.
   */
  public static final LayerKey<CorefMention> CHAINS =
      LayerKey.of("chains", CorefMention.class);

  /** How many sentences back a pronoun may find its antecedent. */
  private static final int MAX_SENTENCE_DISTANCE = 1;

  /** POS tags that mark pronoun tokens: Penn Treebank PRP and PRP$, Universal PRON. */
  private static final Set<String> PRONOUN_TAGS = Set.of("PRP", "PRP$", "PRON");

  /** Third-person singular pronoun forms that resolve only to person entities. */
  private static final Set<String> GENDERED_PRONOUNS = Set.of(
      "he", "him", "his", "himself", "she", "her", "hers", "herself");

  /** Third-person plural pronoun forms that resolve to person or non-person entities. */
  private static final Set<String> PLURAL_PRONOUNS = Set.of(
      "they", "them", "their", "theirs", "themselves");

  /** Third-person neutral pronoun forms that resolve only to non-person entities. */
  private static final Set<String> NEUTRAL_PRONOUNS = Set.of("it", "its", "itself");

  /** Lowercased entity type labels gendered pronouns may resolve to. */
  private final Set<String> personTypes;

  /** Lowercased entity type labels neutral pronouns may resolve to. */
  private final Set<String> neutralTypes;

  /**
   * Initializes the annotator for the conventional entity type names: {@code person}
   * for gendered pronouns and {@code organization} and {@code location} for neutral
   * pronouns.
   */
  public CorefAnnotator() {
    this(Set.of("person"), Set.of("organization", "location"));
  }

  /**
   * Initializes the annotator.
   *
   * @param personTypes The entity type labels gendered pronouns may resolve to, matched
   *                    case-insensitively. Must not be {@code null} or empty.
   * @param neutralTypes The entity type labels neutral pronouns may resolve to, matched
   *                     case-insensitively. Must not be {@code null} or empty.
   * @throws IllegalArgumentException Thrown if a set is {@code null}, empty, or
   *         contains a blank entry.
   */
  public CorefAnnotator(Set<String> personTypes, Set<String> neutralTypes) {
    this.personTypes = lowered(personTypes, "personTypes");
    this.neutralTypes = lowered(neutralTypes, "neutralTypes");
  }

  /**
   * Validates and lowercases one type set.
   *
   * @param types The type labels.
   * @param name The parameter name for error messages.
   * @return The lowercased set. Never {@code null}.
   */
  private static Set<String> lowered(Set<String> types, String name) {
    if (types == null || types.isEmpty()) {
      throw new IllegalArgumentException(name + " must not be null or empty");
    }
    final Set<String> lowered = new HashSet<>(types.size());
    for (final String type : types) {
      if (type == null || type.isBlank()) {
        throw new IllegalArgumentException(name + " must not contain blank entries");
      }
      lowered.add(StringUtil.toLowerCase(type));
    }
    return Set.copyOf(lowered);
  }

  /**
   * One collected mention before chain assignment: its character span, its kind, the
   * index of its source entity or {@link CorefMention#NO_ENTITY} for pronouns, the
   * lowercased entity type or {@code null} for pronouns, the lowercased covered text,
   * and the index of the sentence the mention starts in.
   */
  private record Mention(Span span, String kind, int entity, String type,
      String normalized, int sentence) {
  }

  /**
   * Resolves coreference over the document and adds the {@link #CHAINS} layer.
   *
   * <p>A document whose token layer is empty gets an empty chains layer without further
   * checks. Otherwise the token and POS tag layers must be aligned one to one and the
   * sentence layer must not be empty.</p>
   *
   * @param document The document to annotate. Must not be {@code null}.
   * @return A new {@link Document} carrying the {@link #CHAINS} layer. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null}, the
   *         token and POS tag layers differ in size, or tokens are present but the
   *         sentence layer is empty.
   */
  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final CharSequence text = document.text();
    final List<Annotation<String>> sentences = document.get(Layers.SENTENCES);
    final List<Annotation<String>> tokens = document.get(Layers.TOKENS);
    final List<Annotation<String>> tags = document.get(Layers.POS_TAGS);
    final List<Annotation<String>> entities = document.get(Layers.ENTITIES);
    if (tags.size() != tokens.size()) {
      throw new IllegalArgumentException("document needs aligned "
          + Layers.TOKENS + " and " + Layers.POS_TAGS + " layers");
    }
    if (tokens.isEmpty()) {
      return document.with(CHAINS, List.of());
    }
    if (sentences.isEmpty()) {
      throw new IllegalArgumentException(
          "document needs a non-empty " + Layers.SENTENCES + " layer");
    }

    final List<Mention> mentions = collectMentions(text, sentences, tokens, tags, entities);
    final int[] parent = new int[mentions.size()];
    for (int i = 0; i < parent.length; i++) {
      parent[i] = i;
    }
    exactMatchSieve(mentions, parent);
    containmentSieve(mentions, parent);
    pronounSieve(mentions, parent);

    final Map<Integer, Integer> chainIds = new HashMap<>();
    final List<Annotation<CorefMention>> layer = new ArrayList<>(mentions.size());
    for (int i = 0; i < mentions.size(); i++) {
      final int root = find(parent, i);
      final int chain = chainIds.computeIfAbsent(root, key -> chainIds.size());
      final Mention mention = mentions.get(i);
      layer.add(new Annotation<>(mention.span(),
          new CorefMention(chain, mention.kind(), mention.entity())));
    }
    return document.with(CHAINS, layer);
  }

  @Override
  public Set<LayerKey<?>> requires() {
    return Set.of(Layers.SENTENCES, Layers.TOKENS, Layers.POS_TAGS, Layers.ENTITIES);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(CHAINS);
  }

  /**
   * Collects entity mentions and third-person pronoun mentions in text order. Every
   * entity annotation becomes a mention. A token becomes a pronoun mention only if its
   * POS tag is a pronoun tag, its lowercased form is a known third-person form, and no
   * entity span covers it; first and second person forms are left out entirely.
   *
   * @param text The document text.
   * @param sentences The sentence layer.
   * @param tokens The token layer.
   * @param tags The POS tag layer.
   * @param entities The entity layer.
   * @return The mentions sorted by start offset. Never {@code null}.
   */
  private List<Mention> collectMentions(CharSequence text,
      List<Annotation<String>> sentences, List<Annotation<String>> tokens,
      List<Annotation<String>> tags, List<Annotation<String>> entities) {
    final List<Mention> mentions = new ArrayList<>();
    for (int e = 0; e < entities.size(); e++) {
      final Span span = entities.get(e).span();
      mentions.add(new Mention(span, CorefMention.KIND_ENTITY, e,
          StringUtil.toLowerCase(entities.get(e).value()),
          StringUtil.toLowerCase(text.subSequence(span.getStart(), span.getEnd())),
          sentenceOf(span, sentences)));
    }
    for (int t = 0; t < tokens.size(); t++) {
      if (!PRONOUN_TAGS.contains(tags.get(t).value())) {
        continue;
      }
      final String form = StringUtil.toLowerCase(tokens.get(t).value());
      if (!GENDERED_PRONOUNS.contains(form) && !PLURAL_PRONOUNS.contains(form)
          && !NEUTRAL_PRONOUNS.contains(form)) {
        continue;
      }
      final Span span = tokens.get(t).span();
      if (coveredByEntity(span, entities)) {
        continue;
      }
      mentions.add(new Mention(span, CorefMention.KIND_PRONOUN, CorefMention.NO_ENTITY,
          null, form, sentenceOf(span, sentences)));
    }
    // restore text order after appending the pronouns behind the entity mentions
    mentions.sort((a, b) -> Integer.compare(a.span().getStart(), b.span().getStart()));
    return mentions;
  }

  /**
   * Links entity mentions of compatible types with identical normalized text. Every
   * later occurrence is linked to the earliest compatible one, with no limit on their
   * distance.
   *
   * @param mentions The mentions in text order.
   * @param parent The union-find forest over mention indices.
   */
  private static void exactMatchSieve(List<Mention> mentions, int[] parent) {
    final Map<String, List<Integer>> earlierByText = new HashMap<>();
    for (int i = 0; i < mentions.size(); i++) {
      final Mention mention = mentions.get(i);
      if (mention.entity() == CorefMention.NO_ENTITY) {
        continue;
      }
      final List<Integer> earlier =
          earlierByText.computeIfAbsent(mention.normalized(), key -> new ArrayList<>());
      for (final int candidate : earlier) {
        if (typesCompatible(mentions.get(candidate).type(), mention.type())) {
          union(parent, candidate, i);
          break;
        }
      }
      earlier.add(i);
    }
  }

  /**
   * Links entity mentions of compatible types where one text is a whitespace-delimited
   * prefix or suffix of the other, with no limit on their distance.
   *
   * @param mentions The mentions in text order.
   * @param parent The union-find forest over mention indices.
   */
  private static void containmentSieve(List<Mention> mentions, int[] parent) {
    for (int i = 0; i < mentions.size(); i++) {
      final Mention a = mentions.get(i);
      if (a.entity() == CorefMention.NO_ENTITY) {
        continue;
      }
      for (int j = i + 1; j < mentions.size(); j++) {
        final Mention b = mentions.get(j);
        if (b.entity() == CorefMention.NO_ENTITY || !typesCompatible(a.type(), b.type())) {
          continue;
        }
        if (containsAsWords(a.normalized(), b.normalized())
            || containsAsWords(b.normalized(), a.normalized())) {
          union(parent, i, j);
        }
      }
    }
  }

  /**
   * Checks whether the shorter text is a whitespace-delimited prefix or suffix of the
   * longer one, so that only whole leading or trailing words count as contained.
   *
   * @param longer The candidate containing text.
   * @param shorter The candidate contained text.
   * @return {@code true} if {@code shorter} starts or ends {@code longer} and the
   *         character adjoining the shared part is a whitespace.
   */
  private static boolean containsAsWords(String longer, String shorter) {
    if (longer.length() <= shorter.length()) {
      return false;
    }
    return (longer.startsWith(shorter)
            && StringUtil.isWhitespace(longer.charAt(shorter.length())))
        || (longer.endsWith(shorter)
            && StringUtil.isWhitespace(longer.charAt(longer.length() - shorter.length() - 1)));
  }

  /**
   * Links each pronoun to the nearest preceding compatible entity mention. The backward
   * scan skips other pronouns, stops once a candidate lies more than
   * {@link #MAX_SENTENCE_DISTANCE} sentences back, and leaves the pronoun a singleton
   * when no candidate in range is compatible.
   *
   * @param mentions The mentions in text order.
   * @param parent The union-find forest over mention indices.
   */
  private void pronounSieve(List<Mention> mentions, int[] parent) {
    for (int i = 0; i < mentions.size(); i++) {
      final Mention pronoun = mentions.get(i);
      if (pronoun.entity() != CorefMention.NO_ENTITY) {
        continue;
      }
      for (int j = i - 1; j >= 0; j--) {
        final Mention candidate = mentions.get(j);
        if (candidate.entity() == CorefMention.NO_ENTITY) {
          continue;
        }
        if (pronoun.sentence() - candidate.sentence() > MAX_SENTENCE_DISTANCE) {
          break;
        }
        if (compatible(pronoun.normalized(), candidate.type())) {
          union(parent, j, i);
          break;
        }
      }
    }
  }

  /**
   * Checks whether a pronoun may refer to an entity type. The unknown type
   * {@link NameFinderAnnotator#UNTYPED} is accepted by every pronoun class.
   *
   * @param pronoun The lowercased pronoun form.
   * @param type The lowercased entity type.
   * @return {@code true} if the pronoun class accepts the type.
   */
  private boolean compatible(String pronoun, String type) {
    if (NameFinderAnnotator.UNTYPED.equals(type)) {
      return true;
    }
    if (GENDERED_PRONOUNS.contains(pronoun)) {
      return personTypes.contains(type);
    }
    if (NEUTRAL_PRONOUNS.contains(pronoun)) {
      return neutralTypes.contains(type);
    }
    return personTypes.contains(type) || neutralTypes.contains(type);
  }

  /**
   * Checks whether two entity mentions may corefer by type: their types are equal, or
   * one of them is the unknown type {@link NameFinderAnnotator#UNTYPED}, which never
   * blocks a link.
   *
   * @param a The first lowercased entity type.
   * @param b The second lowercased entity type.
   * @return {@code true} if the types do not rule the link out.
   */
  private static boolean typesCompatible(String a, String b) {
    return a.equals(b)
        || NameFinderAnnotator.UNTYPED.equals(a) || NameFinderAnnotator.UNTYPED.equals(b);
  }

  /**
   * Finds the sentence a span belongs to.
   *
   * @param span The mention span.
   * @param sentences The sentence layer. Must not be empty.
   * @return The index of the first sentence whose end lies beyond the span start, so a
   *         span starting in the gap between two sentences counts to the following one;
   *         the last sentence when the span starts after every sentence end.
   */
  private static int sentenceOf(Span span, List<Annotation<String>> sentences) {
    for (int s = 0; s < sentences.size(); s++) {
      if (span.getStart() < sentences.get(s).span().getEnd()) {
        return s;
      }
    }
    return sentences.size() - 1;
  }

  /**
   * Checks whether a token span lies inside any entity span.
   *
   * @param span The token span.
   * @param entities The entity layer.
   * @return {@code true} if an entity covers the token.
   */
  private static boolean coveredByEntity(Span span, List<Annotation<String>> entities) {
    for (final Annotation<String> entity : entities) {
      if (span.getStart() >= entity.span().getStart()
          && span.getEnd() <= entity.span().getEnd()) {
        return true;
      }
    }
    return false;
  }

  /**
   * Finds the representative of a mention's set, compressing the visited path so later
   * lookups reach the root directly.
   *
   * @param parent The union-find forest.
   * @param i The mention index.
   * @return The root index.
   */
  private static int find(int[] parent, int i) {
    int root = i;
    while (parent[root] != root) {
      root = parent[root];
    }
    while (parent[i] != root) {
      final int next = parent[i];
      parent[i] = root;
      i = next;
    }
    return root;
  }

  /**
   * Merges two mentions' sets, keeping the earlier root.
   *
   * @param parent The union-find forest.
   * @param a The first mention index.
   * @param b The second mention index.
   */
  private static void union(int[] parent, int a, int b) {
    final int rootA = find(parent, a);
    final int rootB = find(parent, b);
    if (rootA != rootB) {
      parent[Math.max(rootA, rootB)] = Math.min(rootA, rootB);
    }
  }
}
