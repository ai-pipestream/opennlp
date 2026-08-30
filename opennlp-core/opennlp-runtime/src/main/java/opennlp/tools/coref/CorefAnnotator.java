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

import opennlp.tools.chunker.ChunkerAnnotator;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;
import opennlp.tools.namefind.NameFinderAnnotator;
import opennlp.tools.parser.ParserAnnotator;
import opennlp.tools.parser.ParserAnnotator.Phrase;
import opennlp.tools.util.StringUtil;

/**
 * Deterministic coreference resolution over the document graph: entity mentions, noun
 * phrase mentions, and third-person pronouns are linked by precision-ordered sieves and
 * provided as {@link #CHAINS}, one annotation per mention carrying its
 * {@link CorefMention}.
 *
 * <p>The resolver follows the entity-centric, precision-ranked design of
 * <a href="https://aclanthology.org/J13-4004/">Lee et al. (Computational Linguistics
 * 2013), "Deterministic Coreference Resolution Based on Entity-Centric, Precision-Ranked
 * Rules"</a>. Mentions are the entity annotations, the noun phrases of a
 * {@link ParserAnnotator#PHRASES} layer or, without one, the noun phrase chunks of a
 * {@link ChunkerAnnotator#CHUNKS} layer when the document carries either, and the
 * tokens tagged as third-person pronouns; of parse phrases sharing a head only the
 * largest counts, a phrase headed by an entity widens that entity's mention to the full
 * noun phrase, and a pleonastic {@code it} is no mention. Each
 * mention carries the number, gender, animacy, and person the text supports: pronoun
 * forms, plural tags, name titles, a bundled first-name list, and a list of gendered and
 * animate nouns. First and second person pronouns refer to the speaker and the
 * addressee rather than to an antecedent in the text: they chain per speaker, read from
 * an optional {@link #SPEAKERS} layer or, without one, from quotation marks, and a quoted
 * first person joins the person a verb of speech attributes the quotation to.</p>
 *
 * <p>Nine sieves run in order of decreasing precision, each linking an anaphor to the
 * first candidate antecedent, in salience order, whose whole cluster passes: exact
 * string match; relaxed string match on the text up to the head; the precise constructs
 * acronym and person-name containment, so a surname finds its full name while a place
 * name never absorbs a compound such as {@code Kansas City}; strict head match with
 * cluster head match, word inclusion, and compatible modifiers, then its two relaxations
 * dropping one condition each; proper head word match; relaxed head match between
 * entities of one type; and pronoun resolution within three sentences under number,
 * gender, animacy, person, and entity type agreement, gendered pronouns to the person
 * types, neutral pronouns to the non-person types, plural pronouns to either.
 * Indefinite noun phrases are antecedents only, no sieve links a mention to one that
 * contains it, and clusters whose known entity types differ never merge; an entity of
 * the unknown type {@link NameFinderAnnotator#UNTYPED} joins a cluster of any type and
 * adopts it without ever bridging two types.</p>
 *
 * <p>Every mention is reported, including those that found no partner, as singleton
 * chains; consumers interested only in links filter by chain size. Chains are numbered
 * in order of first mention. The resolution is rule-based and needs no model or
 * training data; given a {@link CorefModel}, the sieves after the speaker sieve are
 * replaced by a ranking of each anaphor's candidates with the sieve tests among the
 * features, which trades the rules' fixed precision order for what the training corpus
 * shows.</p>
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
      Layers.key("chains", CorefMention.class);

  /**
   * Hand-annotated coreference chains, the gold counterpart of {@link #CHAINS} under
   * the {@code gold:} convention; {@link CorefTrainer} reads it and the annotator never
   * does.
   */
  public static final LayerKey<CorefMention> GOLD_CHAINS =
      LayerKey.of("gold:" + CHAINS.id(), CorefMention.class);

  /**
   * Speakers, an optional input layer: each annotation covers the text one speaker
   * utters, typically a sentence or a turn, and carries the speaker's label. First and
   * second person pronouns resolve per speaker; without the layer, quotation marks
   * delimit the speakers instead.
   */
  public static final LayerKey<String> SPEAKERS = Layers.key("speakers", String.class);

  /** The message prefix of every absent-required-layer rejection in this annotator. */
  private static final String MISSING_LAYER = "document lacks the required layer ";

  /** Lowercased entity type labels gendered pronouns may resolve to. */
  private final Set<String> personTypes;

  /** Lowercased entity type labels neutral pronouns may resolve to. */
  private final Set<String> neutralTypes;

  /** The ranking model, or {@code null} for rule-based resolution. */
  private final CorefModel model;

  /** The least link probability at which the ranker links a pair. */
  private final double threshold;

  /** Word vectors for the ranker's similarity features, or {@code null}. */
  private final WordVectors vectors;

  /**
   * The link probability threshold of the model constructors. Pair classifiers see far
   * more unlinked than linked pairs, so their link probabilities run low; this floor
   * was chosen on the OntoGUM development split for models {@link CorefTrainer}
   * produces.
   */
  public static final double DEFAULT_THRESHOLD = 0.1;

  /**
   * Initializes the rule-based annotator for the conventional entity type names:
   * {@code person} for gendered pronouns and {@code organization} and {@code location}
   * for neutral pronouns.
   */
  public CorefAnnotator() {
    this(Set.of("person"), Set.of("organization", "location"));
  }

  /**
   * Initializes the rule-based annotator.
   *
   * @param personTypes The entity type labels gendered pronouns may resolve to, matched
   *                    case-insensitively. Must not be {@code null} or empty.
   * @param neutralTypes The entity type labels neutral pronouns may resolve to, matched
   *                     case-insensitively. Must not be {@code null} or empty.
   * @throws IllegalArgumentException Thrown if a set is {@code null}, empty, or
   *         contains a blank entry.
   */
  public CorefAnnotator(Set<String> personTypes, Set<String> neutralTypes) {
    this(personTypes, neutralTypes, null, DEFAULT_THRESHOLD);
  }

  /**
   * Initializes a ranking annotator for the conventional entity type names.
   *
   * @param model The ranking model. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code model} is {@code null}.
   */
  public CorefAnnotator(CorefModel model) {
    this(Set.of("person"), Set.of("organization", "location"), requireModel(model),
        DEFAULT_THRESHOLD);
  }

  /**
   * Initializes a ranking annotator.
   *
   * @param personTypes The entity type labels gendered pronouns may resolve to, matched
   *                    case-insensitively. Must not be {@code null} or empty.
   * @param neutralTypes The entity type labels neutral pronouns may resolve to, matched
   *                     case-insensitively. Must not be {@code null} or empty.
   * @param model The ranking model. Must not be {@code null}.
   * @param threshold The least link probability at which a pair is linked, in
   *                  {@code [0, 1]}.
   * @throws IllegalArgumentException Thrown if a set is {@code null}, empty, or
   *         contains a blank entry, {@code model} is {@code null}, or
   *         {@code threshold} lies outside {@code [0, 1]}.
   */
  public CorefAnnotator(Set<String> personTypes, Set<String> neutralTypes, CorefModel model,
      double threshold) {
    this(personTypes, neutralTypes, model, threshold, null);
  }

  /**
   * Initializes an annotator with word vectors for the ranker's head similarity
   * features. The vectors must be the ones the model was trained with; without a model
   * they are unused.
   *
   * @param personTypes The entity type labels gendered pronouns may resolve to, matched
   *                    case-insensitively. Must not be {@code null} or empty.
   * @param neutralTypes The entity type labels neutral pronouns may resolve to, matched
   *                     case-insensitively. Must not be {@code null} or empty.
   * @param model The ranking model, or {@code null} for rule-based resolution.
   * @param threshold The least link probability at which a pair classifier links, in
   *                  {@code [0, 1]}.
   * @param vectors The word vectors, or {@code null} for none.
   * @throws IllegalArgumentException Thrown if a set is {@code null}, empty, or
   *         contains a blank entry, or {@code threshold} lies outside {@code [0, 1]}.
   */
  public CorefAnnotator(Set<String> personTypes, Set<String> neutralTypes, CorefModel model,
      double threshold, WordVectors vectors) {
    this.personTypes = lowered(personTypes, "personTypes");
    this.neutralTypes = lowered(neutralTypes, "neutralTypes");
    this.model = model;
    if (!(threshold >= 0.0 && threshold <= 1.0)) {
      throw new IllegalArgumentException("threshold must lie in [0, 1]: " + threshold);
    }
    this.threshold = threshold;
    this.vectors = vectors;
  }

  private static CorefModel requireModel(CorefModel model) {
    if (model == null) {
      throw new IllegalArgumentException("model must not be null");
    }
    return model;
  }

  /**
   * Validates and lowercases one type set.
   *
   * @param types The type labels.
   * @param name The parameter name for error messages.
   * @return The lowercased set. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code types} is {@code null}, empty, or
   *         contains a {@code null} or blank entry.
   */
  private static Set<String> lowered(Set<String> types, String name) {
    if (types == null || types.isEmpty()) {
      throw new IllegalArgumentException(name + " must not be null or empty");
    }
    final Set<String> lowered = new HashSet<>(types.size());
    for (final String type : types) {
      if (type == null || StringUtil.isBlank(type)) {
        throw new IllegalArgumentException(name + " must not contain blank entries");
      }
      lowered.add(StringUtil.toLowerCase(type));
    }
    return Set.copyOf(lowered);
  }

  /**
   * Resolves coreference over the document and adds the {@link #CHAINS} layer.
   *
   * <p>The required layers must be present, but they may be empty, and the token and POS
   * tag layers must be aligned one to one. A document without tokens then yields a
   * present-but-empty chains layer; a document that has tokens needs a non-empty sentence
   * layer to place its mentions in. A {@link ParserAnnotator#PHRASES} or
   * {@link ChunkerAnnotator#CHUNKS} layer is optional: with either, noun phrases become
   * mentions, the parse taking precedence; without both, only entities and pronouns
   * do. A {@link #SPEAKERS} layer is optional as well.</p>
   *
   * @param document The document to annotate. Must not be {@code null} and must carry the
   *                 {@link Layers#SENTENCES}, {@link Layers#TOKENS},
   *                 {@link Layers#POS_TAGS}, and {@link Layers#ENTITIES} layers.
   * @return A new {@link Document} carrying the {@link #CHAINS} layer. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null}, a
   *         required layer is absent, the token and POS tag layers differ in size, or
   *         tokens are present but the sentence layer is empty.
   */
  @Override
  public Document annotate(Document document) {
    final SieveResolver resolver = resolver(document);
    if (resolver == null) {
      return document.with(CHAINS, List.of());
    }
    if (model == null) {
      resolver.resolve();
    } else {
      resolver.resolve(model.getPairModel(), model.isRanking(), threshold);
    }
    final List<Mention> mentions = resolver.mentions();
    final Clusters clusters = resolver.clusters();
    final Map<Integer, Integer> chainIds = new HashMap<>();
    final List<Annotation<CorefMention>> layer = new ArrayList<>(mentions.size());
    for (int i = 0; i < mentions.size(); i++) {
      final int root = clusters.find(i);
      final int chain = chainIds.computeIfAbsent(root, key -> chainIds.size());
      final Mention mention = mentions.get(i);
      layer.add(new Annotation<>(mention.span(),
          new CorefMention(chain, mention.kind(), mention.entity())));
    }
    return document.with(CHAINS, layer);
  }

  /**
   * Validates a document, detects its mentions, and builds the resolver over them with
   * every cluster still a singleton.
   *
   * @param document The document to annotate.
   * @return The resolver, or {@code null} for a document without tokens, whose chains
   *         layer is empty.
   * @throws IllegalArgumentException Thrown if the document is invalid, as
   *         {@link #annotate(Document)} documents.
   */
  SieveResolver resolver(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    final Set<LayerKey<?>> present = document.layers();
    if (!present.contains(Layers.SENTENCES)) {
      throw new IllegalArgumentException(MISSING_LAYER + Layers.SENTENCES);
    }
    if (!present.contains(Layers.TOKENS)) {
      throw new IllegalArgumentException(MISSING_LAYER + Layers.TOKENS);
    }
    if (!present.contains(Layers.POS_TAGS)) {
      throw new IllegalArgumentException(MISSING_LAYER + Layers.POS_TAGS);
    }
    if (!present.contains(Layers.ENTITIES)) {
      throw new IllegalArgumentException(MISSING_LAYER + Layers.ENTITIES);
    }
    final List<Annotation<String>> sentences = document.get(Layers.SENTENCES);
    final List<Annotation<String>> tokens = document.get(Layers.TOKENS);
    final List<Annotation<String>> tags = document.get(Layers.POS_TAGS);
    final List<Annotation<String>> entities = document.get(Layers.ENTITIES);
    if (tags.size() != tokens.size()) {
      throw new IllegalArgumentException("document needs aligned "
          + Layers.TOKENS + " and " + Layers.POS_TAGS + " layers");
    }
    if (tokens.isEmpty()) {
      return null;
    }
    if (sentences.isEmpty()) {
      throw new IllegalArgumentException(
          "document needs a non-empty " + Layers.SENTENCES + " layer");
    }
    final List<Annotation<String>> chunks = present.contains(ChunkerAnnotator.CHUNKS)
        ? document.get(ChunkerAnnotator.CHUNKS) : null;
    final List<Annotation<Phrase>> phrases = present.contains(ParserAnnotator.PHRASES)
        ? document.get(ParserAnnotator.PHRASES) : null;
    final String[] forms = new String[tokens.size()];
    final int[] sentenceOfToken = new int[tokens.size()];
    for (int t = 0, sentence = 0; t < forms.length; t++) {
      forms[t] = tokens.get(t).value();
      while (sentence < sentences.size() - 1
          && tokens.get(t).span().getStart() >= sentences.get(sentence).span().getEnd()) {
        sentence++;
      }
      sentenceOfToken[t] = sentence;
    }
    final List<Mention> mentions = MentionDetector.detect(personTypes, document.text(),
        sentences, tokens, tags, sentenceOfToken, entities, chunks, phrases);
    final List<Annotation<String>> speakers = present.contains(SPEAKERS)
        ? document.get(SPEAKERS) : null;
    return new SieveResolver(mentions, new Clusters(mentions), forms, sentenceOfToken,
        speakers, personTypes, neutralTypes, vectors);
  }

  /** {@inheritDoc} */
  @Override
  public Set<LayerKey<?>> requires() {
    return Set.of(Layers.SENTENCES, Layers.TOKENS, Layers.POS_TAGS, Layers.ENTITIES);
  }

  /** {@inheritDoc} */
  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(CHAINS);
  }
}
