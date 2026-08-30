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
import java.util.List;
import java.util.Set;

import opennlp.tools.coref.CorefLexicon.Pronoun;
import opennlp.tools.coref.Mention.Animacy;
import opennlp.tools.coref.Mention.Gender;
import opennlp.tools.coref.Mention.Number;
import opennlp.tools.coref.Mention.Person;
import opennlp.tools.document.Annotation;
import opennlp.tools.namefind.NameFinderAnnotator;
import opennlp.tools.parser.ParserAnnotator.Phrase;
import opennlp.tools.util.Span;
import opennlp.tools.util.StringUtil;

/**
 * Collects the candidate mentions of a document and computes their attributes.
 *
 * <p>Entity mentions come from the entity layer, nominal mentions from the noun phrases
 * of the parse layer when one is present or else from the noun phrase chunks of the
 * chunk layer, and pronoun mentions from the tokens tagged as pronouns whose form is a
 * personal pronoun. Nested and duplicate candidates are reconciled the way
 * OntoNotes annotates them: of the parse phrases sharing a head token only the largest
 * remains, a phrase headed by an entity widens that entity's mention to the full noun
 * phrase, a phrase coinciding with an entity or a lone pronoun adds nothing, and a
 * phrase cut by an entity boundary is dropped. A pleonastic {@code it}, one that
 * anticipates a clause rather than referring, is no mention.</p>
 */
final class MentionDetector {

  /** The tag prefix of nouns in the Penn Treebank tag set. */
  private static final String NOUN_TAG_PREFIX = "NN";

  /** The Penn Treebank tags of proper nouns. */
  private static final Set<String> PROPER_TAGS = Set.of("NNP", "NNPS");

  /** The Penn Treebank tags of plural nouns. */
  private static final Set<String> PLURAL_TAGS = Set.of("NNS", "NNPS");

  /** POS tags that mark pronoun tokens: Penn Treebank PRP and PRP$, Universal PRON. */
  private static final Set<String> PRONOUN_TAGS = Set.of("PRP", "PRP$", "PRON");

  /** Tags of the words that open a definite noun phrase. */
  private static final Set<String> DEFINITE_OPENER_TAGS = Set.of("DT", "PRP$", "WDT", "CD");

  /** The chunk type of noun phrases. */
  private static final String NOUN_PHRASE = "NP";

  /** The Penn Treebank tag of the possessive marker. */
  private static final String POSSESSIVE_TAG = "POS";

  /** The lowercased possessive marker, for tag sets that do not single it out. */
  private static final String POSSESSIVE_MARKER = "'s";

  /** The preposition that separates a name's head segment from a trailing qualifier. */
  private static final String OF = "of";

  /** The comma, which separates a head segment from an appositive or qualifier. */
  private static final String COMMA = ",";

  /** Entity type labels of organizations, whose number is left open. */
  private static final Set<String> ORGANIZATION_TYPES = Set.of("organization", "org");

  /** How many tokens after a pleonastic verb its complement may follow. */
  private static final int PLEONASTIC_WINDOW = 6;

  /** One mention candidate before its attributes are computed. */
  private static final class Candidate {
    Span span;
    final String kind;
    final int entity;
    final String type;
    int firstToken;
    int lastToken;
    /** The head token the parse names, or {@code -1} to find it by rule. */
    int head;

    Candidate(Span span, String kind, int entity, String type, int firstToken,
        int lastToken) {
      this.span = span;
      this.kind = kind;
      this.entity = entity;
      this.type = type;
      this.firstToken = firstToken;
      this.lastToken = lastToken;
      this.head = -1;
    }
  }

  /** One noun phrase of the parse layer with its token range and head. */
  private record NounPhrase(Span span, int first, int last, int head) {
  }

  private final Set<String> personTypes;
  private final CharSequence text;
  private final List<Annotation<String>> sentences;
  private final List<Annotation<String>> tokens;
  private final String[] lower;
  private final String[] tag;
  private final int[] sentenceOfToken;
  private final int[] sentenceEndToken;

  private MentionDetector(Set<String> personTypes, CharSequence text,
      List<Annotation<String>> sentences, List<Annotation<String>> tokens,
      List<Annotation<String>> tags, int[] sentenceOfToken) {
    this.personTypes = personTypes;
    this.text = text;
    this.sentences = sentences;
    this.tokens = tokens;
    this.sentenceOfToken = sentenceOfToken;
    lower = new String[tokens.size()];
    tag = new String[tokens.size()];
    sentenceEndToken = new int[tokens.size()];
    for (int t = 0; t < tokens.size(); t++) {
      lower[t] = StringUtil.toLowerCase(tokens.get(t).value());
      tag[t] = tags.get(t).value();
    }
    for (int t = tokens.size() - 1, end = tokens.size() - 1; t >= 0; t--) {
      if (t < tokens.size() - 1 && sentenceOfToken[t] != sentenceOfToken[t + 1]) {
        end = t;
      }
      sentenceEndToken[t] = end;
    }
  }

  /**
   * Detects the mentions of a document.
   *
   * @param personTypes The lowercased entity types of people.
   * @param text The document text.
   * @param sentences The sentence layer. Must not be empty when tokens are present.
   * @param tokens The token layer.
   * @param tags The POS tag layer, aligned with the tokens.
   * @param sentenceOfToken The sentence index of every token.
   * @param entities The entity layer.
   * @param chunks The chunk layer, or {@code null} when the document carries none.
   * @param phrases The parse layer, or {@code null} when the document carries none;
   *                when present it supplies the noun phrases and the chunks are unused.
   * @return The mentions in text order, wider spans first at a shared start. Never
   *         {@code null}.
   */
  static List<Mention> detect(Set<String> personTypes, CharSequence text,
      List<Annotation<String>> sentences, List<Annotation<String>> tokens,
      List<Annotation<String>> tags, int[] sentenceOfToken,
      List<Annotation<String>> entities, List<Annotation<String>> chunks,
      List<Annotation<Phrase>> phrases) {
    return new MentionDetector(personTypes, text, sentences, tokens, tags, sentenceOfToken)
        .detect(entities, chunks, phrases);
  }

  private List<Mention> detect(List<Annotation<String>> entities,
      List<Annotation<String>> chunks, List<Annotation<Phrase>> phrases) {
    final List<Candidate> candidates = new ArrayList<>();
    for (int e = 0; e < entities.size(); e++) {
      final Span span = entities.get(e).span();
      candidates.add(new Candidate(span, CorefMention.KIND_ENTITY, e,
          StringUtil.toLowerCase(entities.get(e).value()), firstToken(span),
          lastToken(span)));
    }
    final int entityCount = candidates.size();
    if (phrases != null) {
      for (final NounPhrase phrase : maximalNounPhrases(phrases)) {
        addNounPhrase(phrase, candidates, entityCount);
      }
    } else if (chunks != null) {
      for (final NounPhrase phrase : chunkNounPhrases(chunks)) {
        addNounPhrase(phrase, candidates, entityCount);
      }
    }
    addPronouns(candidates, entityCount);
    candidates.sort((a, b) -> a.span.getStart() != b.span.getStart()
        ? Integer.compare(a.span.getStart(), b.span.getStart())
        : Integer.compare(b.span.getEnd(), a.span.getEnd()));
    final List<Mention> mentions = new ArrayList<>(candidates.size());
    for (final Candidate candidate : candidates) {
      mentions.add(mention(candidate));
    }
    return mentions;
  }

  /**
   * Reads the noun phrase chunks, joining a chunk that opens with a possessive marker to
   * the noun phrase chunk right before it, since base chunking splits {@code Norton 's
   * capital} into the possessor and the possessed while the mention is the whole phrase.
   */
  private List<NounPhrase> chunkNounPhrases(List<Annotation<String>> chunks) {
    final List<NounPhrase> phrases = new ArrayList<>();
    int pendingFirst = -1;
    int pendingLast = -1;
    for (final Annotation<String> chunk : chunks) {
      final boolean nounPhrase = NOUN_PHRASE.equals(chunk.value());
      final int first = nounPhrase ? firstToken(chunk.span()) : -1;
      final int last = nounPhrase ? lastToken(chunk.span()) : -1;
      final boolean valid = nounPhrase && first >= 0 && last >= first;
      if (valid && pendingFirst >= 0 && first == pendingLast + 1
          && (POSSESSIVE_TAG.equals(tag[first]) || POSSESSIVE_MARKER.equals(lower[first]))) {
        pendingLast = last;
        continue;
      }
      if (pendingFirst >= 0) {
        phrases.add(nounPhrase(pendingFirst, pendingLast));
      }
      pendingFirst = valid ? first : -1;
      pendingLast = valid ? last : -1;
    }
    if (pendingFirst >= 0) {
      phrases.add(nounPhrase(pendingFirst, pendingLast));
    }
    return phrases;
  }

  /** Builds a noun phrase over a token range, finding its head by rule. */
  private NounPhrase nounPhrase(int first, int last) {
    return new NounPhrase(new Span(tokens.get(first).span().getStart(),
        tokens.get(last).span().getEnd()), first, last, headToken(first, last));
  }

  /**
   * Reads the noun phrases of the parse layer, keeping of the phrases that share a head
   * token only the largest, the one OntoNotes annotates.
   */
  private List<NounPhrase> maximalNounPhrases(List<Annotation<Phrase>> phrases) {
    final List<NounPhrase> nounPhrases = new ArrayList<>();
    for (final Annotation<Phrase> phrase : phrases) {
      if (!NOUN_PHRASE.equals(phrase.value().label())) {
        continue;
      }
      final int first = firstToken(phrase.span());
      final int last = lastToken(phrase.span());
      int head = firstToken(phrase.value().head());
      if (first < 0 || last < first || head < first || head > last) {
        continue;
      }
      // Head rules make the possessive marker the head of a possessive phrase; the
      // mention's head is the noun the marker follows.
      if (head > first && (POSSESSIVE_TAG.equals(tag[head]) || POSSESSIVE_MARKER.equals(lower[head]))) {
        head = headToken(first, head - 1);
      }
      nounPhrases.add(new NounPhrase(phrase.span(), first, last, head));
    }
    final List<NounPhrase> maximal = new ArrayList<>();
    for (final NounPhrase phrase : nounPhrases) {
      boolean widest = true;
      for (final NounPhrase other : nounPhrases) {
        if (other != phrase && other.head() == phrase.head()
            && other.span().contains(phrase.span()) && !other.span().equals(phrase.span())) {
          widest = false;
          break;
        }
      }
      if (widest) {
        maximal.add(phrase);
      }
    }
    return maximal;
  }

  /**
   * Reconciles one noun phrase with the entity candidates it touches and adds it as a
   * nominal mention when it stands on its own.
   */
  private void addNounPhrase(NounPhrase phrase, List<Candidate> candidates,
      int entityCount) {
    final Span span = phrase.span();
    final int first = phrase.first();
    final int last = phrase.last();
    Candidate headedBy = null;
    int touching = 0;
    for (int e = 0; e < entityCount; e++) {
      final Candidate entity = candidates.get(e);
      if (entity.lastToken < first || entity.firstToken > last || entity.firstToken < 0) {
        continue;
      }
      touching++;
      if (entity.firstToken == first && entity.lastToken == last) {
        return;
      }
      if (entity.firstToken < first || entity.lastToken > last) {
        return;
      }
      if (entity.lastToken == phrase.head()) {
        headedBy = entity;
      }
    }
    // A phrase headed by an entity is that entity's full mention, unless the phrase
    // coordinates it with others, in which case the coordination is a mention of its own.
    if (headedBy != null && !(touching > 1 && coordinated(first, last))) {
      headedBy.span = span;
      headedBy.firstToken = first;
      headedBy.lastToken = last;
      headedBy.head = phrase.head();
      return;
    }
    if (first == last && CorefLexicon.pronoun(lower[first]) != null) {
      return;
    }
    addPossessors(first, last, candidates, entityCount);
    if (!tag[phrase.head()].startsWith(NOUN_TAG_PREFIX)) {
      return;
    }
    final Candidate nominal = new Candidate(span, CorefMention.KIND_NOMINAL,
        CorefMention.NO_ENTITY, null, first, last);
    nominal.head = phrase.head();
    candidates.add(nominal);
  }

  /**
   * Adds the possessor inside a noun phrase as a mention of its own, {@code Norton 's}
   * in {@code Norton 's capital}: the tokens up to a possessive marker, widening an
   * entity that ends right before the marker to include it, as OntoNotes annotates.
   */
  private void addPossessors(int first, int last, List<Candidate> candidates,
      int entityCount) {
    for (int t = first + 1; t < last; t++) {
      if (!POSSESSIVE_TAG.equals(tag[t]) && !POSSESSIVE_MARKER.equals(lower[t])) {
        continue;
      }
      final int head = headToken(first, t - 1);
      boolean widened = false;
      for (int e = 0; e < entityCount; e++) {
        final Candidate entity = candidates.get(e);
        if (entity.lastToken == t - 1 && entity.firstToken >= first) {
          entity.lastToken = t;
          entity.span = new Span(entity.span.getStart(), tokens.get(t).span().getEnd());
          widened = true;
        } else if (entity.firstToken <= t && entity.lastToken >= t) {
          widened = true;
        }
      }
      if (widened || !tag[head].startsWith(NOUN_TAG_PREFIX)
          || CorefLexicon.pronoun(lower[head]) != null) {
        continue;
      }
      final Candidate possessor = new Candidate(
          new Span(tokens.get(first).span().getStart(), tokens.get(t).span().getEnd()),
          CorefMention.KIND_NOMINAL, CorefMention.NO_ENTITY, null, first, t);
      possessor.head = head;
      candidates.add(possessor);
    }
  }

  /** {@return whether a token range holds a coordinating {@code and}} */
  private boolean coordinated(int first, int last) {
    for (int t = first; t <= last; t++) {
      if ("and".equals(lower[t])) {
        return true;
      }
    }
    return false;
  }

  /** Adds every personal pronoun token no entity covers as a pronoun mention. */
  private void addPronouns(List<Candidate> candidates, int entityCount) {
    for (int t = 0; t < tokens.size(); t++) {
      if (!PRONOUN_TAGS.contains(tag[t])) {
        continue;
      }
      if (CorefLexicon.pronoun(lower[t]) == null) {
        continue;
      }
      final Span span = tokens.get(t).span();
      if (coveredByEntity(span, candidates, entityCount) || pleonastic(t)) {
        continue;
      }
      candidates.add(new Candidate(span, CorefMention.KIND_PRONOUN,
          CorefMention.NO_ENTITY, null, t, t));
    }
  }

  private boolean coveredByEntity(Span span, List<Candidate> candidates, int entityCount) {
    for (int e = 0; e < entityCount; e++) {
      if (candidates.get(e).span.contains(span)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Recognizes a pleonastic {@code it}: one followed, possibly after an adverb, by a
   * verb such as {@code is} or {@code seems} and within a few tokens by a complement
   * opener such as {@code that} or {@code to}, all inside the sentence.
   */
  private boolean pleonastic(int t) {
    if (!"it".equals(lower[t])) {
      return false;
    }
    final int end = sentenceEndToken[t];
    int verb = t + 1;
    if (verb <= end && tag[verb].startsWith("RB")) {
      verb++;
    }
    if (verb > end || !CorefLexicon.pleonasticVerb(lower[verb])) {
      return false;
    }
    for (int k = verb + 1; k <= end && k <= verb + PLEONASTIC_WINDOW; k++) {
      if (CorefLexicon.pleonasticComplement(lower[k])) {
        return true;
      }
    }
    return false;
  }

  /** Computes a candidate's attributes. */
  private Mention mention(Candidate candidate) {
    final int first = candidate.firstToken;
    final int last = candidate.lastToken;
    if (first < 0 || last < first) {
      return unaligned(candidate);
    }
    final List<String> words = new ArrayList<>(last - first + 1);
    for (int t = first; t <= last; t++) {
      words.add(lower[t]);
    }
    final int headToken = candidate.head >= 0 ? candidate.head : headToken(first, last);
    final String head = lower[headToken];
    final String headPrefix = join(words, 0, headToken - first + 1);
    final int end = POSSESSIVE_TAG.equals(tag[last]) || POSSESSIVE_MARKER.equals(lower[last])
        || "'".equals(lower[last]) ? words.size() - 1 : words.size();
    final String normalized = join(words, 0, Math.max(end, 1));
    final boolean pronoun = CorefMention.KIND_PRONOUN.equals(candidate.kind);
    final boolean entity = candidate.entity != CorefMention.NO_ENTITY;
    final boolean proper = entity || PROPER_TAGS.contains(tag[headToken]);
    // An entity of the unknown type gives no attribute evidence: it is neither a person
    // nor a thing, and its name is not read for a gender, so every pronoun class may
    // still reach it.
    final String type = NameFinderAnnotator.UNTYPED.equals(candidate.type)
        ? null : candidate.type;
    final boolean person = type != null && personTypes.contains(type);

    Number number = Number.UNKNOWN;
    Gender gender = Gender.UNKNOWN;
    Animacy animacy = Animacy.UNKNOWN;
    Person grammaticalPerson = Person.THIRD;
    if (pronoun) {
      final Pronoun form = CorefLexicon.pronoun(head);
      number = form.number();
      gender = form.gender();
      animacy = form.animacy();
      grammaticalPerson = form.person();
    } else {
      if (coordinated(first, last)) {
        number = Number.PLURAL;
      } else if (type != null && ORGANIZATION_TYPES.contains(type)) {
        number = Number.UNKNOWN;
      } else if (PLURAL_TAGS.contains(tag[headToken])) {
        number = Number.PLURAL;
      } else if (tag[headToken].startsWith(NOUN_TAG_PREFIX)) {
        number = Number.SINGULAR;
      }
      if (person) {
        gender = nameGender(first, last);
        animacy = Animacy.ANIMATE;
      } else if (type != null) {
        gender = Gender.NEUTRAL;
        animacy = Animacy.INANIMATE;
      } else if (proper && !entity) {
        gender = nameGender(first, last);
        if (gender != Gender.UNKNOWN) {
          animacy = Animacy.ANIMATE;
        }
      } else if (!entity) {
        gender = CorefLexicon.nounGender(head);
        if (CorefLexicon.animateNoun(head)) {
          animacy = Animacy.ANIMATE;
        }
      }
    }
    final boolean indefinite = !pronoun && !proper && indefinite(first, headToken);
    return new Mention(candidate.span, candidate.kind, candidate.entity, type,
        sentenceOfToken[first], first, last, List.copyOf(words), head, headPrefix,
        normalized, proper, indefinite, number, gender, animacy, grammaticalPerson);
  }

  /**
   * Builds the mention of an entity whose span aligns with no token, from the
   * whitespace-delimited words of its text; it has no tag evidence, so its attributes
   * beyond the entity type stay unknown.
   */
  private Mention unaligned(Candidate candidate) {
    final Span span = candidate.span;
    final String normalized = StringUtil.toLowerCase(
        text.subSequence(span.getStart(), span.getEnd()).toString());
    final List<String> words = new ArrayList<>();
    int start = -1;
    for (int i = 0; i < normalized.length(); i++) {
      if (StringUtil.isWhitespace(normalized.charAt(i))) {
        if (start >= 0) {
          words.add(normalized.substring(start, i));
          start = -1;
        }
      } else if (start < 0) {
        start = i;
      }
    }
    if (start >= 0) {
      words.add(normalized.substring(start));
    }
    final int of = words.indexOf(OF);
    final int headIndex = of > 0 ? of - 1 : words.size() - 1;
    final String head = words.isEmpty() ? null : words.get(headIndex);
    final String type = NameFinderAnnotator.UNTYPED.equals(candidate.type)
        ? null : candidate.type;
    final boolean person = type != null && personTypes.contains(type);
    return new Mention(span, candidate.kind, candidate.entity, type, sentenceOf(span),
        -1, -1, List.copyOf(words), head, join(words, 0, headIndex + 1),
        join(words, 0, words.size()), true, false, Number.UNKNOWN,
        person ? Gender.UNKNOWN : type != null ? Gender.NEUTRAL : Gender.UNKNOWN,
        person ? Animacy.ANIMATE : type != null ? Animacy.INANIMATE : Animacy.UNKNOWN,
        Person.THIRD);
  }

  /**
   * Finds the head token of a token range: the last noun before the first {@code of}
   * or comma past the first token, leaving out a trailing possessive marker, and
   * falling back to the last token of that prefix.
   */
  private int headToken(int first, int last) {
    int stop = last;
    for (int t = first + 1; t <= last; t++) {
      if (OF.equals(lower[t]) || COMMA.equals(lower[t])) {
        stop = t - 1;
        break;
      }
    }
    if (stop > first && (POSSESSIVE_TAG.equals(tag[stop]) || POSSESSIVE_MARKER.equals(lower[stop]))) {
      stop--;
    }
    for (int t = stop; t >= first; t--) {
      if (tag[t].startsWith(NOUN_TAG_PREFIX)) {
        return t;
      }
    }
    return stop;
  }

  /** Reads the gender a person name implies from its title or first name. */
  private Gender nameGender(int first, int last) {
    int t = first;
    if (CorefLexicon.title(lower[t])) {
      final Gender titled = CorefLexicon.nounGender(lower[t]);
      if (titled != Gender.UNKNOWN || t == last) {
        return titled;
      }
      t++;
    }
    return CorefLexicon.firstNameGender(lower[t]);
  }

  /**
   * Checks whether a nominal mention is indefinite or generic: opened by an indefinite
   * word, or a bare common noun, plural or singular, without a definite opener.
   */
  private boolean indefinite(int first, int headToken) {
    if (CorefLexicon.indefiniteWord(lower[first])) {
      return true;
    }
    return tag[headToken].startsWith(NOUN_TAG_PREFIX)
        && !PROPER_TAGS.contains(tag[headToken])
        && !DEFINITE_OPENER_TAGS.contains(tag[first])
        && !"the".equals(lower[first]);
  }

  private String join(List<String> words, int from, int to) {
    final StringBuilder joined = new StringBuilder();
    for (int i = from; i < to && i < words.size(); i++) {
      if (i > from) {
        joined.append(' ');
      }
      joined.append(words.get(i));
    }
    return joined.toString();
  }

  /** {@return the index of the first token a span intersects, or {@code -1}} */
  private int firstToken(Span span) {
    int low = 0;
    int high = tokens.size();
    while (low < high) {
      final int mid = (low + high) >>> 1;
      if (tokens.get(mid).span().getEnd() <= span.getStart()) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    return low < tokens.size() && tokens.get(low).span().getStart() < span.getEnd()
        ? low : -1;
  }

  /** {@return the index of the last token a span intersects, or {@code -1}} */
  private int lastToken(Span span) {
    int low = 0;
    int high = tokens.size();
    while (low < high) {
      final int mid = (low + high) >>> 1;
      if (tokens.get(mid).span().getStart() < span.getEnd()) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    return low > 0 && tokens.get(low - 1).span().getEnd() > span.getStart() ? low - 1 : -1;
  }

  /** {@return the index of the sentence a span starts in} */
  private int sentenceOf(Span span) {
    for (int s = 0; s < sentences.size(); s++) {
      if (span.getStart() < sentences.get(s).span().getEnd()) {
        return s;
      }
    }
    return sentences.size() - 1;
  }
}
