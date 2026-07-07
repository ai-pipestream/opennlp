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
package opennlp.wordnet;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import opennlp.tools.commons.ThreadSafe;
import opennlp.tools.lemmatizer.Lemmatizer;
import opennlp.tools.wordnet.WordNetLexicon;
import opennlp.tools.wordnet.WordNetPos;

/**
 * A clean-room implementation of the documented Morphy algorithm as a {@link Lemmatizer}:
 * exception-list lookup first, then the per-part-of-speech iterative detachment rules, with
 * every rule-derived candidate validated against a {@link WordNetLexicon} before it is
 * returned.
 *
 * <p><b>Algorithm.</b> For each token the part-of-speech tag is mapped to a
 * {@link WordNetPos}; the token is folded (lowercase with the root locale, underscore as
 * space); the {@link MorphyExceptions exception lists} are consulted and a hit is returned
 * directly, without lexicon validation, because the lists themselves are authoritative for
 * irregular forms; otherwise the folded token itself, when the lexicon contains it, and every
 * detachment-rule result the lexicon confirms become the candidate lemmas, in rule order. The
 * rules are the documented Morphy suffix substitutions: for nouns {@code -s}, {@code -ses},
 * {@code -xes}, {@code -zes}, {@code -ches}, {@code -shes}, {@code -men}, {@code -ies}; for
 * verbs {@code -s}, {@code -ies}, {@code -es}, {@code -ed}, {@code -ing} (with their {@code e}
 * restorations); for adjectives {@code -er} and {@code -est} (plain and {@code e}-restoring);
 * and none for adverbs, which rely on the exception list alone. Returned lemmas are in the
 * lexicon's folded canonical form (lowercase, spaces in multiword lemmas).</p>
 *
 * <p><b>Tag mapping.</b> Tags map to WordNet parts of speech by their conventional Penn
 * Treebank prefixes: {@code N} to noun, {@code V} to verb, {@code J} to adjective, and
 * {@code R} to adverb, case-insensitively. The names {@code ADJ} and {@code ADV} and the
 * WordNet letters {@code n}, {@code v}, {@code a}, {@code r}, and {@code s} (satellite,
 * treated as adjective) are also accepted. Anything else, including Penn tags for closed
 * classes such as {@code DT} or {@code MD}, maps to no part of speech and yields the
 * unknown-word result.</p>
 *
 * <p><b>Unknown words.</b> Following the convention of
 * {@code opennlp.tools.lemmatizer.DictionaryLemmatizer}, a token with no lemma yields the
 * string {@code "O"} from {@link #lemmatize(String[], String[])} and a singleton
 * {@code ["O"]} from {@link #lemmatize(List, List)}.</p>
 *
 * <p>No lexicon or exception data is bundled: point {@link WndbReader} and
 * {@link MorphyExceptions#load(java.nio.file.Path) MorphyExceptions} at a WordNet database
 * directory you
 * downloaded, or combine {@link WnLmfReader} with a downloaded exception-list directory. Both
 * inputs are required because rule-derived candidates are meaningless without a lexicon to
 * validate them against; an exception-only mode would be a misleading half-feature, so it does
 * not exist. Bundling the permissively licensed data is a recorded follow-up.</p>
 *
 * <p>Instances are immutable and safe for concurrent use once constructed with a loaded
 * lexicon and exception lists.</p>
 */
@ThreadSafe
public final class MorphyLemmatizer implements Lemmatizer {

  /** The unknown-word output, matching the DictionaryLemmatizer convention. */
  private static final String UNKNOWN = "O";

  private static final String[][] NOUN_RULES = {
      {"s", ""}, {"ses", "s"}, {"xes", "x"}, {"zes", "z"},
      {"ches", "ch"}, {"shes", "sh"}, {"men", "man"}, {"ies", "y"},
  };

  private static final String[][] VERB_RULES = {
      {"s", ""}, {"ies", "y"}, {"es", "e"}, {"es", ""},
      {"ed", "e"}, {"ed", ""}, {"ing", "e"}, {"ing", ""},
  };

  private static final String[][] ADJECTIVE_RULES = {
      {"er", ""}, {"est", ""}, {"er", "e"}, {"est", "e"},
  };

  private static final String[][] NO_RULES = {};

  private final WordNetLexicon lexicon;
  private final MorphyExceptions exceptions;

  /**
   * Creates a Morphy lemmatizer over a loaded lexicon and exception lists.
   *
   * @param lexicon    The lexicon rule candidates are validated against. Must not be
   *                   {@code null}.
   * @param exceptions The irregular-form exception lists. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code lexicon} or {@code exceptions} is
   *     {@code null}.
   */
  public MorphyLemmatizer(WordNetLexicon lexicon, MorphyExceptions exceptions) {
    if (lexicon == null) {
      throw new IllegalArgumentException("Lexicon must not be null");
    }
    if (exceptions == null) {
      throw new IllegalArgumentException("Exceptions must not be null");
    }
    this.lexicon = lexicon;
    this.exceptions = exceptions;
  }

  @Override
  public String[] lemmatize(String[] toks, String[] tags) {
    if (toks == null || tags == null) {
      throw new IllegalArgumentException("Toks and tags must not be null");
    }
    if (toks.length != tags.length) {
      throw new IllegalArgumentException("Toks and tags must have the same length, got "
          + toks.length + " and " + tags.length);
    }
    final String[] lemmas = new String[toks.length];
    for (int i = 0; i < toks.length; i++) {
      final List<String> candidates = lemmasOf(toks[i], tags[i]);
      lemmas[i] = candidates.isEmpty() ? UNKNOWN : candidates.get(0);
    }
    return lemmas;
  }

  @Override
  public List<List<String>> lemmatize(List<String> toks, List<String> tags) {
    if (toks == null || tags == null) {
      throw new IllegalArgumentException("Toks and tags must not be null");
    }
    if (toks.size() != tags.size()) {
      throw new IllegalArgumentException("Toks and tags must have the same size, got "
          + toks.size() + " and " + tags.size());
    }
    final List<List<String>> lemmas = new ArrayList<>(toks.size());
    for (int i = 0; i < toks.size(); i++) {
      final List<String> candidates = lemmasOf(toks.get(i), tags.get(i));
      lemmas.add(candidates.isEmpty() ? List.of(UNKNOWN) : candidates);
    }
    return lemmas;
  }

  // All lemmas of one token, most preferred first; empty when the word is unknown.
  private List<String> lemmasOf(String token, String tag) {
    if (token == null || tag == null) {
      throw new IllegalArgumentException("Tokens and tags must not contain null elements");
    }
    final WordNetPos pos = posFromTag(tag);
    if (pos == null) {
      return List.of();
    }
    final String folded = MorphyExceptions.fold(token);
    final List<String> irregular = exceptions.lookup(folded, pos);
    if (!irregular.isEmpty()) {
      return irregular;
    }
    final List<String> candidates = new ArrayList<>(2);
    if (lexicon.contains(folded, pos)) {
      candidates.add(folded);
    }
    for (final String[] rule : rulesFor(pos)) {
      final String suffix = rule[0];
      if (folded.length() > suffix.length() && folded.endsWith(suffix)) {
        final String candidate =
            folded.substring(0, folded.length() - suffix.length()) + rule[1];
        if (!candidates.contains(candidate) && lexicon.contains(candidate, pos)) {
          candidates.add(candidate);
        }
      }
    }
    return candidates;
  }

  private static String[][] rulesFor(WordNetPos pos) {
    return switch (pos) {
      case NOUN -> NOUN_RULES;
      case VERB -> VERB_RULES;
      case ADJECTIVE -> ADJECTIVE_RULES;
      case ADVERB -> NO_RULES;
    };
  }

  // The documented tag mapping; package-private so tests can pin it directly.
  static WordNetPos posFromTag(String tag) {
    if (tag.isEmpty()) {
      return null;
    }
    final String upper = tag.toUpperCase(Locale.ROOT);
    if (upper.startsWith("ADJ")) {
      return WordNetPos.ADJECTIVE;
    }
    if (upper.startsWith("ADV")) {
      return WordNetPos.ADVERB;
    }
    return switch (upper.charAt(0)) {
      case 'N' -> WordNetPos.NOUN;
      case 'V' -> WordNetPos.VERB;
      case 'J', 'A', 'S' -> WordNetPos.ADJECTIVE;
      case 'R' -> WordNetPos.ADVERB;
      default -> null;
    };
  }
}
