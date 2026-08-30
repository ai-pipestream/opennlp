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
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.IntPredicate;

import opennlp.tools.coref.Mention.Gender;

/**
 * Runs the precision-ranked sieves of
 * <a href="https://aclanthology.org/J13-4004/">Lee et al. (Computational Linguistics
 * 2013), "Deterministic Coreference Resolution Based on Entity-Centric, Precision-Ranked
 * Rules"</a> over the mentions of one document.
 *
 * <p>Every sieve visits the mentions in text order and considers only a mention that is
 * still the first of its cluster as an anaphor, so a mention linked by an earlier, more
 * precise sieve is never re-resolved. Candidate antecedents are ordered by salience:
 * earlier mentions of the same sentence left to right, then the preceding sentences
 * nearest first, each read left to right for a nominal anaphor and right to left for a
 * pronoun, which favors proximity. The first candidate whose cluster passes the sieve's
 * test is linked, comparing the accumulated attributes of both clusters rather than the
 * two mentions alone. No sieve links a mention to one that contains it, and no sieve
 * merges clusters whose known entity types differ.</p>
 */
final class SieveResolver {

  /** How many sentences back a pronoun may find its antecedent. */
  private static final int PRONOUN_WINDOW = 3;

  /** No limit on how many sentences back an antecedent may lie. */
  private static final int UNLIMITED = Integer.MAX_VALUE;

  /** Tests whether the anaphor may link to the candidate antecedent. */
  private interface Link {
    boolean test(int antecedent, int anaphor);
  }

  private final List<Mention> mentions;
  private final Clusters clusters;
  private final String[] forms;
  private final Set<String> personTypes;
  private final Set<String> neutralTypes;
  private final List<List<Integer>> bySentence;

  /**
   * Initializes the resolver.
   *
   * @param mentions The mentions in text order.
   * @param clusters The clusters over the mentions, initially all singletons.
   * @param forms The original token forms.
   * @param personTypes The lowercased entity types gendered pronouns may resolve to.
   * @param neutralTypes The lowercased entity types neutral pronouns may resolve to.
   */
  SieveResolver(List<Mention> mentions, Clusters clusters, String[] forms,
      Set<String> personTypes, Set<String> neutralTypes) {
    this.mentions = mentions;
    this.clusters = clusters;
    this.forms = forms;
    this.personTypes = personTypes;
    this.neutralTypes = neutralTypes;
    bySentence = new ArrayList<>();
    for (int i = 0; i < mentions.size(); i++) {
      final int sentence = mentions.get(i).sentence();
      while (bySentence.size() <= sentence) {
        bySentence.add(new ArrayList<>());
      }
      bySentence.get(sentence).add(i);
    }
  }

  /** Runs every sieve in precision order. */
  void resolve() {
    final IntPredicate nominal = j -> !mentions.get(j).pronoun()
        && !mentions.get(j).indefinite() && mentions.get(j).head() != null;
    pass(nominal, UNLIMITED, this::exactMatch);
    pass(nominal, UNLIMITED, this::relaxedStringMatch);
    pass(nominal, UNLIMITED, (i, j) -> acronym(i, j) || personName(i, j));
    pass(nominal, UNLIMITED, (i, j) -> strictHeadMatch(i, j, true, true));
    pass(nominal, UNLIMITED, (i, j) -> strictHeadMatch(i, j, true, false));
    pass(nominal, UNLIMITED, (i, j) -> strictHeadMatch(i, j, false, true));
    pass(nominal, UNLIMITED, this::properHeadMatch);
    pass(nominal, UNLIMITED, this::relaxedHeadMatch);
    pass(j -> mentions.get(j).pronoun(), PRONOUN_WINDOW, this::pronounMatch);
  }

  /**
   * Runs one sieve: for each anaphor that heads its own cluster, links the first
   * candidate antecedent in salience order that passes the link test.
   */
  private void pass(IntPredicate anaphor, int window, Link link) {
    for (int j = 0; j < mentions.size(); j++) {
      if (clusters.find(j) != j || !anaphor.test(j)) {
        continue;
      }
      final Mention mention = mentions.get(j);
      final int limit = reflexive(mention) ? 0 : window;
      for (final int i : candidates(j, mention, limit)) {
        if (clusters.find(i) == clusters.find(j) || mention.nests(mentions.get(i))
            || !clusters.typesCompatible(i, j)) {
          continue;
        }
        if (link.test(i, j) && clusters.union(i, j)) {
          break;
        }
      }
    }
  }

  /** {@return whether a pronoun is reflexive, so its antecedent shares its sentence} */
  private static boolean reflexive(Mention mention) {
    return mention.pronoun()
        && (mention.head().endsWith("self") || mention.head().endsWith("selves"));
  }

  /**
   * Orders the candidate antecedents of an anaphor: the same sentence left to right,
   * then each preceding sentence within the window, nearest first, read right to left
   * for a pronoun and left to right otherwise.
   */
  private List<Integer> candidates(int j, Mention mention, int window) {
    final List<Integer> ordered = new ArrayList<>();
    final int sentence = mention.sentence();
    for (final int i : bySentence.get(sentence)) {
      if (i < j) {
        ordered.add(i);
      }
    }
    final int farthest = window == UNLIMITED ? 0 : Math.max(0, sentence - window);
    for (int s = sentence - 1; s >= farthest; s--) {
      final List<Integer> earlier = bySentence.get(s);
      if (mention.pronoun()) {
        for (int k = earlier.size() - 1; k >= 0; k--) {
          ordered.add(earlier.get(k));
        }
      } else {
        ordered.addAll(earlier);
      }
    }
    return ordered;
  }

  /** Links mentions whose normalized text is identical. */
  private boolean exactMatch(int i, int j) {
    final Mention antecedent = mentions.get(i);
    return !antecedent.pronoun()
        && !antecedent.normalized().isEmpty()
        && antecedent.normalized().equals(mentions.get(j).normalized());
  }

  /** Links mentions whose text up to and including the head is identical. */
  private boolean relaxedStringMatch(int i, int j) {
    final Mention antecedent = mentions.get(i);
    return !antecedent.pronoun()
        && !antecedent.headPrefix().isEmpty()
        && antecedent.headPrefix().equals(mentions.get(j).headPrefix());
  }

  /** Links a proper name to its acronym in either direction. */
  private boolean acronym(int i, int j) {
    final Mention antecedent = mentions.get(i);
    final Mention anaphor = mentions.get(j);
    return antecedent.proper() && anaphor.proper()
        && (acronymOf(anaphor, antecedent) || acronymOf(antecedent, anaphor));
  }

  /**
   * Checks whether a one-token uppercase mention spells the initials of the capitalized
   * words of a multi-word mention.
   */
  private boolean acronymOf(Mention candidate, Mention name) {
    if (candidate.firstToken() < 0 || candidate.firstToken() != candidate.lastToken()
        || name.firstToken() < 0 || name.lastToken() - name.firstToken() < 1) {
      return false;
    }
    final String acronym = forms[candidate.firstToken()];
    if (acronym.length() < 2) {
      return false;
    }
    for (int c = 0; c < acronym.length(); c++) {
      if (!Character.isUpperCase(acronym.charAt(c))) {
        return false;
      }
    }
    final StringBuilder initials = new StringBuilder();
    for (int t = name.firstToken(); t <= name.lastToken(); t++) {
      final String form = forms[t];
      if (!form.isEmpty() && Character.isUpperCase(form.charAt(0))) {
        initials.append(form.charAt(0));
      }
    }
    return initials.length() >= 2 && initials.toString().equals(acronym);
  }

  /**
   * Links a person's single-word name to a longer name of the same person that starts
   * or ends with it, so a surname or first name finds the full name. Both mentions must
   * be typed as people; place and organization names, where a shared word need not mean
   * a shared referent, never take this link.
   */
  private boolean personName(int i, int j) {
    final Mention antecedent = mentions.get(i);
    final Mention anaphor = mentions.get(j);
    if (!antecedent.proper() || !anaphor.proper()
        || !personType(antecedent.type()) || !personType(anaphor.type())
        || !clusters.numbersAgree(i, j)) {
      return false;
    }
    final List<String> shorter = nameWords(antecedent.words().size() <= anaphor.words().size()
        ? antecedent : anaphor);
    final List<String> longer = nameWords(antecedent.words().size() <= anaphor.words().size()
        ? anaphor : antecedent);
    return shorter.size() == 1 && longer.size() >= 2
        && (longer.get(0).equals(shorter.get(0))
            || longer.get(longer.size() - 1).equals(shorter.get(0)));
  }

  private boolean personType(String type) {
    return type != null && personTypes.contains(type);
  }

  /** {@return a name's words without leading titles} */
  private static List<String> nameWords(Mention mention) {
    final List<String> words = mention.words();
    int start = 0;
    while (start < words.size() - 1 && CorefLexicon.title(words.get(start))) {
      start++;
    }
    return words.subList(start, words.size());
  }

  /**
   * Links an anaphor whose head matches a head in the candidate's cluster, optionally
   * requiring that every content word of the anaphor's cluster appears in the
   * candidate's cluster and that every modifier of the anaphor appears in the candidate
   * mention.
   */
  private boolean strictHeadMatch(int i, int j, boolean wordInclusion,
      boolean compatibleModifiers) {
    final Mention antecedent = mentions.get(i);
    final Mention anaphor = mentions.get(j);
    if (antecedent.pronoun() || !clusters.heads(i).contains(anaphor.head())) {
      return false;
    }
    if (wordInclusion && !clusters.words(i).containsAll(clusters.words(j))) {
      return false;
    }
    return !compatibleModifiers || modifiersAppearIn(anaphor, antecedent);
  }

  /** Checks whether every non-stop word before the anaphor's head is in the antecedent. */
  private static boolean modifiersAppearIn(Mention anaphor, Mention antecedent) {
    final List<String> words = anaphor.words();
    final Set<String> candidateWords = new HashSet<>(antecedent.words());
    for (final String word : words) {
      if (word.equals(anaphor.head())) {
        break;
      }
      if (!CorefLexicon.stopWord(word) && !candidateWords.contains(word)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Links two proper names with the same head word, agreeing number, and no differing
   * numeric modifiers, so {@code President Obama} finds {@code Barack Obama}. A head
   * that names a kind of place or institution, such as {@code University}, is shared by
   * many distinct names, so there the modifiers of one side must appear in the other:
   * {@code Harvard University} never finds {@code Stanford University}.
   */
  private boolean properHeadMatch(int i, int j) {
    final Mention antecedent = mentions.get(i);
    final Mention anaphor = mentions.get(j);
    if (!antecedent.proper() || !anaphor.proper()
        || !antecedent.head().equals(anaphor.head())
        || !clusters.numbersAgree(i, j)
        || numbersDiffer(antecedent, anaphor)) {
      return false;
    }
    return !CorefLexicon.compoundNameHead(anaphor.head())
        || modifiersAppearIn(anaphor, antecedent)
        || modifiersAppearIn(antecedent, anaphor);
  }

  private static boolean numbersDiffer(Mention a, Mention b) {
    final Set<String> numbersA = numerals(a);
    final Set<String> numbersB = numerals(b);
    return !numbersA.isEmpty() && !numbersB.isEmpty() && !numbersA.equals(numbersB);
  }

  private static Set<String> numerals(Mention mention) {
    final Set<String> numerals = new HashSet<>();
    for (final String word : mention.words()) {
      if (!word.isEmpty() && Character.isDigit(word.charAt(0))) {
        numerals.add(word);
      }
    }
    return numerals;
  }

  /**
   * Links two entity mentions of the same type when the anaphor's head appears anywhere
   * in the candidate's cluster and the cluster words include the anaphor's, unless the
   * two names differ by a compound head such as {@code city}, which tells
   * {@code Kansas City} apart from {@code Kansas}.
   */
  private boolean relaxedHeadMatch(int i, int j) {
    final Mention antecedent = mentions.get(i);
    final Mention anaphor = mentions.get(j);
    if (antecedent.pronoun() || antecedent.type() == null
        || !antecedent.type().equals(anaphor.type())
        || !clusters.words(i).contains(anaphor.head())
        || !clusters.words(i).containsAll(clusters.words(j))) {
      return false;
    }
    return !compoundDiffers(antecedent, anaphor) && !compoundDiffers(anaphor, antecedent);
  }

  /** Checks whether one name ends in a compound head the other name lacks. */
  private static boolean compoundDiffers(Mention compound, Mention other) {
    return CorefLexicon.compoundNameHead(compound.head())
        && !other.words().contains(compound.head());
  }

  /**
   * Links a pronoun to the nearest candidate whose cluster agrees in number, gender,
   * animacy, and person and whose entity type the pronoun class accepts.
   */
  private boolean pronounMatch(int i, int j) {
    if (!clusters.attributesAgree(i, j)) {
      return false;
    }
    final String type = clusters.type(i);
    if (type == null) {
      return true;
    }
    final Gender gender = mentions.get(j).gender();
    if (gender == Gender.NEUTRAL) {
      return neutralTypes.contains(type);
    }
    if (gender == Gender.MALE || gender == Gender.FEMALE) {
      return personTypes.contains(type);
    }
    return personTypes.contains(type) || neutralTypes.contains(type);
  }
}
