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
import java.util.List;
import java.util.Map;

/**
 * Generates the features of one anaphor and candidate antecedent pair for the ranking
 * model: the sieve tests as predicates, the surface shape of each mention, the distance
 * between them, their attribute pairs, and conjunctions of mention kinds with the rest,
 * after the feature set of
 * <a href="https://aclanthology.org/D13-1203/">Durrett and Klein (EMNLP 2013), "Easy
 * Victories and Uphill Battles in Coreference Resolution"</a>.
 *
 * <p>Features are opaque strings of the form {@code name=value}; the model learns a
 * weight per string. Head words and pronoun forms are lexical features and so make the
 * model corpus-specific; every other feature is closed-class.</p>
 */
final class CorefContextGenerator {

  /** The value of a similarity feature when a head has no vector. */
  private static final String UNKNOWN_SIMILARITY = "unknown";

  private final SieveResolver resolver;
  private final List<Mention> mentions;
  private final Clusters clusters;
  private final WordVectors vectors;
  private final Map<String, float[]> vectorCache = new HashMap<>();

  /**
   * Initializes the generator over a resolver's mentions and clusters.
   *
   * @param resolver The resolver whose sieve tests supply the predicates.
   */
  CorefContextGenerator(SieveResolver resolver) {
    this.resolver = resolver;
    this.mentions = resolver.mentions();
    this.clusters = resolver.clusters();
    this.vectors = resolver.vectors();
  }

  /**
   * Generates the features of the option that starts a new chain: the anaphor's own
   * shape and what precedes it, so the ranker learns which mentions are first mentions.
   *
   * @param j The anaphor's index.
   * @return The feature strings. Never {@code null}.
   */
  String[] newChainFeatures(int j) {
    final Mention anaphor = mentions.get(j);
    final List<String> features = new ArrayList<>(16);
    final String kind = "new|" + kind(anaphor);
    features.add(kind);
    features.add(kind + "|head=" + anaphor.head());
    features.add(kind + "|first=" + first(anaphor));
    features.add(kind + "|shape=" + shape(anaphor));
    features.add(kind + "|length=" + bucket(anaphor.words().size()));
    features.add(kind + "|type=" + anaphor.type());
    features.add(kind + "|sentence=" + bucket(anaphor.sentence()));
    features.add(kind + "|position=" + bucket(positionInSentence(j)));
    features.add(kind + "|earlierHead=" + earlierHead(j));
    features.add(kind + "|earlierExact=" + earlierExact(j));
    if (anaphor.pronoun()) {
      features.add(kind + "|form=" + anaphor.head());
    }
    return features.toArray(new String[0]);
  }

  /**
   * Generates the features of a pair.
   *
   * @param i The candidate antecedent's index.
   * @param j The anaphor's index.
   * @return The feature strings. Never {@code null}.
   */
  String[] features(int i, int j) {
    final Mention antecedent = mentions.get(i);
    final Mention anaphor = mentions.get(j);
    final List<String> features = new ArrayList<>(80);
    final String kinds = kind(anaphor) + '>' + kind(antecedent);
    features.add("kinds=" + kinds);
    features.add("antClusterSize=" + bucket(clusters.size(i)));
    features.add(kinds + "|antClusterSize=" + bucket(clusters.size(i)));
    features.add("clusterExact=" + clusters.normalizedForms(i).contains(anaphor.normalized()));
    features.add("clusterHead=" + clusters.heads(i).contains(anaphor.head()));
    features.add("antPosition=" + bucket(positionInSentence(i)));
    features.add("sameSpeaker=" + (resolver.speakerOf(i).equals(resolver.speakerOf(j))));
    if (antecedent.pronoun()) {
      features.add("antForm=" + antecedent.head());
      features.add("antForm=" + antecedent.head() + "|anaHead=" + anaphor.head());
    }
    if (vectors != null && !anaphor.pronoun()) {
      final String similarity = "sim=" + similarity(anaphor.head(), antecedent.head());
      features.add(similarity);
      features.add(kinds + '|' + similarity);
      String best = UNKNOWN_SIMILARITY;
      double bestCosine = -2.0;
      for (final String head : clusters.heads(i)) {
        final double cosine = cosine(anaphor.head(), head);
        if (cosine > bestCosine) {
          bestCosine = cosine;
          best = similarityBucket(cosine);
        }
      }
      features.add("clusterSim=" + best);
    }

    sieve(features, "exact", resolver.exactMatch(i, j), kinds);
    sieve(features, "relaxed", resolver.relaxedStringMatch(i, j), kinds);
    sieve(features, "acronym", resolver.acronym(i, j), kinds);
    sieve(features, "personName", resolver.personName(i, j), kinds);
    sieve(features, "strictHead", resolver.strictHeadMatch(i, j, true, true), kinds);
    sieve(features, "headNoModifiers", resolver.strictHeadMatch(i, j, true, false), kinds);
    sieve(features, "headNoInclusion", resolver.strictHeadMatch(i, j, false, true), kinds);
    sieve(features, "properHead", resolver.properHeadMatch(i, j), kinds);
    sieve(features, "relaxedHead", resolver.relaxedHeadMatch(i, j), kinds);
    sieve(features, "agree", clusters.attributesAgree(i, j), kinds);
    sieve(features, "numberAgree", clusters.numbersAgree(i, j), kinds);
    if (anaphor.pronoun()) {
      sieve(features, "pronounMatch", resolver.pronounMatch(i, j), kinds);
    }

    final boolean sameHead = anaphor.head() != null && anaphor.head().equals(antecedent.head());
    features.add("sameHead=" + sameHead);
    features.add("anaHead=" + anaphor.head());
    features.add("antHead=" + antecedent.head());
    features.add("heads=" + anaphor.head() + '|' + antecedent.head());
    features.add("anaFirst=" + first(anaphor));
    features.add("antFirst=" + first(antecedent));
    features.add("anaShape=" + shape(anaphor));
    features.add("antShape=" + shape(antecedent));
    features.add("anaLength=" + bucket(anaphor.words().size()));
    features.add("antLength=" + bucket(antecedent.words().size()));
    features.add("anaType=" + anaphor.type());
    features.add("antType=" + antecedent.type());
    features.add("types=" + anaphor.type() + '|' + antecedent.type());
    features.add("gender=" + anaphor.gender() + '|' + antecedent.gender());
    features.add("number=" + anaphor.number() + '|' + antecedent.number());
    features.add("animacy=" + anaphor.animacy() + '|' + antecedent.animacy());
    features.add("antPerson=" + antecedent.person());
    features.add("antFirstInCluster=" + (clusters.find(i) == i));

    final int sentences = anaphor.sentence() - antecedent.sentence();
    final String distance = "dist=" + bucket(sentences) + '|' + bucket(j - i);
    features.add(distance);
    features.add(kinds + '|' + distance);
    features.add(kinds + "|sameHead=" + sameHead);
    if (anaphor.pronoun()) {
      final String form = "form=" + anaphor.head();
      features.add(form);
      features.add(form + '|' + kind(antecedent));
      features.add(form + "|antGender=" + antecedent.gender());
      features.add(form + "|antNumber=" + antecedent.number());
      features.add(form + "|antAnimacy=" + antecedent.animacy());
      features.add(form + "|antType=" + antecedent.type());
      features.add(form + '|' + distance);
      features.add(form + "|antHead=" + antecedent.head());
    } else {
      features.add("anaFirst=" + first(anaphor) + '|' + kind(antecedent));
      features.add("anaIndefinite=" + anaphor.indefinite());
      features.add("antIndefinite=" + antecedent.indefinite());
      features.add("bothProper=" + (anaphor.proper() && antecedent.proper()));
    }
    return features.toArray(new String[0]);
  }

  /** {@return whether an earlier mention shares the anaphor's head} */
  private boolean earlierHead(int j) {
    final String head = mentions.get(j).head();
    for (int i = 0; i < j; i++) {
      if (head.equals(mentions.get(i).head())) {
        return true;
      }
    }
    return false;
  }

  /** {@return whether an earlier mention has the anaphor's normalized text} */
  private boolean earlierExact(int j) {
    final String normalized = mentions.get(j).normalized();
    for (int i = 0; i < j; i++) {
      if (normalized.equals(mentions.get(i).normalized())) {
        return true;
      }
    }
    return false;
  }

  /** {@return the bucketed cosine similarity of two head words, or unknown} */
  private String similarity(String a, String b) {
    return similarityBucket(cosine(a, b));
  }

  /** Buckets a cosine into five ranges; {@code -2} stands for a missing vector. */
  private String similarityBucket(double cosine) {
    if (cosine < -1.0) {
      return UNKNOWN_SIMILARITY;
    }
    if (cosine < 0.3) {
      return "low";
    }
    if (cosine < 0.5) {
      return "mid";
    }
    if (cosine < 0.7) {
      return "high";
    }
    return cosine < 0.9 ? "higher" : "same";
  }

  /** {@return the cosine of two head words' vectors, or {@code -2} when one is missing} */
  private double cosine(String a, String b) {
    final float[] u = vector(a);
    final float[] v = vector(b);
    if (u == null || v == null || u.length != v.length) {
      return -2.0;
    }
    double dot = 0.0;
    double uu = 0.0;
    double vv = 0.0;
    for (int d = 0; d < u.length; d++) {
      dot += u[d] * v[d];
      uu += u[d] * u[d];
      vv += v[d] * v[d];
    }
    return uu == 0.0 || vv == 0.0 ? -2.0 : dot / Math.sqrt(uu * vv);
  }

  private float[] vector(String word) {
    if (word == null) {
      return null;
    }
    return vectorCache.computeIfAbsent(word, vectors::vector);
  }

  /** {@return how many mentions precede a mention in its sentence} */
  private int positionInSentence(int j) {
    int position = 0;
    for (int i = j - 1; i >= 0 && mentions.get(i).sentence() == mentions.get(j).sentence(); i--) {
      position++;
    }
    return position;
  }

  /** Adds a predicate feature and its conjunction with the mention kinds when it holds. */
  private void sieve(List<String> features, String name, boolean holds, String kinds) {
    if (holds) {
      features.add(name);
      features.add(name + '|' + kinds);
    }
  }

  /** {@return a mention's kind: its pronoun form class, proper, nominal, or entity} */
  private String kind(Mention mention) {
    if (mention.pronoun()) {
      return "pronoun";
    }
    if (mention.namedEntity()) {
      return "entity";
    }
    return mention.proper() ? "proper" : "nominal";
  }

  /** {@return a mention's first word, the determiner or opener that marks its definiteness} */
  private String first(Mention mention) {
    return mention.words().isEmpty() ? "" : mention.words().get(0);
  }

  /** {@return whether a mention is a proper name, a definite, or an indefinite phrase} */
  private String shape(Mention mention) {
    if (mention.proper()) {
      return "proper";
    }
    return mention.indefinite() ? "indefinite" : "definite";
  }

  /** Buckets a count so distances and lengths share a few coarse values. */
  private String bucket(int value) {
    if (value <= 3) {
      return Integer.toString(value);
    }
    if (value <= 7) {
      return "4-7";
    }
    if (value <= 15) {
      return "8-15";
    }
    return "16+";
  }
}
