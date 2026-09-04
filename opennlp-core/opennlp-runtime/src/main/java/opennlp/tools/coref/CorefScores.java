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

/**
 * The scores a {@link CorefScorer} has accumulated: one precision, recall, and F1 triple
 * per coreference metric, the mention detection triple, and the CoNLL average.
 *
 * <p>The CoNLL average is the arithmetic mean of the MUC, B-cubed, and entity-based CEAF
 * F1 scores, the official figure of the CoNLL-2011 and CoNLL-2012 shared tasks. Every
 * value lies in {@code [0, 1]}.</p>
 *
 * @param muc The link-based MUC score.
 * @param bCubed The mention-based B-cubed score.
 * @param ceafM The mention-based CEAF score.
 * @param ceafE The entity-based CEAF score.
 * @param mentions The mention detection score, comparing the mention sets alone.
 * @param conll The CoNLL average F1.
 *
 * @since 3.0.0
 */
public record CorefScores(Score muc, Score bCubed, Score ceafM, Score ceafE,
    Score mentions, double conll) {

  /**
   * Validates the scores.
   *
   * @throws IllegalArgumentException Thrown if a score is {@code null} or {@code conll}
   *         lies outside {@code [0, 1]}.
   */
  public CorefScores {
    if (muc == null || bCubed == null || ceafM == null || ceafE == null
        || mentions == null) {
      throw new IllegalArgumentException("scores must not be null");
    }
    if (!(conll >= 0.0 && conll <= 1.0)) {
      throw new IllegalArgumentException("conll must lie in [0, 1]: " + conll);
    }
  }

  /**
   * One precision, recall, and F1 triple.
   *
   * @param precision The precision in {@code [0, 1]}.
   * @param recall The recall in {@code [0, 1]}.
   * @param f1 The harmonic mean of precision and recall, {@code 0} when both are zero.
   *
   * @since 3.0.0
   */
  public record Score(double precision, double recall, double f1) {

    /**
     * Validates the triple.
     *
     * @throws IllegalArgumentException Thrown if a value lies outside {@code [0, 1]}.
     */
    public Score {
      requireUnit(precision, "precision");
      requireUnit(recall, "recall");
      requireUnit(f1, "f1");
    }

    /**
     * Builds a triple from a numerator and denominator pair for each side, treating an
     * empty denominator as a score of zero, as the reference scorer does.
     *
     * @param precisionNumerator The precision numerator.
     * @param precisionDenominator The precision denominator.
     * @param recallNumerator The recall numerator.
     * @param recallDenominator The recall denominator.
     * @return The triple. Never {@code null}.
     */
    static Score of(double precisionNumerator, double precisionDenominator,
        double recallNumerator, double recallDenominator) {
      final double precision = ratio(precisionNumerator, precisionDenominator);
      final double recall = ratio(recallNumerator, recallDenominator);
      final double f1 = precision + recall == 0.0
          ? 0.0 : 2.0 * precision * recall / (precision + recall);
      return new Score(precision, recall, f1);
    }

    /** Divides, mapping an empty denominator to zero. */
    private static double ratio(double numerator, double denominator) {
      return denominator == 0.0 ? 0.0 : numerator / denominator;
    }

    /** Rejects a value outside the unit interval, NaN included. */
    private static void requireUnit(double value, String name) {
      if (!(value >= 0.0 && value <= 1.0)) {
        throw new IllegalArgumentException(name + " must lie in [0, 1]: " + value);
      }
    }
  }
}
