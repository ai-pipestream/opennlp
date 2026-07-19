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

package opennlp.tools.postag;

/**
 * A first-order linear-chain CRF over tag emissions: learned transition, start, and
 * end scores, the forward-backward algorithm for the negative log-likelihood and its
 * gradients, and Viterbi for decoding. All arithmetic runs in {@code double} with
 * max-shifted log-sum-exp, matching the precision discipline of {@link LstmLayer};
 * instances are stateless and thread-safe.
 */
final class CrfScorer {

  private final int tagCount;

  /**
   * @param tagCount The number of tags. Must be positive.
   */
  CrfScorer(int tagCount) {
    if (tagCount <= 0) {
      throw new IllegalArgumentException("tagCount must be positive");
    }
    this.tagCount = tagCount;
  }

  /**
   * @return The number of tags.
   */
  int tagCount() {
    return tagCount;
  }

  /**
   * Computes the CRF negative log-likelihood of the gold tag sequence and the
   * gradient of the loss with respect to every emission and transition parameter.
   *
   * @param emissions The per-position emission scores, {@code [T][tagCount]}. Must
   *                  not be {@code null} or empty.
   * @param gold The gold tag indices, one per position. Must not be {@code null}.
   * @param transitions The transition scores, {@code [tagCount][tagCount]}. Must not
   *                    be {@code null}.
   * @param start The start scores. Must not be {@code null}.
   * @param end The end scores. Must not be {@code null}.
   * @param emissionGrads Receives the loss gradient per emission, added in place.
   *                      Must not be {@code null}.
   * @param transitionGrads Receives the loss gradient per transition, added in place.
   *                        Must not be {@code null}.
   * @param startGrads Receives the loss gradient per start score, added in place.
   *                   Must not be {@code null}.
   * @param endGrads Receives the loss gradient per end score, added in place. Must
   *                 not be {@code null}.
   * @return The negative log-likelihood of the gold sequence.
   */
  double lossAndGradients(double[][] emissions, int[] gold, double[][] transitions,
      double[] start, double[] end, double[][] emissionGrads,
      double[][] transitionGrads, double[] startGrads, double[] endGrads) {
    final int steps = emissions.length;
    final double[][] forward = forward(emissions, transitions, start);
    final double[][] backward = backward(emissions, transitions, end);
    final double logZ = logSumExp(lastColumnLogits(forward, end));

    double goldScore = start[gold[0]];
    for (int t = 0; t < steps; t++) {
      goldScore += emissions[t][gold[t]];
      if (t > 0) {
        goldScore += transitions[gold[t - 1]][gold[t]];
      }
    }
    goldScore += end[gold[steps - 1]];

    for (int t = 0; t < steps; t++) {
      for (int k = 0; k < tagCount; k++) {
        final double marginal = Math.exp(forward[t][k] + backward[t][k] - logZ);
        emissionGrads[t][k] += marginal;
      }
      emissionGrads[t][gold[t]] -= 1.0d;
    }
    for (int j = 0; j < tagCount; j++) {
      startGrads[j] += Math.exp(forward[0][j] + backward[0][j] - logZ);
      endGrads[j] += Math.exp(forward[steps - 1][j] + backward[steps - 1][j] - logZ);
    }
    startGrads[gold[0]] -= 1.0d;
    endGrads[gold[steps - 1]] -= 1.0d;
    for (int t = 0; t < steps - 1; t++) {
      for (int j = 0; j < tagCount; j++) {
        for (int k = 0; k < tagCount; k++) {
          final double pairwise = Math.exp(forward[t][j] + transitions[j][k]
              + emissions[t + 1][k] + backward[t + 1][k] - logZ);
          transitionGrads[j][k] += pairwise;
        }
      }
      transitionGrads[gold[t]][gold[t + 1]] -= 1.0d;
    }
    return logZ - goldScore;
  }

  /**
   * Computes the posterior tag marginals of every position,
   * {@code exp(forward + backward - logZ)}.
   *
   * @param emissions The per-position emission scores. Must not be {@code null}.
   * @param transitions The transition scores. Must not be {@code null}.
   * @param start The start scores. Must not be {@code null}.
   * @param end The end scores. Must not be {@code null}.
   * @return The per-position tag marginals, {@code [T][tagCount]}. Never
   *         {@code null}.
   */
  double[][] marginals(double[][] emissions, double[][] transitions, double[] start,
      double[] end) {
    final double[][] forward = forward(emissions, transitions, start);
    final double[][] backward = backward(emissions, transitions, end);
    final double logZ = logSumExp(lastColumnLogits(forward, end));
    final double[][] marginals = new double[emissions.length][tagCount];
    for (int t = 0; t < emissions.length; t++) {
      for (int k = 0; k < tagCount; k++) {
        marginals[t][k] = Math.exp(forward[t][k] + backward[t][k] - logZ);
      }
    }
    return marginals;
  }

  /**
   * Decodes the highest-scoring tag sequence under the model.
   *
   * @param emissions The per-position emission scores. Must not be {@code null}.
   * @param transitions The transition scores. Must not be {@code null}.
   * @param start The start scores. Must not be {@code null}.
   * @param end The end scores. Must not be {@code null}.
   * @return The tag indices of the best sequence. Never {@code null}.
   */
  int[] viterbi(double[][] emissions, double[][] transitions, double[] start,
      double[] end) {
    final int steps = emissions.length;
    final double[][] delta = new double[steps][tagCount];
    final int[][] backpointers = new int[steps][tagCount];
    for (int k = 0; k < tagCount; k++) {
      delta[0][k] = start[k] + emissions[0][k];
    }
    for (int t = 1; t < steps; t++) {
      for (int k = 0; k < tagCount; k++) {
        double best = Double.NEGATIVE_INFINITY;
        int bestPrevious = 0;
        for (int j = 0; j < tagCount; j++) {
          final double candidate = delta[t - 1][j] + transitions[j][k];
          if (candidate > best) {
            best = candidate;
            bestPrevious = j;
          }
        }
        delta[t][k] = best + emissions[t][k];
        backpointers[t][k] = bestPrevious;
      }
    }
    double best = Double.NEGATIVE_INFINITY;
    int bestLast = 0;
    for (int k = 0; k < tagCount; k++) {
      final double candidate = delta[steps - 1][k] + end[k];
      if (candidate > best) {
        best = candidate;
        bestLast = k;
      }
    }
    final int[] path = new int[steps];
    path[steps - 1] = bestLast;
    for (int t = steps - 1; t > 0; t--) {
      path[t - 1] = backpointers[t][path[t]];
    }
    return path;
  }

  private double[][] forward(double[][] emissions, double[][] transitions,
      double[] start) {
    final int steps = emissions.length;
    final double[][] alpha = new double[steps][tagCount];
    for (int k = 0; k < tagCount; k++) {
      alpha[0][k] = start[k] + emissions[0][k];
    }
    for (int t = 1; t < steps; t++) {
      for (int k = 0; k < tagCount; k++) {
        double max = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < tagCount; j++) {
          max = Math.max(max, alpha[t - 1][j] + transitions[j][k]);
        }
        double sum = 0.0d;
        for (int j = 0; j < tagCount; j++) {
          sum += Math.exp(alpha[t - 1][j] + transitions[j][k] - max);
        }
        alpha[t][k] = emissions[t][k] + max + Math.log(sum);
      }
    }
    return alpha;
  }

  private double[][] backward(double[][] emissions, double[][] transitions,
      double[] end) {
    final int steps = emissions.length;
    final double[][] beta = new double[steps][tagCount];
    System.arraycopy(end, 0, beta[steps - 1], 0, tagCount);
    for (int t = steps - 2; t >= 0; t--) {
      for (int j = 0; j < tagCount; j++) {
        double max = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < tagCount; k++) {
          max = Math.max(max, transitions[j][k] + emissions[t + 1][k] + beta[t + 1][k]);
        }
        double sum = 0.0d;
        for (int k = 0; k < tagCount; k++) {
          sum += Math.exp(transitions[j][k] + emissions[t + 1][k] + beta[t + 1][k] - max);
        }
        beta[t][j] = max + Math.log(sum);
      }
    }
    return beta;
  }

  private double[] lastColumnLogits(double[][] forward, double[] end) {
    final double[] logits = new double[tagCount];
    final int last = forward.length - 1;
    for (int k = 0; k < tagCount; k++) {
      logits[k] = forward[last][k] + end[k];
    }
    return logits;
  }

  private static double logSumExp(double[] values) {
    double max = Double.NEGATIVE_INFINITY;
    for (final double value : values) {
      max = Math.max(max, value);
    }
    double sum = 0.0d;
    for (final double value : values) {
      sum += Math.exp(value - max);
    }
    return max + Math.log(sum);
  }
}
