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

import opennlp.tools.util.Sequence;

/**
 * The bidirectional LSTM {@link POSTagger}: one forward pass of the sentence BiLSTM
 * scores every tag at every position, and the per-position argmax forms the tagging.
 * Unlike the feedforward tagger there is no left-to-right tag conditioning, so every
 * position is scored from whole-sentence context instead of the two previously
 * assigned tags.
 *
 * <p>Inference is ordinary array arithmetic with no native runtime involved, so this
 * tagger deploys exactly like the classical one. Decoding picks the per-position
 * argmax rather than running a search, so the tagger produces exactly one tag
 * sequence for a sentence. It supports the whole {@link POSTagger} interface on that
 * basis: {@link #topKSequences(String[])} returns an array holding that single
 * sequence, which is always the tagging {@link #tag(String[])} returns, carrying the
 * model's probability for every tag it assigned.</p>
 *
 * <p>The {@code additionalContext} of the interface carries no information this model
 * was trained on, so both overloads that take it ignore it.</p>
 *
 * <p>The tagger holds an immutable model and no per-call state, so one instance can
 * be shared between threads.</p>
 *
 * @see BilstmPOSTrainer
 * @see BilstmPOSModel
 * @since 3.0.0
 */
public class BilstmPOSTagger implements POSTagger {

  private final BilstmPOSModel model;
  private final String[] tags;

  /**
   * Initializes a {@link BilstmPOSTagger}.
   *
   * @param model The model to tag with. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code model} is {@code null}.
   */
  public BilstmPOSTagger(BilstmPOSModel model) {
    if (model == null) {
      throw new IllegalArgumentException("model must not be null");
    }
    this.model = model;
    this.tags = model.tags();
    // A tagger only ever reads a frozen model, so the representation cache is safe.
    model.enableRepresentationCache();
  }

  /**
   * Assigns the sentence of tokens pos tags.
   *
   * @param sentence The sentence of tokens to be tagged. Must not be {@code null}.
   * @return One pos tag per token of {@code sentence}. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code sentence} is {@code null}.
   */
  @Override
  public String[] tag(String[] sentence) {
    return decode(sentence, null);
  }

  /**
   * Assigns the sentence of tokens pos tags, ignoring {@code additionalContext}.
   *
   * @param sentence The sentence of tokens to be tagged. Must not be {@code null}.
   * @param additionalContext Ignored, as this model is not trained on any.
   * @return One pos tag per token of {@code sentence}. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code sentence} is {@code null}.
   */
  @Override
  public String[] tag(String[] sentence, Object[] additionalContext) {
    return decode(sentence, null);
  }

  /**
   * Assigns the sentence its tag sequences. The argmax decoder has exactly one
   * tagging for a sentence, so the returned array always holds a single
   * {@link Sequence}: the tagging {@link #tag(String[])} returns, whose probability
   * per tag is the model's probability for the tag it assigned and whose score is the
   * sum of the logs of those probabilities. This tagger never returns several ranked
   * alternatives, which the interface permits because it fixes no minimum number of
   * sequences.
   *
   * @param sentence The sentence of tokens to be tagged. Must not be {@code null}.
   * @return An array of length one holding the decoded tagging. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code sentence} is {@code null}.
   */
  @Override
  public Sequence[] topKSequences(String[] sentence) {
    final Sequence sequence = new Sequence();
    decode(sentence, sequence);
    return new Sequence[] {sequence};
  }

  /**
   * Assigns the sentence its tag sequences, ignoring {@code additionalContext}. The
   * result is the single tagging described on {@link #topKSequences(String[])}.
   *
   * @param sentence The sentence of tokens to be tagged. Must not be {@code null}.
   * @param additionalContext Ignored, as this model is not trained on any.
   * @return An array of length one holding the decoded tagging. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code sentence} is {@code null}.
   */
  @Override
  public Sequence[] topKSequences(String[] sentence, Object[] additionalContext) {
    return topKSequences(sentence);
  }

  private String[] decode(String[] sentence, Sequence collected) {
    if (sentence == null) {
      throw new IllegalArgumentException("sentence must not be null");
    }
    final String[] assigned = new String[sentence.length];
    if (sentence.length == 0) {
      return assigned;
    }
    final double[][] scores = model.score(sentence);
    if (model.isCrf()) {
      final CrfScorer scorer = new CrfScorer(tags.length);
      final int[] path = scorer.viterbi(scores, model.transitionWeights(),
          model.startWeights(), model.endWeights());
      final double[][] marginals = collected != null
          ? scorer.marginals(scores, model.transitionWeights(), model.startWeights(),
              model.endWeights())
          : null;
      for (int i = 0; i < sentence.length; i++) {
        assigned[i] = tags[path[i]];
        if (collected != null) {
          collected.add(assigned[i], marginals[i][path[i]]);
        }
      }
      return assigned;
    }
    for (int i = 0; i < sentence.length; i++) {
      int best = 0;
      for (int o = 1; o < tags.length; o++) {
        if (scores[i][o] > scores[i][best]) {
          best = o;
        }
      }
      assigned[i] = tags[best];
      if (collected != null) {
        collected.add(tags[best], probability(scores[i], best));
      }
    }
    return assigned;
  }

  /**
   * Turns one position's unnormalized tag scores into the probability of the assigned
   * tag, applying the softmax shifted by the highest score so that no term of the sum
   * can overflow.
   *
   * @param scores One unnormalized score per tag. Must not be empty.
   * @param best The index of the highest scoring tag.
   * @return The model's probability of the tag at {@code best}, in the range
   *         {@code (0, 1]}.
   * @throws IllegalStateException Thrown if a score is not finite, which no trained
   *         model produces.
   */
  private static double probability(double[] scores, int best) {
    double total = 0.0d;
    for (final double score : scores) {
      total += Math.exp(score - scores[best]);
    }
    final double probability = 1.0d / total;
    if (Double.isNaN(probability)) {
      throw new IllegalStateException(
          "the model produced a non-finite tag score; the model file is corrupt");
    }
    return probability;
  }
}
