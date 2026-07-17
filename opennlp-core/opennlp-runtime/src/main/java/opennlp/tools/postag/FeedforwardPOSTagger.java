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
 * The pure-Java neural {@link POSTagger}: a greedy left-to-right decoder over the
 * {@link FeedforwardPOSModel}, feeding each position the two previously assigned tags.
 *
 * <p>Inference is ordinary array arithmetic with no native runtime involved, so this
 * tagger deploys exactly like the classical one while scoring positions with learned
 * dense representations of words, suffixes, and shapes instead of sparse feature
 * conjunctions.</p>
 *
 * <p>Decoding is greedy rather than a search, so the tagger produces exactly one tag
 * sequence for a sentence. It supports the whole {@link POSTagger} interface on that
 * basis: {@link #topKSequences(String[])} returns an array holding that single sequence,
 * which is always the tagging {@link #tag(String[])} returns, carrying the model's
 * probability for every tag it assigned. It never returns the several ranked alternatives
 * a search-based tagger such as {@link POSTaggerME} returns, because a greedy decoder has
 * none to rank. Consumers that explore alternative taggings, in particular
 * {@link opennlp.tools.parser.AbstractBottomUpParser}, therefore work with this tagger
 * but consider one tagging per sentence instead of a beam of them, which reduces the
 * parse candidates they can recover from a tagging error.</p>
 *
 * <p>The {@code additionalContext} of the interface carries no information this model was
 * trained on, so both overloads that take it ignore it.</p>
 *
 * <p>The tagger holds an immutable model and no per-call state, so one instance can be
 * shared between threads.</p>
 *
 * @see FeedforwardPOSTrainer
 * @since 3.0.0
 */
public class FeedforwardPOSTagger implements POSTagger {

  private final FeedforwardPOSModel model;
  private final String[] tags;

  /**
   * Initializes a {@link FeedforwardPOSTagger}.
   *
   * @param model The model to tag with. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code model} is {@code null}.
   */
  public FeedforwardPOSTagger(FeedforwardPOSModel model) {
    if (model == null) {
      throw new IllegalArgumentException("model must not be null");
    }
    this.model = model;
    this.tags = model.tags();
    // A tagger only ever reads a frozen model, so the scoring cache is safe here.
    model.enableScoringCache();
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
   * Assigns the sentence its tag sequences. The greedy decoder has exactly one tagging
   * for a sentence, so the returned array always holds a single {@link Sequence}: the
   * tagging {@link #tag(String[])} returns, whose probability per tag is the model's
   * probability for the tag it assigned and whose score is the sum of the logs of those
   * probabilities. This tagger never returns several ranked alternatives, which the
   * interface permits because it fixes no minimum number of sequences.
   *
   * @param sentence The sentence of tokens to be tagged. Must not be {@code null}.
   * @return An array of length one holding the greedy tagging. Never {@code null}.
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
   * result is the single greedy tagging described on {@link #topKSequences(String[])}.
   *
   * @param sentence The sentence of tokens to be tagged. Must not be {@code null}.
   * @param additionalContext Ignored, as this model is not trained on any.
   * @return An array of length one holding the greedy tagging. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code sentence} is {@code null}.
   */
  @Override
  public Sequence[] topKSequences(String[] sentence, Object[] additionalContext) {
    return topKSequences(sentence);
  }

  /**
   * Runs the greedy left-to-right decoder, which is the single decoding path behind
   * every tagging method, so that the tags and the sequence can never disagree.
   *
   * @param sentence The sentence of tokens to be tagged. Must not be {@code null}.
   * @param collected The {@link Sequence} to record every assigned tag and its
   *                  probability in, or {@code null} to skip recording when only the
   *                  tags are wanted.
   * @return One pos tag per token of {@code sentence}. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code sentence} is {@code null}.
   */
  private String[] decode(String[] sentence, Sequence collected) {
    if (sentence == null) {
      throw new IllegalArgumentException("sentence must not be null");
    }
    final String[] assigned = new String[sentence.length];
    for (int i = 0; i < sentence.length; i++) {
      final double[] scores = model.score(model.featureIds(
          FeedforwardPOSContext.extract(sentence, i,
              i > 0 ? assigned[i - 1] : null, i > 1 ? assigned[i - 2] : null)));
      int best = 0;
      for (int o = 1; o < scores.length; o++) {
        if (scores[o] > scores[best]) {
          best = o;
        }
      }
      assigned[i] = tags[best];
      if (collected != null) {
        collected.add(tags[best], probability(scores, best));
      }
    }
    return assigned;
  }

  /**
   * Turns the model's unnormalized tag scores into the probability of the tag the
   * decoder assigned, by applying the softmax shifted by the highest score so that no
   * term of the sum can overflow.
   *
   * @param scores One unnormalized score per tag, as returned by
   *               {@link FeedforwardPOSModel#score(int[])}. Must not be empty.
   * @param best The index of the highest scoring tag. Because it is the highest, its
   *             own term of the sum is exactly one, which keeps the sum at or above one
   *             and the result strictly positive.
   * @return The model's probability of the tag at {@code best}, in the range
   *         {@code (0, 1]}.
   * @throws IllegalStateException Thrown if a score is not finite, which no trained
   *         model produces; the {@code NaN} such a score would yield violates the
   *         documented range and silently corrupts every score-ordered collection
   *         downstream.
   */
  private static double probability(double[] scores, int best) {
    double total = 0.0;
    for (final double score : scores) {
      total += StrictMath.exp(score - scores[best]);
    }
    final double probability = 1.0 / total;
    if (Double.isNaN(probability)) {
      throw new IllegalStateException(
          "the model produced a non-finite tag score; the model file is corrupt");
    }
    return probability;
  }
}
