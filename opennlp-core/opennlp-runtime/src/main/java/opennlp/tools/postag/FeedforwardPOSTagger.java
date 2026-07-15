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
  }

  @Override
  public String[] tag(String[] sentence) {
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
    }
    return assigned;
  }

  @Override
  public String[] tag(String[] sentence, Object[] additionalContext) {
    return tag(sentence);
  }

  @Override
  public Sequence[] topKSequences(String[] sentence) {
    throw new UnsupportedOperationException(
        "the greedy feedforward tagger does not produce k-best sequences");
  }

  @Override
  public Sequence[] topKSequences(String[] sentence, Object[] additionalContext) {
    throw new UnsupportedOperationException(
        "the greedy feedforward tagger does not produce k-best sequences");
  }
}
