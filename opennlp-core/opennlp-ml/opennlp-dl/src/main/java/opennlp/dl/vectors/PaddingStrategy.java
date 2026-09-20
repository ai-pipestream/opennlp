/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package opennlp.dl.vectors;

import java.util.List;

/**
 * How {@link SentenceVectorsDL} shapes the tensors of one inference when the inputs of a call
 * differ in tokenized length.
 *
 * <p>All three produce the same sentence vectors, within float tolerance: a padded position is
 * {@code 0} in the attention mask, and {@link Pooling} reads only the positions a row's own mask
 * covers. Where the pooling happens inside the model instead, through a {@code sentence_embedding}
 * output, that equality holds only if the graph honors {@code attention_mask}. Nothing here can
 * check whether it does, so a caller pairing a padding strategy with such a model is relying on a
 * property of that model. Every export that pools with the mask, which is what
 * sentence-transformers produces, satisfies it. What differs between the strategies is the shape
 * of the tensors and how many times the session runs for a given call, which is a throughput and
 * memory decision, not an accuracy one.</p>
 *
 * <p>These are the three choices a batching encoder has, and they carry the names the wider
 * ecosystem uses: {@link #LONGEST} and {@link #MAX_LENGTH} are DJL's
 * {@code PaddingStrategy.LONGEST} and {@code PaddingStrategy.MAX_LENGTH}, and Hugging Face's
 * {@code padding=True} and {@code padding="max_length"}. {@link #EXACT_LENGTH} corresponds to
 * asking for no padding at all.</p>
 *
 * @see SentenceVectorsDL#embedAll(List)
 * @since 3.0.0
 */
public enum PaddingStrategy {

  /**
   * No padding. The inputs of a call are grouped by their exact tokenized length and each group
   * runs on its own, so no tensor ever holds a padded position and every row is computed from
   * the tensors its single-input call would have used.
   *
   * <p>This is the default, and it fragments on natural text: measured on English news
   * sentences, a call of 64 inputs splits into roughly 33 groups and a call of 8 into roughly 7,
   * so most inferences carry one or two rows. Prefer {@link #LONGEST} unless the vocabulary has
   * no padding token or every input of a call is known to tokenize to the same length.</p>
   */
  EXACT_LENGTH,

  /**
   * Every row of a batch is padded to the longest row in that batch, so a whole call of
   * mixed-length inputs can run as a single inference at shape {@code [rows, longest row]}. Rows
   * are ordered by tokenized length before batching, so one long input cannot pad out a batch of
   * short ones. Requires a padding token in the vocabulary.
   */
  LONGEST,

  /**
   * Every row of every batch is padded to the configured maximum sequence length, so every
   * inference has the same shape {@code [rows, maxLength]} whatever the inputs are. This wastes
   * work on short text, and is the choice for a fixed-shape accelerator or a graph that will not
   * accept a dynamic sequence axis. Truncation still applies first, so an input longer than the
   * maximum is cut to it rather than widening the tensor. Requires a padding token in the
   * vocabulary.
   */
  MAX_LENGTH
}
