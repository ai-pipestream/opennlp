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

import opennlp.dl.ExecutionProviderRequest;
import opennlp.dl.ExecutionProviders;
import opennlp.dl.InferenceOptions;

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
 * <p>A caller that states no strategy gets the one {@link #defaultFor(List)} derives from the
 * execution providers of the session, since the best shape depends on where the session runs:
 * {@link #LONGEST} on an accelerator, {@link #EXACT_LENGTH} on the CPU.</p>
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
   * <p>This is what a CPU session applies unless a caller requests another strategy, and what
   * {@link #defaultFor(List)} derives for any execution provider it cannot classify as an
   * accelerator. It fragments on natural text: on English news sentences, a call of 64 inputs
   * splits into roughly 33 groups and a call of 8 into roughly 7, so most inferences have one or
   * two rows. On the CPU that is still faster than padding; on a GPU it is where six to eight times
   * the throughput goes missing, which is why {@link #defaultFor(List)} picks {@link #LONGEST}
   * there.</p>
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
  MAX_LENGTH;

  /**
   * {@return the strategy to apply where a caller states none, derived from the execution providers
   * a session was configured with}
   *
   * <p>{@link #LONGEST} for an accelerator placement and {@link #EXACT_LENGTH} for anything else,
   * since the two placements want different tensor shapes, by a wide margin in both cases. On
   * an RTX 4080 SUPER through CUDA, a call of 32 inputs ran at 15910 embeddings per second under
   * {@link #LONGEST} against 2629 under {@link #EXACT_LENGTH} and 1955 for a loop of one embed per
   * input, a factor of six to eight: one wide tensor keeps the device occupied, while a stream of
   * narrow ones leaves it idle between kernels and flattens out near 1950 at any call size. With the
   * CPU execution provider on the same machine at eight intra-op threads, a call of 128 inputs ran
   * at 690 under {@link #LONGEST} against 1233 under {@link #EXACT_LENGTH}, a factor of 0.81, and
   * 0.66 at four intra-op threads: a CPU does the arithmetic of the padded positions and gets no
   * occupancy in return. {@link #MAX_LENGTH} is around twenty times behind on either placement, so
   * it is not derived at all and remains the explicit choice of a fixed-shape graph or device.</p>
   *
   * <p>An execution provider from an addon states its own placement, through
   * {@link opennlp.dl.ExecutionProviderConfigurer#placement(opennlp.tools.util.ext.ProviderSpec)},
   * and is derived for like any other: the {@code openvino} configurer of
   * {@code opennlp-dl-openvino} answers {@link #LONGEST} for {@code device_type=GPU.0} and
   * {@link #EXACT_LENGTH} for {@code device_type=CPU}, which are the same execution provider on two
   * placements. Nothing in {@code opennlp-dl} holds the id.</p>
   *
   * <p>An execution provider whose placement is unstated, which is the case for an id no configurer
   * answers to and for every configurer written before that method existed, is treated as the CPU
   * case and gets {@link #EXACT_LENGTH}. Three reasons for the conservative answer over the fast
   * one:</p>
   *
   * <ul>
   *   <li>{@link #EXACT_LENGTH} needs no padding token in the vocabulary and no assumption about a
   *       graph honoring {@code attention_mask}, so it is the one strategy that cannot be wrong for
   *       an unfamiliar model on an unfamiliar device.</li>
   *   <li>It is what a caller gets from this class today, so an addon on the class path changes no
   *       vectors by its presence. On a GPU the tensor width moves a component by about
   *       {@code 1.2e-4}, so turning a default into padding is a change anyone caching vectors has
   *       to plan for.</li>
   *   <li>The two errors do not cost the same. Guessing the accelerator answer costs up to a third
   *       of the throughput where the guess is wrong, as the CPU figures above show, while the
   *       conservative answer leaves throughput on the table and one constructor argument takes
   *       it.</li>
   * </ul>
   *
   * <p>Refining this for a further execution provider is the addon's own work: its configurer
   * overrides
   * {@link opennlp.dl.ExecutionProviderConfigurer#placement(opennlp.tools.util.ext.ProviderSpec)}
   * and reads whichever of its provider options decides the placement. No edit to this class or to
   * {@link ExecutionProviders} is involved.</p>
   *
   * @param executionProviders The execution providers of the session, in priority order, as
   *     {@link ExecutionProviders#resolve(InferenceOptions)} returns them. An empty list is the CPU
   *     case, since ONNX Runtime places such a session on the CPU. Must not be {@code null} or start
   *     with a {@code null} element.
   * @throws IllegalArgumentException Thrown if {@code executionProviders} is {@code null} or its
   *     first element is {@code null}.
   * @see ExecutionProviders#runsOnAccelerator(List)
   * @see SentenceVectorsDL#DEFAULT_PADDING
   * @since 3.0.0
   */
  public static PaddingStrategy defaultFor(
      final List<ExecutionProviderRequest> executionProviders) {
    return ExecutionProviders.runsOnAccelerator(executionProviders) ? LONGEST : EXACT_LENGTH;
  }
}
