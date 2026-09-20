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

package opennlp.dl;

import java.util.ArrayList;
import java.util.List;

import ai.onnxruntime.OrtSession;

/**
 * The settings a deep-learning component reads before it creates its ONNX Runtime session and
 * while it prepares its input: which execution providers to run on, how the session may use
 * threads, and how text is normalized, split and cased.
 *
 * <p><b>Execution providers.</b> {@link #setExecutionProviders(List)} takes an ordered list, which
 * is the shape ONNX Runtime has: it keeps the execution providers of a session in the order they
 * were appended and runs a node on the first one that accepts it. An empty list, the default, asks
 * for nothing and leaves the session where ONNX Runtime puts it, which is the CPU. The deprecated
 * {@link #setGpu(boolean)} stands for a single {@value ExecutionProviders#CUDA} request and is read
 * only while the list is empty; see {@link ExecutionProviders#resolve(InferenceOptions)} for the
 * precedence.</p>
 *
 * <p><b>Session configuration.</b> The thread counts and the graph optimization level are unset by
 * default, and every one of them that is unset leaves ONNX Runtime's own default in place. Setting
 * one is not reversible: there is no value that means "unset again", the same way
 * {@link #setLowerCase(boolean)} cannot be taken back.</p>
 *
 * <p>Not every component reads every setting. A component states in its own documentation which of
 * these it consults, and nothing is read after construction, so an instance may be reused or
 * changed afterwards.</p>
 *
 * <p>This class is not thread-safe.</p>
 */
public class InferenceOptions {

  /**
   * The largest thread count {@link #setIntraOpNumThreads(int)} and
   * {@link #setInterOpNumThreads(int)} accept. ONNX Runtime starts that many operating system
   * threads when the session is created, and the bound is above the hardware thread count of any
   * current machine, so a larger value is a mistyped number rather than a configuration and is
   * rejected instead of exhausting the machine.
   */
  public static final int MAX_NUM_THREADS = 1024;

  private boolean includeAttentionMask = true;
  private boolean includeTokenTypeIds = true;
  private boolean gpu;
  private int gpuDeviceId = 0;
  private int documentSplitSize = 250;
  private int splitOverlapSize = 50;
  private Boolean lowerCase;
  private boolean normalizeWhitespace;
  private boolean normalizeDashes;
  private List<ExecutionProviderRequest> executionProviders = List.of();
  private Integer intraOpNumThreads;
  private Integer interOpNumThreads;
  private OrtSession.SessionOptions.OptLevel optimizationLevel;

  public boolean isIncludeAttentionMask() {
    return includeAttentionMask;
  }

  public void setIncludeAttentionMask(boolean includeAttentionMask) {
    this.includeAttentionMask = includeAttentionMask;
  }

  public boolean isIncludeTokenTypeIds() {
    return includeTokenTypeIds;
  }

  public void setIncludeTokenTypeIds(boolean includeTokenTypeIds) {
    this.includeTokenTypeIds = includeTokenTypeIds;
  }

  /**
   * {@return whether the deprecated GPU flag is set, which stands for a single request for the
   * {@value ExecutionProviders#CUDA} execution provider on {@link #getGpuDeviceId()}}
   *
   * @deprecated A flag can name one of the fourteen execution providers ONNX Runtime's Java API
   *     exposes and cannot order a fallback behind it. Use
   *     {@link #setExecutionProviders(List)} with
   *     {@link ExecutionProviderRequest#of(String, java.util.Map)} and
   *     {@link ExecutionProviders#CUDA} instead. This flag keeps working and keeps selecting CUDA,
   *     and {@link ExecutionProviders#resolve(InferenceOptions)} states when it is read.
   */
  @Deprecated(since = "3.0.0", forRemoval = true)
  public boolean isGpu() {
    return gpu;
  }

  /**
   * Requests the {@value ExecutionProviders#CUDA} execution provider on {@link #getGpuDeviceId()}.
   *
   * @param gpu {@code true} to run on CUDA.
   *
   * @deprecated A flag can name one of the fourteen execution providers ONNX Runtime's Java API
   *     exposes and cannot order a fallback behind it. Use
   *     {@link #setExecutionProviders(List)} with
   *     {@link ExecutionProviderRequest#of(String, java.util.Map)} and
   *     {@link ExecutionProviders#CUDA} instead. This setter keeps working and keeps selecting
   *     CUDA, and {@link ExecutionProviders#resolve(InferenceOptions)} states when it is read.
   */
  @Deprecated(since = "3.0.0", forRemoval = true)
  public void setGpu(boolean gpu) {
    this.gpu = gpu;
  }

  /**
   * {@return the CUDA device the deprecated GPU flag runs on, {@code 0} by default}
   *
   * @deprecated Part of the flag replaced by {@link #setExecutionProviders(List)}. The device is
   *     the {@value CudaExecutionProviderConfigurer#DEVICE_ID_OPTION} provider option of a
   *     {@value ExecutionProviders#CUDA} request, which is also where every other CUDA provider
   *     option goes.
   */
  @Deprecated(since = "3.0.0", forRemoval = true)
  public int getGpuDeviceId() {
    return gpuDeviceId;
  }

  /**
   * Sets the CUDA device the deprecated GPU flag runs on.
   *
   * @param gpuDeviceId The device id.
   *
   * @deprecated Part of the flag replaced by {@link #setExecutionProviders(List)}. The device is
   *     the {@value CudaExecutionProviderConfigurer#DEVICE_ID_OPTION} provider option of a
   *     {@value ExecutionProviders#CUDA} request, which is also where every other CUDA provider
   *     option goes.
   */
  @Deprecated(since = "3.0.0", forRemoval = true)
  public void setGpuDeviceId(int gpuDeviceId) {
    this.gpuDeviceId = gpuDeviceId;
  }

  /**
   * {@return the execution providers to append to the session, in priority order. Empty by
   *     default, which leaves the session where ONNX Runtime puts it. Unmodifiable and never
   *     {@code null}}
   */
  public List<ExecutionProviderRequest> getExecutionProviders() {
    return executionProviders;
  }

  /**
   * Sets the execution providers to append to the session, in priority order.
   *
   * <p>ONNX Runtime runs a node on the first execution provider of the list that accepts it and
   * takes the next for the rest, so the order is the fallback order:
   * {@code [cuda, cpu]} asks for CUDA where it can run and the CPU for the rest. Every id is
   * resolved through the {@link ExecutionProviderConfigurer} SPI, which
   * {@code opennlp-dl} answers for {@value ExecutionProviders#CPU} and
   * {@value ExecutionProviders#CUDA} and an addon answers for anything else. An id no configurer
   * answers to is reported when the session is configured, and so is an execution provider ONNX
   * Runtime cannot register; neither one is quietly dropped, and there is no fallback to the CPU
   * beyond the one this list asks for.</p>
   *
   * <p>An empty list, the default, asks for nothing, which is not the same as asking for
   * {@value ExecutionProviders#CPU}: ONNX Runtime appends the CPU execution provider to every
   * session on its own, so the session runs on the CPU either way, but the empty list adds no call
   * at all and leaves the session configuration of a component exactly as it was before execution
   * providers could be chosen.</p>
   *
   * <p>While this list is not empty, the deprecated {@link #setGpu(boolean)} is ignored.</p>
   *
   * @param executionProviders The requests, in priority order, copied into these options. Must not
   *     be {@code null} or hold a {@code null} element.
   * @throws IllegalArgumentException Thrown if {@code executionProviders} is {@code null} or holds
   *     a {@code null} element.
   */
  public void setExecutionProviders(final List<ExecutionProviderRequest> executionProviders) {
    if (executionProviders == null) {
      throw new IllegalArgumentException("The executionProviders must not be null.");
    }
    final List<ExecutionProviderRequest> copy = new ArrayList<>(executionProviders.size());
    for (final ExecutionProviderRequest request : executionProviders) {
      if (request == null) {
        throw new IllegalArgumentException("The executionProviders must not hold a null element.");
      }
      copy.add(request);
    }
    this.executionProviders = List.copyOf(copy);
  }

  /**
   * Appends one execution provider behind those already requested, so a list can be built up one
   * call at a time. The order of the calls is the priority order.
   *
   * @param executionProvider The request to append. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code executionProvider} is {@code null}.
   */
  public void addExecutionProvider(final ExecutionProviderRequest executionProvider) {
    if (executionProvider == null) {
      throw new IllegalArgumentException("The executionProvider must not be null.");
    }
    final List<ExecutionProviderRequest> appended = new ArrayList<>(executionProviders);
    appended.add(executionProvider);
    this.executionProviders = List.copyOf(appended);
  }

  /**
   * {@return the number of threads one operator of the session may run on, or {@code null} to
   *     leave ONNX Runtime's default in place, which is unset by default}
   */
  public Integer getIntraOpNumThreads() {
    return intraOpNumThreads;
  }

  /**
   * Sets the number of threads one operator of the session may run on.
   *
   * <p>This is the setting that matters most on the CPU. ONNX Runtime's default is {@code 0},
   * which is one thread per hardware thread of the machine, and its threads spin while they wait,
   * so on a many-core machine running small tensors the operators oversubscribe the machine and
   * spend their time handing work to each other. Measured on a sixteen core, thirty-two thread
   * machine, embedding throughput rose by a factor of two to four when this was set to a small
   * value instead of being left at the default. The right value depends on the tensor sizes and on
   * how many sessions and callers share the machine, so there is no default here beyond ONNX
   * Runtime's own.</p>
   *
   * @param intraOpNumThreads The thread count. {@code 0} asks ONNX Runtime to choose, which is
   *     what it does when this is not set at all; otherwise a positive count of at most
   *     {@value #MAX_NUM_THREADS}.
   * @throws IllegalArgumentException Thrown if {@code intraOpNumThreads} is negative or greater
   *     than {@value #MAX_NUM_THREADS}. ONNX Runtime accepts a negative count without complaint
   *     and acts on it later, so it is rejected here.
   */
  public void setIntraOpNumThreads(final int intraOpNumThreads) {
    this.intraOpNumThreads = validateNumThreads(intraOpNumThreads, "intraOpNumThreads");
  }

  /**
   * {@return the number of threads the session may run independent operators on, or {@code null}
   *     to leave ONNX Runtime's default in place, which is unset by default}
   */
  public Integer getInterOpNumThreads() {
    return interOpNumThreads;
  }

  /**
   * Sets the number of threads the session may run independent operators on. It only has an effect
   * where the graph has branches that can run at the same time and the session runs operators in
   * parallel; the transformer graphs these components load are mostly a chain, so
   * {@link #setIntraOpNumThreads(int)} is the setting that moves their throughput.
   *
   * @param interOpNumThreads The thread count. {@code 0} asks ONNX Runtime to choose, which is
   *     what it does when this is not set at all; otherwise a positive count of at most
   *     {@value #MAX_NUM_THREADS}.
   * @throws IllegalArgumentException Thrown if {@code interOpNumThreads} is negative or greater
   *     than {@value #MAX_NUM_THREADS}. ONNX Runtime accepts a negative count without complaint
   *     and acts on it later, so it is rejected here.
   */
  public void setInterOpNumThreads(final int interOpNumThreads) {
    this.interOpNumThreads = validateNumThreads(interOpNumThreads, "interOpNumThreads");
  }

  /**
   * {@return the graph optimizations ONNX Runtime applies when it loads the model, or {@code null}
   *     to leave its default in place, which is unset by default}
   */
  public OrtSession.SessionOptions.OptLevel getOptimizationLevel() {
    return optimizationLevel;
  }

  /**
   * Sets the graph optimizations ONNX Runtime applies when it loads the model.
   *
   * <p>ONNX Runtime already applies every optimization by default, so this exists to take
   * optimizations away, which is what a graph whose exported operators an optimization rewrites
   * incorrectly needs, or what an investigation of such a graph needs. Raising it has nothing to
   * raise.</p>
   *
   * @param optimizationLevel The level to apply. Must not be {@code null}; not calling this at all
   *     is how ONNX Runtime's default is kept.
   * @throws IllegalArgumentException Thrown if {@code optimizationLevel} is {@code null}.
   */
  public void setOptimizationLevel(final OrtSession.SessionOptions.OptLevel optimizationLevel) {
    if (optimizationLevel == null) {
      throw new IllegalArgumentException("The optimizationLevel must not be null.");
    }
    this.optimizationLevel = optimizationLevel;
  }

  /**
   * Checks a thread count at the boundary, since ONNX Runtime accepts a negative one and an absurd
   * one on the options object and only acts on it when the session is created.
   *
   * @param numThreads The requested count.
   * @param name The name of the setting, for the message.
   * @return The count.
   * @throws IllegalArgumentException Thrown if the count is negative or above
   *     {@value #MAX_NUM_THREADS}.
   */
  private static int validateNumThreads(final int numThreads, final String name) {
    if (numThreads < 0) {
      throw new IllegalArgumentException("The " + name + " must not be negative.");
    }
    if (numThreads > MAX_NUM_THREADS) {
      throw new IllegalArgumentException(
          "The " + name + " must be at most " + MAX_NUM_THREADS + ".");
    }
    return numThreads;
  }

  public int getDocumentSplitSize() {
    return documentSplitSize;
  }

  public void setDocumentSplitSize(int documentSplitSize) {
    this.documentSplitSize = documentSplitSize;
  }

  public int getSplitOverlapSize() {
    return splitOverlapSize;
  }

  public void setSplitOverlapSize(int splitOverlapSize) {
    this.splitOverlapSize = splitOverlapSize;
  }

  /** {@return whether input whitespace is normalized to ASCII spaces before inference} */
  public boolean isNormalizeWhitespace() {
    return normalizeWhitespace;
  }

  /**
   * Replaces every Unicode whitespace character in the input with an ASCII space before inference.
   * This is offset preserving (each whitespace code point maps to one space), so any spans a model
   * produces still align with the input. Off by default.
   *
   * <p>This is a one-for-one replacement, not the collapse-and-trim whitespace fold of the runtime
   * {@code TextNormalizer.whitespace()} normalizer: runs of whitespace are not merged and leading or
   * trailing whitespace is not removed, so offsets are preserved.</p>
   *
   * @param normalizeWhitespace Whether to normalize whitespace.
   */
  public void setNormalizeWhitespace(boolean normalizeWhitespace) {
    this.normalizeWhitespace = normalizeWhitespace;
  }

  /** {@return whether input dashes are normalized to the ASCII hyphen before inference} */
  public boolean isNormalizeDashes() {
    return normalizeDashes;
  }

  /**
   * Replaces Unicode dashes in the input with the ASCII hyphen-minus before inference. This is
   * offset preserving for the dash characters in the Basic Multilingual Plane (the common case).
   * The mathematical minus signs are not affected. Off by default.
   *
   * <p>A supplementary-plane dash shrinks from two chars to one, which shifts later offsets, so
   * with this enabled {@code find(...)} reports offsets into the normalized text in that case. Use
   * {@code NameFinderDL.findInOriginal(...)} for offsets mapped back to the original input.</p>
   *
   * @param normalizeDashes Whether to normalize dashes.
   */
  public void setNormalizeDashes(boolean normalizeDashes) {
    this.normalizeDashes = normalizeDashes;
  }

  /**
   * Returns whether tokenization should lower case the input text and strip
   * accents, as required by uncased models.
   *
   * @return {@code Boolean.TRUE} for uncased models, {@code Boolean.FALSE} for
   *     cased models, or {@code null} if not set, in which case each component
   *     applies the default that matches its commonly used models.
   */
  public Boolean getLowerCase() {
    return lowerCase;
  }

  /**
   * Sets whether tokenization should lower case the input text and strip
   * accents. Set {@code true} for uncased models and {@code false} for cased
   * models. If not set, each component applies the default that matches its
   * commonly used models.
   *
   * @param lowerCase Whether to lower case the input text during tokenization.
   */
  public void setLowerCase(boolean lowerCase) {
    this.lowerCase = lowerCase;
  }

}
