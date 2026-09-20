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
import java.util.Collections;
import java.util.List;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.OrtCUDAProviderOptions;

/**
 * Session options that record what was done to them instead of doing it, so a test can assert which
 * ONNX Runtime calls a configuration made and in which order.
 *
 * <p>This exists because onnxruntime 1.29.0 lets a test read almost nothing back off session
 * options: {@code OrtSession} has no {@code getProviders()}, {@code SessionOptions} has no getter
 * for the execution providers appended to it or for its thread counts, and
 * {@code getConfigEntries()} stays empty for every call that goes straight to the native layer.
 * {@code SessionOptions} is not final and neither are its setters, so the calls can be observed
 * where they are made.</p>
 *
 * <p>The overridden methods do not call {@code super}, so nothing reaches the native layer and no
 * execution provider has to be installed on the machine for a test to check that it would have been
 * requested. {@link #addConfigEntry(String, String)} is the exception: it records and delegates,
 * since config entries are the one thing ONNX Runtime does read back.</p>
 */
public class RecordingSessionOptions extends OrtSession.SessionOptions {

  private final List<String> calls = new ArrayList<>();

  /** {@return the calls this instance received, in order. Unmodifiable and never {@code null}} */
  public List<String> calls() {
    return Collections.unmodifiableList(calls);
  }

  /** {@inheritDoc} */
  @Override
  public void addCPU(final boolean useArena) {
    calls.add("addCPU(" + useArena + ")");
  }

  /** {@inheritDoc} */
  @Override
  public void addCUDA() {
    calls.add("addCUDA()");
  }

  /** {@inheritDoc} */
  @Override
  public void addCUDA(final int deviceId) {
    calls.add("addCUDA(" + deviceId + ")");
  }

  /** {@inheritDoc} */
  @Override
  public void addCUDA(final OrtCUDAProviderOptions providerOptions) {
    calls.add("addCUDA(" + providerOptions.getOptionsString() + ")");
  }

  /** {@inheritDoc} */
  @Override
  public void addOpenVINO(final String deviceId) {
    calls.add("addOpenVINO(" + deviceId + ")");
  }

  /** {@inheritDoc} */
  @Override
  public void setIntraOpNumThreads(final int numThreads) {
    calls.add("setIntraOpNumThreads(" + numThreads + ")");
  }

  /** {@inheritDoc} */
  @Override
  public void setInterOpNumThreads(final int numThreads) {
    calls.add("setInterOpNumThreads(" + numThreads + ")");
  }

  /** {@inheritDoc} */
  @Override
  public void setOptimizationLevel(final OrtSession.SessionOptions.OptLevel level) {
    calls.add("setOptimizationLevel(" + level + ")");
  }

  /**
   * {@inheritDoc}
   * Recorded and applied, since config entries are readable through
   * {@link OrtSession.SessionOptions#getConfigEntries()}.
   */
  @Override
  public void addConfigEntry(final String key, final String value) throws OrtException {
    calls.add("addConfigEntry(" + key + "=" + value + ")");
    super.addConfigEntry(key, value);
  }
}
