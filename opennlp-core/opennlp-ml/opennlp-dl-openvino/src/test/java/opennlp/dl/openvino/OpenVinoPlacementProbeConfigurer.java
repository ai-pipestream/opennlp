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

package opennlp.dl.openvino;

import opennlp.dl.ExecutionProvider;
import opennlp.dl.ExecutionProviderConfigurer;
import opennlp.dl.ExecutionProviderPlacement;
import opennlp.tools.util.ext.ProviderSpec;

/**
 * A test-only {@link ExecutionProviderConfigurer} that derives its placement with the real
 * {@link OpenVinoExecutionProviderConfigurer} and then registers no execution provider at all,
 * writing a session config entry instead.
 *
 * <p>It exists because a session cannot be created on OpenVINO here. The ONNX Runtime the build pins
 * was not built with that execution provider, so
 * {@link ai.onnxruntime.OrtSession.SessionOptions#addOpenVINO(String)} fails with
 * {@code ORT_EP_FAIL: Failed to find OpenVINO shared provider} and the session is never created,
 * which leaves the padding strategy of a running component unobservable on the real id. This
 * configurer keeps the part of the chain the placement decides, the derivation from the device type,
 * and substitutes only the native registration, so a component can be constructed and its batch plan
 * read. It is the same substitution {@code EchoExecutionProviderConfigurer} makes in
 * {@code opennlp-dl}.</p>
 *
 * <p>What it therefore does not prove is that OpenVINO runs anything. That the device type reaches
 * {@code addOpenVINO} is observed on the real configurer, in
 * {@link OpenVinoExecutionProviderConfigurerTest}, through {@code RecordingSessionOptions}.</p>
 */
public final class OpenVinoPlacementProbeConfigurer implements ExecutionProviderConfigurer {

  /** The id this configurer answers to. */
  public static final String ID = "openvino-placement-probe";

  /** The session config entry written in place of the OpenVINO execution provider. */
  public static final String KEY = "opennlp.test.openvino.device_type";

  private final OpenVinoExecutionProviderConfigurer openVino =
      new OpenVinoExecutionProviderConfigurer();

  /** {@inheritDoc} */
  @Override
  public String name() {
    return ID;
  }

  /** {@inheritDoc} */
  @Override
  public boolean supports(final ProviderSpec spec) {
    return openVino.supports(spec);
  }

  /**
   * {@inheritDoc}
   * Whatever {@link OpenVinoExecutionProviderConfigurer#placement(ProviderSpec)} states for the same
   * spec. This is the code under test; only the registration below is a substitute.
   */
  @Override
  public ExecutionProviderPlacement placement(final ProviderSpec spec) {
    return openVino.placement(spec);
  }

  /** {@inheritDoc} */
  @Override
  public ExecutionProvider create(final ProviderSpec spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    final String deviceType =
        spec.option(OpenVinoExecutionProviderConfigurer.DEVICE_TYPE_OPTION, "");
    return sessionOptions -> {
      if (sessionOptions == null) {
        throw new IllegalArgumentException("sessionOptions must not be null");
      }
      sessionOptions.addConfigEntry(KEY, deviceType);
    };
  }
}
