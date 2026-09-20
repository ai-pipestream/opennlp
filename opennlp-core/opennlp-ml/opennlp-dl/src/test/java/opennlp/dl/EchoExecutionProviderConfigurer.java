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

import java.util.Map;

import ai.onnxruntime.OrtSession;

import opennlp.tools.util.ext.ProviderSpec;

/**
 * A test-only {@link ExecutionProviderConfigurer} registered the way an addon registers one, in
 * {@code META-INF/services/opennlp.dl.ExecutionProviderConfigurer} of this module's test resources.
 * It answers to the id {@value #ID}, which nothing in {@code opennlp-dl} knows about, so a test that
 * resolves it has proved that a jar outside {@code opennlp-dl} could contribute an execution
 * provider the same way.
 *
 * <p>Instead of registering an execution provider, which would need one to be installed, it writes
 * each of its provider options as an ONNX Runtime session config entry named {@value #PREFIX}
 * followed by the option name. Config entries are the one part of
 * {@link OrtSession.SessionOptions} that ONNX Runtime reads back, through
 * {@link OrtSession.SessionOptions#getConfigEntries()}, so a plain set of session options is enough
 * to observe that this configurer ran and which option values reached it.</p>
 */
public final class EchoExecutionProviderConfigurer implements ExecutionProviderConfigurer {

  /** The id this configurer answers to. */
  public static final String ID = "opennlp-test-echo";

  /** The prefix of the session config entry each provider option is written to. */
  public static final String PREFIX = "opennlp.test.echo.";

  /** {@inheritDoc} */
  @Override
  public String name() {
    return ID;
  }

  /** {@inheritDoc} */
  @Override
  public boolean supports(final ProviderSpec spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    return spec.location().isEmpty();
  }

  /** {@inheritDoc} */
  @Override
  public ExecutionProvider create(final ProviderSpec spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    final Map<String, String> options = Map.copyOf(spec.options());
    return sessionOptions -> {
      if (sessionOptions == null) {
        throw new IllegalArgumentException("sessionOptions must not be null");
      }
      sessionOptions.addConfigEntry(PREFIX + "ran", ID);
      for (final Map.Entry<String, String> option : options.entrySet()) {
        sessionOptions.addConfigEntry(PREFIX + option.getKey(), option.getValue());
      }
    };
  }
}
