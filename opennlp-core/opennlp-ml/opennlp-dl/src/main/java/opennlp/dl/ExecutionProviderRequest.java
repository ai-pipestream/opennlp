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
import java.util.Objects;

import opennlp.tools.util.ext.ProviderSpec;

/**
 * A request for one ONNX Runtime execution provider: the id of the
 * {@link ExecutionProviderConfigurer} that knows how to register it, and the provider options to
 * register it with. Instances are immutable.
 *
 * <p>A session is configured with an ordered {@link java.util.List} of these, which is the shape
 * ONNX Runtime actually has: it keeps the execution providers of a session in the order they were
 * appended and runs a node on the first one that accepts it. A list of
 * {@code [cuda, cpu]} therefore means "CUDA where it can, CPU for the rest", which no single
 * choice can express.</p>
 *
 * <p>The options are strings on both sides, because ONNX Runtime's own provider option maps are
 * string keyed. Which keys a provider accepts is the configurer's business and, for a provider
 * whose options ONNX Runtime parses itself, the runtime's.</p>
 *
 * @see InferenceOptions#setExecutionProviders(java.util.List)
 * @see ExecutionProviders
 * @since 3.0.0
 */
public final class ExecutionProviderRequest {

  /**
   * The maximum length of an id, matching what
   * {@link opennlp.tools.util.ext.Providers#byName(String)} can resolve.
   */
  public static final int MAX_ID_LENGTH = 64;

  private final String id;
  private final ProviderSpec spec;

  private ExecutionProviderRequest(final String id, final Map<String, String> options) {
    if (id == null) {
      throw new IllegalArgumentException("id must not be null");
    }
    if (!isUsableId(id)) {
      throw new IllegalArgumentException("id must be 1 to " + MAX_ID_LENGTH
          + " characters of ASCII letters, digits, '.', '_' or '-': '" + id + "'");
    }
    if (options == null) {
      throw new IllegalArgumentException("options must not be null");
    }
    this.id = id;
    // ProviderSpec copies the map and rejects a null key, a null value and a blank key, which is
    // the same contract every other OpenNLP SPI option map has.
    this.spec = ProviderSpec.of(options);
  }

  /**
   * Requests an execution provider without provider options.
   *
   * @param id The case-sensitive id of the {@link ExecutionProviderConfigurer} to register it
   *     with. Must not be {@code null} and must be 1 to {@value #MAX_ID_LENGTH} characters of
   *     ASCII letters, digits, {@code .}, {@code _} or {@code -}.
   * @return The request. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code id} is {@code null} or not usable.
   */
  public static ExecutionProviderRequest of(final String id) {
    return new ExecutionProviderRequest(id, Map.of());
  }

  /**
   * Requests an execution provider with provider options.
   *
   * @param id The case-sensitive id of the {@link ExecutionProviderConfigurer} to register it
   *     with. Must not be {@code null} and must be 1 to {@value #MAX_ID_LENGTH} characters of
   *     ASCII letters, digits, {@code .}, {@code _} or {@code -}.
   * @param options The provider options, copied into the request. Must not be {@code null}, an
   *     option name must not be {@code null} or blank and an option value must not be
   *     {@code null}.
   * @return The request. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code id} is {@code null} or not usable, or if
   *     {@code options} is {@code null} or holds a null or blank name or a null value.
   */
  public static ExecutionProviderRequest of(final String id, final Map<String, String> options) {
    return new ExecutionProviderRequest(id, options);
  }

  /** {@return the case-sensitive id of the configurer that registers this execution provider} */
  public String id() {
    return id;
  }

  /** {@return the provider options, in insertion order. Unmodifiable and never {@code null}} */
  public Map<String, String> options() {
    return spec.options();
  }

  /**
   * {@return this request as the {@link ProviderSpec} an {@link ExecutionProviderConfigurer} is
   * created from: the provider options, and no location}
   */
  ProviderSpec spec() {
    return spec;
  }

  /**
   * Checks an id the way {@link opennlp.tools.util.ext.Providers} checks a provider name, so a
   * request that could never resolve is rejected where it is built rather than where it is used.
   * The scan is over characters and accepts only ASCII, which rejects a surrogate pair as well.
   *
   * @param candidate The id to check. Must not be {@code null}.
   * @return {@code true} if {@code candidate} can name a provider.
   */
  private static boolean isUsableId(final String candidate) {
    if (candidate.isEmpty() || candidate.length() > MAX_ID_LENGTH) {
      return false;
    }
    for (int character = 0; character < candidate.length(); character++) {
      final char current = candidate.charAt(character);
      final boolean ascii = current >= 'a' && current <= 'z' || current >= 'A' && current <= 'Z'
          || current >= '0' && current <= '9';
      if (!ascii && current != '.' && current != '_' && current != '-') {
        return false;
      }
    }
    return true;
  }

  /** {@inheritDoc} */
  @Override
  public boolean equals(final Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof ExecutionProviderRequest request)) {
      return false;
    }
    return id.equals(request.id) && spec.equals(request.spec);
  }

  /** {@inheritDoc} */
  @Override
  public int hashCode() {
    return Objects.hash(id, spec);
  }

  /**
   * {@inheritDoc}
   * The option names are rendered and their values are not, the same way
   * {@link ProviderSpec#toString()} leaves them out, since a provider option can carry a
   * credential such as an endpoint key.
   */
  @Override
  public String toString() {
    return id + spec.options().keySet();
  }
}
