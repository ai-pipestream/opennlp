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

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What an {@link ExecutionProviderRequest} accepts and what it keeps.
 *
 * <p>The request is the boundary the caller's values pass through, so everything a bad value could
 * do later is prevented here: an id that no lookup could ever resolve, an option map that mutates
 * after the request was built, and the null keys and values ONNX Runtime's option maps cannot
 * carry.</p>
 */
class ExecutionProviderRequestTest {

  @Test
  void testIdAndOptionsAreKept() {
    final ExecutionProviderRequest request = ExecutionProviderRequest.of(ExecutionProviders.CUDA,
        Map.of(CudaExecutionProviderConfigurer.DEVICE_ID_OPTION, "3"));

    assertEquals(ExecutionProviders.CUDA, request.id());
    assertEquals(Map.of(CudaExecutionProviderConfigurer.DEVICE_ID_OPTION, "3"), request.options());
  }

  @Test
  void testRequestWithoutOptionsHasAnEmptyOptionMap() {
    assertEquals(Map.of(), ExecutionProviderRequest.of(ExecutionProviders.CPU).options());
  }

  /** An empty option map is a legal request, not a missing one. */
  @Test
  void testEmptyOptionMapIsAccepted() {
    final ExecutionProviderRequest request =
        ExecutionProviderRequest.of(ExecutionProviders.CPU, Map.of());

    assertEquals(ExecutionProviders.CPU, request.id());
    assertEquals(Map.of(), request.options());
  }

  /** The option map is copied, so a caller that keeps mutating its map cannot change a request. */
  @Test
  void testOptionsAreCopiedFromTheCaller() {
    final Map<String, String> mutable = new HashMap<>();
    mutable.put(CudaExecutionProviderConfigurer.DEVICE_ID_OPTION, "0");
    final ExecutionProviderRequest request =
        ExecutionProviderRequest.of(ExecutionProviders.CUDA, mutable);

    mutable.put(CudaExecutionProviderConfigurer.DEVICE_ID_OPTION, "7");
    mutable.put("added", "later");

    assertEquals(Map.of(CudaExecutionProviderConfigurer.DEVICE_ID_OPTION, "0"), request.options());
  }

  @Test
  void testOptionsAreUnmodifiable() {
    final Map<String, String> options =
        ExecutionProviderRequest.of(ExecutionProviders.CPU, Map.of("a", "b")).options();

    assertThrows(UnsupportedOperationException.class, () -> options.put("c", "d"));
  }

  /** The options keep the order they were given in, which is the order they reach ONNX Runtime. */
  @Test
  void testOptionsKeepTheirOrder() {
    final Map<String, String> ordered = new LinkedHashMap<>();
    ordered.put("z", "1");
    ordered.put("a", "2");
    ordered.put("m", "3");

    assertIterableEquals(ordered.keySet(),
        ExecutionProviderRequest.of("anything", ordered).options().keySet());
  }

  @Test
  void testNullIdRejected() {
    assertThrows(IllegalArgumentException.class, () -> ExecutionProviderRequest.of(null));
    assertThrows(IllegalArgumentException.class,
        () -> ExecutionProviderRequest.of(null, Map.of()));
  }

  @Test
  void testNullOptionMapRejected() {
    assertThrows(IllegalArgumentException.class,
        () -> ExecutionProviderRequest.of(ExecutionProviders.CPU, null));
  }

  /**
   * A null option name or value is rejected, since ONNX Runtime's provider option maps are string
   * to string and have nowhere to put either.
   */
  @Test
  void testNullOptionNameOrValueRejected() {
    final Map<String, String> nullValue = new HashMap<>();
    nullValue.put("key", null);
    final Map<String, String> nullKey = new HashMap<>();
    nullKey.put(null, "value");

    assertThrows(IllegalArgumentException.class,
        () -> ExecutionProviderRequest.of(ExecutionProviders.CPU, nullValue));
    assertThrows(IllegalArgumentException.class,
        () -> ExecutionProviderRequest.of(ExecutionProviders.CPU, nullKey));
  }

  @Test
  void testBlankOptionNameRejected() {
    assertThrows(IllegalArgumentException.class,
        () -> ExecutionProviderRequest.of(ExecutionProviders.CPU, Map.of("  ", "value")));
  }

  /**
   * An id that no provider lookup could resolve is rejected where it is written rather than where
   * it is used, because the rules are the rules of
   * {@link opennlp.tools.util.ext.Providers#byName(String)}.
   *
   * @param id The unusable id.
   */
  @ParameterizedTest
  @ValueSource(strings = {"", " ", "cu da", "cuda!", "cuda/0", "cu:da", "é", "😀",
      "cuda\n"})
  void testUnusableIdRejected(final String id) {
    final IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
        () -> ExecutionProviderRequest.of(id));

    assertTrue(failure.getMessage().contains("id"), failure.getMessage());
  }

  @ParameterizedTest
  @ValueSource(strings = {"cpu", "cuda", "open_vino", "open-vino", "open.vino", "A1", "0"})
  void testUsableIdAccepted(final String id) {
    assertEquals(id, ExecutionProviderRequest.of(id).id());
  }

  @Test
  void testIdOfMaximumLengthAccepted() {
    final String id = "a".repeat(ExecutionProviderRequest.MAX_ID_LENGTH);

    assertEquals(id, ExecutionProviderRequest.of(id).id());
  }

  @Test
  void testIdOverMaximumLengthRejected() {
    final String id = "a".repeat(ExecutionProviderRequest.MAX_ID_LENGTH + 1);

    assertThrows(IllegalArgumentException.class, () -> ExecutionProviderRequest.of(id));
  }

  /** Ids are case-sensitive, as provider names are. */
  @Test
  void testIdsAreCaseSensitive() {
    assertNotEquals(ExecutionProviderRequest.of("cuda"), ExecutionProviderRequest.of("CUDA"));
  }

  @Test
  void testEqualsAndHashCode() {
    final ExecutionProviderRequest first =
        ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of("device_id", "1"));
    final ExecutionProviderRequest same =
        ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of("device_id", "1"));
    final ExecutionProviderRequest otherValue =
        ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of("device_id", "2"));
    final ExecutionProviderRequest otherId =
        ExecutionProviderRequest.of(ExecutionProviders.CPU, Map.of("device_id", "1"));

    assertEquals(first, same);
    assertEquals(first.hashCode(), same.hashCode());
    assertNotEquals(first, otherValue);
    assertNotEquals(first, otherId);
    assertNotEquals(first, ExecutionProviderRequest.of(ExecutionProviders.CUDA));
    assertNotEquals(null, first);
    assertNotEquals("cuda[device_id]", first);
  }

  /**
   * The rendering names the id and the option names and leaves the option values out, the way
   * {@link opennlp.tools.util.ext.ProviderSpec#toString()} leaves them out, since a provider option
   * can carry a credential and this rendering goes into a log entry.
   */
  @Test
  void testToStringNamesTheIdAndTheOptionNamesOnly() {
    final String rendered = ExecutionProviderRequest
        .of(ExecutionProviders.CUDA, Map.of("device_id", "1")).toString();

    assertTrue(rendered.contains(ExecutionProviders.CUDA), rendered);
    assertTrue(rendered.contains("device_id"), rendered);
    assertEquals("cuda[device_id]", rendered);
  }
}
