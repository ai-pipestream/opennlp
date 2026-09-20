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
package opennlp.dl.vectors;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.tools.embeddings.TextEmbedderProvider;
import opennlp.tools.util.ext.ProviderSpec;
import opennlp.tools.util.ext.Providers;

import static opennlp.dl.vectors.OnnxTextEmbedderProvider.GPU_DEVICE_ID_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.GPU_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.LOWER_CASE_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.MAX_LENGTH_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.NAME;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.NORMALIZE_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.PADDING_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.POOLING_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.VOCABULARY_OPTION;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The support, availability and registration of the ONNX provider, without a model. */
class OnnxTextEmbedderProviderTest {

  private static final Map<String, String> OPTIONS = Map.of(VOCABULARY_OPTION, "vocab.txt");

  private final OnnxTextEmbedderProvider provider = new OnnxTextEmbedderProvider();

  @Test
  void testIdentityAndAvailability() {
    assertEquals(NAME, provider.name());
    assertEquals(0, provider.priority());
    assertTrue(provider.isAvailable(), "onnxruntime is on the test classpath");
  }

  @ParameterizedTest
  @ValueSource(strings = {"model.onnx", "MODEL.ONNX", "dir/model.Onnx"})
  void testSupportsOnnxFilesWithItsOwnOptions(final String model) {
    assertTrue(provider.supports(ProviderSpec.of(Path.of(model), OPTIONS)));
    assertTrue(provider.supports(ProviderSpec.of(Path.of(model))));
    assertTrue(provider.supports(ProviderSpec.of(Path.of(model),
        Map.of(VOCABULARY_OPTION, "vocab.txt", LOWER_CASE_OPTION, "false"))));
    assertTrue(provider.supports(ProviderSpec.of(Path.of(model),
        Map.of(VOCABULARY_OPTION, "vocab.txt", POOLING_OPTION, "cls", NORMALIZE_OPTION, "false",
            MAX_LENGTH_OPTION, "256"))));
    assertTrue(provider.supports(ProviderSpec.of(Path.of(model),
        Map.of(VOCABULARY_OPTION, "vocab.txt", PADDING_OPTION, "longest"))));
    assertTrue(provider.supports(ProviderSpec.of(Path.of(model),
        Map.of(VOCABULARY_OPTION, "vocab.txt", GPU_OPTION, "true",
            GPU_DEVICE_ID_OPTION, "1"))));
  }

  @ParameterizedTest
  @ValueSource(strings = {"model.bin", "model.onnx.bak", "onnx", "model.pt"})
  void testDoesNotSupportOtherModelFiles(final String model) {
    assertFalse(provider.supports(ProviderSpec.of(Path.of(model), OPTIONS)));
  }

  @Test
  void testDoesNotSupportForeignOptionsOrLocations() {
    assertFalse(provider.supports(ProviderSpec.of(Path.of("model.onnx"),
        Map.of(VOCABULARY_OPTION, "v", "device", "gpu"))), "a foreign option");
    assertFalse(provider.supports(ProviderSpec.of(OPTIONS)), "no location");
    assertFalse(provider.supports(ProviderSpec.of(URI.create("https://example.org/model.onnx"),
        OPTIONS)), "not a local file");
    assertThrows(IllegalArgumentException.class, () -> provider.supports(null));
  }

  /**
   * The execution provider options, checked without loading a model. {@value
   * OnnxTextEmbedderProvider#GPU_OPTION} defaults to {@code false}, which is the only default that
   * keeps the SPI path on the CPU, and a malformed value of either option is rejected before any
   * session is created rather than quietly ignored.
   */
  @Test
  void testGpuOptionsAreValidatedAndDefaultToTheCpu(@TempDir final Path dir) throws Exception {
    final Path model = Files.createFile(dir.resolve("model.onnx"));
    assertThrows(IllegalArgumentException.class, () -> provider.create(ProviderSpec.of(model,
        Map.of(VOCABULARY_OPTION, "v.txt", GPU_OPTION, "yes"))), "gpu must be true or false");
    assertThrows(IllegalArgumentException.class, () -> provider.create(ProviderSpec.of(model,
        Map.of(VOCABULARY_OPTION, "v.txt", GPU_DEVICE_ID_OPTION, "-1"))), "a negative device id");
    assertThrows(IllegalArgumentException.class, () -> provider.create(ProviderSpec.of(model,
        Map.of(VOCABULARY_OPTION, "v.txt", GPU_DEVICE_ID_OPTION, "one"))), "not an integer");
    // An empty file is not a model, so this reaches ONNX Runtime and fails there rather than on
    // an option: proof that gpu=false does not trip the CUDA path on a CPU-only runtime.
    assertThrows(IOException.class, () -> provider.create(ProviderSpec.of(model,
        Map.of(VOCABULARY_OPTION, "v.txt", GPU_OPTION, "false", GPU_DEVICE_ID_OPTION, "3"))));
  }

  /** The provider is registered in this module and is the one selected for an ONNX model. */
  @Test
  void testRegisteredAndSelectedForOnnxModels() {
    final Providers<TextEmbedderProvider> providers = Providers.of(TextEmbedderProvider.class);
    assertInstanceOf(OnnxTextEmbedderProvider.class, providers.byName(NAME).orElseThrow());
    assertInstanceOf(OnnxTextEmbedderProvider.class,
        providers.select(ProviderSpec.of(Path.of("model.onnx"), OPTIONS)));
  }
}
