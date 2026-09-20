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

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import ai.onnxruntime.OrtSession;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.dl.CudaExecutionProviderConfigurer;
import opennlp.dl.EchoExecutionProviderConfigurer;
import opennlp.dl.ExecutionProviderPlacement;
import opennlp.dl.ExecutionProviderRequest;
import opennlp.dl.ExecutionProviders;
import opennlp.dl.InferenceOptions;
import opennlp.dl.PlacementEchoExecutionProviderConfigurer;
import opennlp.dl.RecordingSessionOptions;
import opennlp.dl.SessionOptionsProbe;
import opennlp.tools.embeddings.TextEmbedder;
import opennlp.tools.util.ext.ProviderSpec;

import static opennlp.dl.vectors.OnnxTextEmbedderProvider.EXECUTION_PROVIDERS_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.GPU_DEVICE_ID_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.GPU_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.INTER_OP_NUM_THREADS_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.INTRA_OP_NUM_THREADS_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.OPTIMIZATION_LEVEL_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.PADDING_OPTION;
import static opennlp.dl.vectors.OnnxTextEmbedderProvider.VOCABULARY_OPTION;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The execution provider list and the session settings a spec can ask for, which is what makes the
 * {@code ExecutionProviderConfigurer} SPI reachable from the {@code TextEmbedderProvider} entry point
 * rather than only from the Java API.
 *
 * <p>Most of these read the {@link InferenceOptions} a spec produces through
 * {@link OnnxTextEmbedderProvider#inferenceOptions(ProviderSpec)}, because the rest of what a spec
 * asks for lands in native ONNX Runtime session state that the runtime does not read back. Two of
 * them go further: one hands those options to the same configuration step a component uses and reads
 * the calls ONNX Runtime received, and one constructs a real embedder on an addon execution provider
 * id and reads the batch plan the request decided.</p>
 */
class OnnxTextEmbedderProviderExecutionProviderTest {

  /** Inputs of tokenized length 3, 4, 5, 3, 4, 3, as in {@code PaddingStrategyDefaultTest}. */
  private static final List<String> DISTINCT = List.of("x", "hello hello",
      "hello hello world", "hello", "hello world", "world");

  /** One inference per distinct tokenized length, with no padding, in first-seen order. */
  private static final int[][] EXACT_SHAPES = {{3, 3}, {2, 4}, {1, 5}};

  /** One inference for the entire call, at the longest row of it. */
  private static final int[][] LONGEST_SHAPES = {{6, 5}};

  private static final String DEVICE_ID = CudaExecutionProviderConfigurer.DEVICE_ID_OPTION;

  private final OnnxTextEmbedderProvider provider = new OnnxTextEmbedderProvider();

  /** A spec for {@code model} with the given option name and value pairs. */
  private static ProviderSpec spec(final Path model, final String... options) {
    final Map<String, String> map = new LinkedHashMap<>();
    map.put(VOCABULARY_OPTION, "vocab.txt");
    for (int at = 0; at < options.length; at += 2) {
      map.put(options[at], options[at + 1]);
    }
    return ProviderSpec.of(model, map);
  }

  /** A spec with no real model behind it, which is all the option readers need. */
  private static ProviderSpec spec(final String... options) {
    return spec(Path.of("model.onnx"), options);
  }

  // Copied out of the classpath rather than resolved in place, because these tests also run from the
  // test-jar of this module, where the resource URI is not hierarchical.
  private static File model(final Path dir) throws IOException {
    final Path file = dir.resolve("tiny-vectors.onnx");
    try (InputStream is = Objects.requireNonNull(
        OnnxTextEmbedderProviderExecutionProviderTest.class
            .getResourceAsStream("/opennlp/dl/vectors/tiny-vectors.onnx"))) {
      Files.copy(is, file, StandardCopyOption.REPLACE_EXISTING);
    }
    return file.toFile();
  }

  /** The vocabulary of the vector tests of this module, with {@code [PAD]} at id {@code 0}. */
  private static void vocab(final Path dir) throws IOException {
    Files.write(dir.resolve("vocab.txt"), List.of("[PAD]", "unused1", "[UNK]", "[SEP]", "hello",
        "world", "unused2", "[CLS]"));
  }

  /**
   * {@return a {@value #EXECUTION_PROVIDERS_OPTION} value asking the test-only placement configurer
   * to state one placement}
   *
   * @param placement The placement to state.
   */
  private static String probing(final ExecutionProviderPlacement placement) {
    return PlacementEchoExecutionProviderConfigurer.ID + "("
        + PlacementEchoExecutionProviderConfigurer.PLACEMENT_OPTION + "=" + placement.name() + ")";
  }

  /** {@return the requests a spec option value resolves to, as a component would resolve them} */
  private List<ExecutionProviderRequest> resolved(final String... options) {
    return ExecutionProviders.resolve(provider.inferenceOptions(spec(options)));
  }

  /** Every new option is one this provider claims, and an unknown one is still refused. */
  @Test
  void testSupportsTheNewOptions() {
    assertTrue(provider.supports(spec(EXECUTION_PROVIDERS_OPTION, "cuda(device_id=1),cpu")));
    assertTrue(provider.supports(spec(INTRA_OP_NUM_THREADS_OPTION, "2")));
    assertTrue(provider.supports(spec(INTER_OP_NUM_THREADS_OPTION, "1")));
    assertTrue(provider.supports(spec(OPTIMIZATION_LEVEL_OPTION, "basic_opt")));
    assertTrue(provider.supports(spec(EXECUTION_PROVIDERS_OPTION, "cpu",
        INTRA_OP_NUM_THREADS_OPTION, "2", INTER_OP_NUM_THREADS_OPTION, "1",
        OPTIMIZATION_LEVEL_OPTION, "all_opt")));
    assertFalse(provider.supports(spec("executionProvider", "cpu")),
        "the option is named executionProviders, and a near miss is not guessed at");
  }

  /** A spec without the option asks for nothing, which is the state that leaves ORT to place it. */
  @Test
  void testNoExecutionProvidersOptionRequestsNothing() {
    assertTrue(provider.inferenceOptions(spec()).getExecutionProviders().isEmpty());
    assertTrue(resolved().isEmpty());
  }

  /** One id without provider options, which is the shortest thing the syntax accepts. */
  @Test
  void testASingleIdWithoutProviderOptions() {
    final List<ExecutionProviderRequest> requests =
        resolved(EXECUTION_PROVIDERS_OPTION, ExecutionProviders.CPU);
    assertEquals(List.of(ExecutionProviderRequest.of(ExecutionProviders.CPU)), requests);
    assertEquals(Map.of(), requests.get(0).options());
  }

  /**
   * The order of the list is the order of the value, which is the whole point of a list: ONNX Runtime
   * offers each node to the execution providers in the order they were appended.
   */
  @Test
  void testTheListKeepsTheOrderOfTheValue() {
    assertEquals(List.of(ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "1")),
            ExecutionProviderRequest.of(ExecutionProviders.CPU)),
        resolved(EXECUTION_PROVIDERS_OPTION, "cuda(device_id=1),cpu"));
    assertEquals(List.of(ExecutionProviderRequest.of(ExecutionProviders.CPU),
            ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "1"))),
        resolved(EXECUTION_PROVIDERS_OPTION, "cpu,cuda(device_id=1)"),
        "the reverse value must give the reverse list, not a sorted one");
  }

  /** Several provider options of one request, in the order the value lists them. */
  @Test
  void testSeveralProviderOptionsOfOneRequest() {
    final List<ExecutionProviderRequest> requests = resolved(EXECUTION_PROVIDERS_OPTION,
        "openvino(device_type=GPU.0;cache_dir=/var/cache/ov;num_streams=2),cpu");
    assertEquals(2, requests.size());
    assertEquals(List.of("device_type", "cache_dir", "num_streams"),
        List.copyOf(requests.get(0).options().keySet()));
    assertEquals("GPU.0", requests.get(0).options().get("device_type"));
    assertEquals("/var/cache/ov", requests.get(0).options().get("cache_dir"));
    assertEquals("2", requests.get(0).options().get("num_streams"));
    assertEquals(Map.of(), requests.get(1).options());
  }

  /**
   * The separator inside a request is {@code ;}, so a comma in a provider option value is an ordinary
   * character. This is the case the syntax exists for: OpenVINO's device types include
   * {@code MULTI:GPU,CPU}.
   */
  @Test
  void testACommaInsideAProviderOptionValueIsAValueCharacter() {
    final List<ExecutionProviderRequest> requests = resolved(EXECUTION_PROVIDERS_OPTION,
        "openvino(device_type=MULTI:GPU,CPU),cpu");
    assertEquals(2, requests.size(), "the comma inside the parentheses must not split the list");
    assertEquals("MULTI:GPU,CPU", requests.get(0).options().get("device_type"));
    assertEquals(ExecutionProviders.CPU, requests.get(1).id());
  }

  /** The first {@code =} separates a provider option, so a value may hold further ones. */
  @Test
  void testTheFirstEqualsSeparatesAProviderOption() {
    assertEquals("a=b=c", resolved(EXECUTION_PROVIDERS_OPTION, "cpu(k=a=b=c)")
        .get(0).options().get("k"));
  }

  /**
   * An empty provider option value reaches the configurer, which is the authority on its own options.
   * A blank name does not, because {@code ProviderSpec} rejects one everywhere in OpenNLP.
   */
  @Test
  void testAnEmptyProviderOptionValueIsCarriedAndABlankNameIsNot() {
    assertEquals("", resolved(EXECUTION_PROVIDERS_OPTION, "cuda(device_id=)")
        .get(0).options().get(DEVICE_ID));
    assertThrows(IllegalArgumentException.class,
        () -> resolved(EXECUTION_PROVIDERS_OPTION, "cuda( =1)"));
  }

  /**
   * Every malformed shape of the value is reported rather than repaired, and the failure names the
   * option and quotes what was read.
   *
   * @param value The malformed value.
   */
  @ParameterizedTest
  @ValueSource(strings = {"", ",", "cpu,", ",cpu", "cpu,,cuda", "cpu(", "cpu)", "cpu(a=1",
      "cpu(a=1))", "cpu(a=1)x", "cpu()", "cpu(a)", "cpu(=1)", "cpu(a=1;a=2)", "cpu(a=1;)",
      "cpu(a=1;;b=2)", "(a=1)", "cpu(a=1(b=2))", " cpu", "cpu, cpu", "cpu cuda", "not!an!id",
      "cpu,cpu(",
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"})
  void testMalformedValuesAreRejected(final String value) {
    final IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
        () -> provider.inferenceOptions(spec(EXECUTION_PROVIDERS_OPTION, value)),
        "'" + value + "' must not be accepted");
    assertTrue(failure.getMessage().contains(EXECUTION_PROVIDERS_OPTION),
        "the failure must name the option: " + failure.getMessage());
  }

  /**
   * Each malformed shape is reported as itself, not merely rejected.
   *
   * <p>This matters because several of them would be refused anyway further down: an empty list
   * element and a request that is only provider options both end up asking for an execution provider
   * whose id is the empty string, which {@code ExecutionProviderRequest} refuses on its own. A spec
   * author reading "id must be 1 to 64 characters" would have to work out that a stray comma caused
   * it, so the shape is named where it is found, and this is what keeps that naming in place.</p>
   *
   * @param value The malformed value.
   * @param fault Words the failure has to contain.
   */
  @ParameterizedTest
  @CsvSource(delimiter = '|', value = {
      "|must name at least one execution provider",
      ",cpu|has an empty element",
      "cpu,,cuda|has an empty element",
      "cpu,|ends with a ','",
      "cpu(a=1(b=2))|has a nested '('",
      "cpu)|has a ')' without a '('",
      "cpu(|has a '(' that is never closed",
      "cpu(a=1)x|has characters between a ')' and the next ','",
      "(a=1)|has provider options without an execution provider id",
      "cpu()|has an empty '()'",
      "cpu(a=1;)|ends its provider options with a ';'",
      "cpu(a)|has the provider option 'a' without a '='",
      "cpu(=1)|has a provider option with a blank name",
      "cpu(a=1;a=2)|names the provider option 'a' twice in one request"})
  void testEachMalformedShapeIsNamedInTheFailure(final String value, final String fault) {
    final IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
        () -> provider.inferenceOptions(spec(EXECUTION_PROVIDERS_OPTION,
            value == null ? "" : value)));
    assertTrue(failure.getMessage().contains(fault),
        "the failure for '" + value + "' must say it " + fault + ", and it says: "
            + failure.getMessage());
  }

  /** The same malformed value fails the public entry point, before ONNX Runtime is touched. */
  @Test
  void testAMalformedValueFailsCreate(@TempDir final Path dir) throws Exception {
    final Path model = Files.createFile(dir.resolve("model.onnx"));
    assertThrows(IllegalArgumentException.class, () -> provider.create(
        spec(model, EXECUTION_PROVIDERS_OPTION, "cpu(")));
    // The negative control: the same empty file with a well-formed value gets past every option
    // reader and fails in ONNX Runtime, so the case above failed on the syntax and not on the file.
    assertThrows(IOException.class, () -> provider.create(
        spec(model, EXECUTION_PROVIDERS_OPTION, "cpu")));
  }

  /** An id no configurer answers to fails at construction and names the id. */
  @Test
  void testAnUnknownIdFailsAndNamesItself(@TempDir final Path dir) throws Exception {
    final Path model = Files.createFile(dir.resolve("model.onnx"));
    final IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
        () -> provider.create(spec(model, EXECUTION_PROVIDERS_OPTION, "no-such-provider")));
    assertTrue(failure.getMessage().contains("no-such-provider"), failure.getMessage());
  }

  /**
   * The thread counts a spec names, at both ends of the accepted range.
   *
   * @param value The value to read.
   * @param expected The count it must produce.
   */
  @ParameterizedTest
  @CsvSource({"0, 0", "1, 1", "8, 8", "1024, 1024"})
  void testThreadCountsAreRead(final String value, final int expected) {
    assertEquals(expected,
        provider.inferenceOptions(spec(INTRA_OP_NUM_THREADS_OPTION, value)).getIntraOpNumThreads());
    assertEquals(expected,
        provider.inferenceOptions(spec(INTER_OP_NUM_THREADS_OPTION, value)).getInterOpNumThreads());
  }

  /** An unset thread count stays unset, which is how ONNX Runtime's own default is kept. */
  @Test
  void testUnsetThreadCountsStayUnset() {
    assertNull(provider.inferenceOptions(spec()).getIntraOpNumThreads());
    assertNull(provider.inferenceOptions(spec()).getInterOpNumThreads());
    assertNull(provider.inferenceOptions(spec()).getOptimizationLevel());
  }

  /**
   * A thread count ONNX Runtime would accept on the options object and act on much later is rejected
   * here instead.
   *
   * @param value The malformed value.
   */
  @ParameterizedTest
  @ValueSource(strings = {"-1", "1025", "two", "", " 1", "1 ", "2.0", "0x2",
      "99999999999999999999"})
  void testMalformedThreadCountsAreRejected(final String value) {
    assertThrows(IllegalArgumentException.class,
        () -> provider.inferenceOptions(spec(INTRA_OP_NUM_THREADS_OPTION, value)),
        "'" + value + "' must not be accepted as an intra-op count");
    assertThrows(IllegalArgumentException.class,
        () -> provider.inferenceOptions(spec(INTER_OP_NUM_THREADS_OPTION, value)),
        "'" + value + "' must not be accepted as an inter-op count");
  }

  /**
   * Every graph optimization level ONNX Runtime declares is nameable, under the lower case form of
   * its constant, which is how {@code padding} and {@code pooling} name theirs.
   *
   * @param level The level to name.
   */
  @ParameterizedTest
  @EnumSource(OrtSession.SessionOptions.OptLevel.class)
  void testEveryOptimizationLevelIsNameable(final OrtSession.SessionOptions.OptLevel level) {
    final String value = level.name().toLowerCase(Locale.ROOT);
    assertEquals(level,
        provider.inferenceOptions(spec(OPTIMIZATION_LEVEL_OPTION, value)).getOptimizationLevel());
  }

  /**
   * A level that is not one of them is rejected, the upper case constant name included, since no
   * other option of this provider is matched case-insensitively either.
   *
   * @param value The malformed value.
   */
  @ParameterizedTest
  @ValueSource(strings = {"ALL_OPT", "All_Opt", "all", "opt", "", "3", "all opt"})
  void testMalformedOptimizationLevelsAreRejected(final String value) {
    final IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
        () -> provider.inferenceOptions(spec(OPTIMIZATION_LEVEL_OPTION, value)),
        "'" + value + "' must not be accepted");
    assertTrue(failure.getMessage().contains(OPTIMIZATION_LEVEL_OPTION), failure.getMessage());
  }

  /**
   * The deprecated flag still means a CUDA request on the device the other deprecated option names,
   * which is what it has always meant.
   */
  @Test
  void testTheDeprecatedGpuOptionsStillWork() {
    assertEquals(List.of(ExecutionProviderRequest.of(ExecutionProviders.CUDA,
            Map.of(DEVICE_ID, "0"))), resolved(GPU_OPTION, "true"));
    assertEquals(List.of(ExecutionProviderRequest.of(ExecutionProviders.CUDA,
            Map.of(DEVICE_ID, "3"))), resolved(GPU_OPTION, "true", GPU_DEVICE_ID_OPTION, "3"));
    assertTrue(resolved(GPU_OPTION, "false", GPU_DEVICE_ID_OPTION, "3").isEmpty(),
        "the device id alone must not request anything");
  }

  /**
   * The list wins over the deprecated flag, which is the precedence
   * {@link ExecutionProviders#resolve(InferenceOptions)} documents: a list a spec author ordered is
   * not prepended with a request from a leftover flag.
   */
  @Test
  void testTheListWinsOverTheDeprecatedFlag() {
    assertEquals(List.of(ExecutionProviderRequest.of(ExecutionProviders.CPU)),
        resolved(EXECUTION_PROVIDERS_OPTION, "cpu", GPU_OPTION, "true",
            GPU_DEVICE_ID_OPTION, "3"));
    assertEquals(List.of(ExecutionProviderRequest.of(ExecutionProviders.CPU),
            ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "1"))),
        resolved(EXECUTION_PROVIDERS_OPTION, "cpu,cuda(device_id=1)", GPU_OPTION, "true",
            GPU_DEVICE_ID_OPTION, "3"),
        "the device id of the flag must not reach the CUDA request of the list");
  }

  /**
   * The deprecated options are still validated while the list wins, so a mistyped value is reported
   * rather than silently ignored because something else took precedence.
   */
  @Test
  void testTheDeprecatedOptionsAreStillValidatedBehindTheList() {
    assertThrows(IllegalArgumentException.class,
        () -> provider.inferenceOptions(spec(EXECUTION_PROVIDERS_OPTION, "cpu",
            GPU_OPTION, "yes")));
    assertThrows(IllegalArgumentException.class,
        () -> provider.inferenceOptions(spec(EXECUTION_PROVIDERS_OPTION, "cpu",
            GPU_DEVICE_ID_OPTION, "-1")));
  }

  /**
   * What a spec asks for reaches ONNX Runtime: the options a spec produces are handed to the same
   * configuration step a component uses, over session options that record the calls they receive.
   * The execution provider of the request is the test-only addon configurer, which writes a session
   * config entry per provider option instead of registering an execution provider that would have to
   * be installed.
   */
  @Test
  void testEverySettingReachesTheSessionConfiguration() throws Exception {
    final InferenceOptions options = provider.inferenceOptions(spec(
        EXECUTION_PROVIDERS_OPTION, EchoExecutionProviderConfigurer.ID + "(device_type=GPU.0)",
        INTRA_OP_NUM_THREADS_OPTION, "2",
        INTER_OP_NUM_THREADS_OPTION, "1",
        OPTIMIZATION_LEVEL_OPTION, "basic_opt"));
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      SessionOptionsProbe.configure(recorded, options);
      assertEquals(List.of(
          "addConfigEntry(" + EchoExecutionProviderConfigurer.PREFIX + "ran="
              + EchoExecutionProviderConfigurer.ID + ")",
          "addConfigEntry(" + EchoExecutionProviderConfigurer.PREFIX + "device_type=GPU.0)",
          "setIntraOpNumThreads(2)",
          "setInterOpNumThreads(1)",
          "setOptimizationLevel(BASIC_OPT)"), recorded.calls());
    }
  }

  /** A spec that names no session setting adds no session setting call at all. */
  @Test
  void testASpecWithoutSessionSettingsAddsNoCalls() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      SessionOptionsProbe.configure(recorded, provider.inferenceOptions(spec()));
      assertEquals(List.of(), recorded.calls(),
          "an unset setting must leave the ONNX Runtime default in place by not being set");
    }
  }

  /**
   * An addon execution provider id requested from a spec reaches a constructed embedder and decides
   * its batch plan.
   *
   * <p>The id is {@value PlacementEchoExecutionProviderConfigurer#ID}, which nothing in
   * {@code opennlp-dl} knows about and which arrives through
   * {@code META-INF/services/opennlp.dl.ExecutionProviderConfigurer} exactly as an addon's would. It
   * states the placement its {@code placement} provider option names, so the spec below carries a per
   * provider option through to the padding default of a running component: an accelerator pads a
   * mixed-length call to its longest row and runs one inference, and a CPU groups the call by exact
   * tokenized length and runs one inference per length. That chain, spec option to addon configurer
   * to batch plan, is what the deprecated {@code gpu} flag could not reach.</p>
   */
  @Test
  void testAnAddonIdFromASpecReachesAConstructedEmbedder(@TempDir final Path dir) throws Exception {
    vocab(dir);
    final Path model = model(dir).toPath();
    final TextEmbedder onAnAccelerator = provider.create(spec(model,
        EXECUTION_PROVIDERS_OPTION, probing(ExecutionProviderPlacement.ACCELERATOR)));
    final TextEmbedder onACpu = provider.create(spec(model,
        EXECUTION_PROVIDERS_OPTION, probing(ExecutionProviderPlacement.CPU)));
    try (SentenceVectorsDL accelerated = assertInstanceOf(SentenceVectorsDL.class, onAnAccelerator);
         SentenceVectorsDL cpu = assertInstanceOf(SentenceVectorsDL.class, onACpu)) {
      assertArrayEquals(LONGEST_SHAPES, accelerated.batchShapes(DISTINCT));
      assertArrayEquals(EXACT_SHAPES, cpu.batchShapes(DISTINCT));
      // And the vectors are the same either way, so the plan is the only difference.
      assertArrayEquals(cpu.embed("hello world"), accelerated.embed("hello world"), 1e-5f);
    }
  }

  /**
   * A padding strategy the spec names wins over the one derived from the addon execution provider, so
   * the derivation is a default and not a policy.
   */
  @Test
  void testAStatedPaddingStrategyWinsOverTheAddonDerivation(@TempDir final Path dir)
      throws Exception {
    vocab(dir);
    final Path model = model(dir).toPath();
    final TextEmbedder embedder = provider.create(spec(model, EXECUTION_PROVIDERS_OPTION,
        probing(ExecutionProviderPlacement.ACCELERATOR),
        PADDING_OPTION, "exact_length"));
    try (SentenceVectorsDL vectors = assertInstanceOf(SentenceVectorsDL.class, embedder)) {
      assertArrayEquals(EXACT_SHAPES, vectors.batchShapes(DISTINCT));
    }
  }

  /** The thread counts a spec names reach a real session, which is created and used here. */
  @Test
  void testASpecWithThreadCountsBuildsAWorkingEmbedder(@TempDir final Path dir) throws Exception {
    vocab(dir);
    final Path model = model(dir).toPath();
    final TextEmbedder embedder = provider.create(spec(model, INTRA_OP_NUM_THREADS_OPTION, "1",
        INTER_OP_NUM_THREADS_OPTION, "1", OPTIMIZATION_LEVEL_OPTION, "all_opt",
        PADDING_OPTION, "longest"));
    try (SentenceVectorsDL vectors = assertInstanceOf(SentenceVectorsDL.class, embedder)) {
      final float[] embedded = vectors.embed("hello world");
      assertNotNull(embedded);
      assertEquals(vectors.dimension(), embedded.length);
    }
  }
}
