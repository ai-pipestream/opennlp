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

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.slf4j.LoggerFactory;

import opennlp.dl.CpuExecutionProviderConfigurer;
import opennlp.dl.CudaExecutionProviderConfigurer;
import opennlp.dl.EchoExecutionProviderConfigurer;
import opennlp.dl.ExecutionProviderRequest;
import opennlp.dl.ExecutionProviders;
import opennlp.dl.InferenceOptions;
import opennlp.tools.util.ext.ProviderSpec;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The {@link PaddingStrategy} a caller gets when it states none: the rule
 * {@link PaddingStrategy#defaultFor(List)} applies to a resolved execution provider list, and the
 * {@link SentenceVectorsDL} constructor that applies that rule.
 *
 * <p>The rule cases need no session, since the derivation reads execution provider ids and no more
 * than that. The cases that run a session use {@code tiny-vectors.onnx} and read the batch plan
 * through
 * {@link SentenceVectorsDL#batchShapes(List)}, the way {@code SentenceVectorsDLPaddedBatchTest}
 * does, because the strategies differ in tensor shapes and inference count while returning the same
 * vectors.</p>
 *
 * <p>A session on an accelerator cannot be created here: {@code opennlp-dl} depends on the CPU-only
 * ONNX Runtime, so a CUDA request fails while the session options are built, before any padding
 * decision. The CUDA rule is therefore covered by the derivation cases on every build, and the one
 * case that runs a CUDA session is skipped unless CUDA works, which it does in
 * {@code opennlp-dl-gpu}.</p>
 */
class PaddingStrategyDefaultTest {

  /** Inputs of tokenized length 3, 4, 5, 3, 4, 3 with mean-pooled vectors that all differ. */
  private static final List<String> DISTINCT = List.of("x", "hello hello",
      "hello hello world", "hello", "hello world", "world");

  /** One inference per distinct tokenized length, with no padding, in first-seen order. */
  private static final int[][] EXACT_SHAPES = {{3, 3}, {2, 4}, {1, 5}};

  /** One inference for the entire call, at the longest row of it. */
  private static final int[][] LONGEST_SHAPES = {{6, 5}};

  /** One inference for the entire call, at the configured maximum. */
  private static final int[][] MAX_LENGTH_SHAPES = {{6, 8}};

  private static final int MAX_LENGTH = 8;

  private static final String DEVICE_ID = CudaExecutionProviderConfigurer.DEVICE_ID_OPTION;

  /** An execution provider id no configurer in {@code opennlp-dl} answers to. */
  private static final String ADDON_ID = "openvino";

  private static final String UNOBSERVABLE_LOGGING =
      "the SLF4J provider in this JVM does not write this logger to the file it was pointed at";

  /** The line the class writes through the logger of {@link SentenceVectorsDL} to test the setup. */
  private static final String SENTINEL = "opennlp-dl padding strategy logging probe";

  /** The file slf4j-simple is pointed at, so the log can be read back. */
  private static Path logFile;

  /**
   * Points {@code slf4j-simple} at a file and turns the logger of {@link SentenceVectorsDL} on, then
   * writes {@link #SENTINEL} through it.
   *
   * <p>A file rather than {@code System.err}, because a redirection of the console streams does not
   * survive here: the test runner installs its own streams around each test method, so a stream this
   * class put in place before them is no longer the one a logger writes to. A file is decided when
   * the SLF4J provider initializes and then stays put.</p>
   *
   * <p>Which provider wins and whether it reads these properties is a property of the class
   * path, and a provider that was already initialized by an earlier test class ignores them, so
   * {@link #loggingIsObservable()} checks for the sentinel rather than assuming any of it. The
   * file is left for the JVM to delete on exit: the provider keeps writing to it, and taking it
   * away while it is open would leave later entries in this JVM without a destination.</p>
   */
  @BeforeAll
  static void logToAFile() throws IOException {
    logFile = Files.createTempFile("opennlp-dl-padding-strategy-log", ".txt");
    logFile.toFile().deleteOnExit();
    System.setProperty("org.slf4j.simpleLogger.logFile", logFile.toAbsolutePath().toString());
    System.setProperty("org.slf4j.simpleLogger.log." + SentenceVectorsDL.class.getName(), "info");
    LoggerFactory.getLogger(SentenceVectorsDL.class).info(SENTINEL);
  }

  // Copied out of the classpath rather than resolved in place, because this test also runs from the
  // opennlp-dl test-jar in opennlp-dl-gpu, where the resource URI is not hierarchical.
  private static File model(final Path dir) throws IOException {
    final Path file = dir.resolve("tiny-vectors.onnx");
    try (InputStream is = Objects.requireNonNull(PaddingStrategyDefaultTest.class
        .getResourceAsStream("/opennlp/dl/vectors/tiny-vectors.onnx"))) {
      Files.copy(is, file, StandardCopyOption.REPLACE_EXISTING);
    }
    return file.toFile();
  }

  /** The vocabulary of the other vector tests, with {@code [PAD]} at id {@code 0}. */
  private static File vocab(final Path dir) throws IOException {
    final Path file = dir.resolve("vocab.txt");
    Files.write(file, List.of("[PAD]", "unused1", "[UNK]", "[SEP]", "hello", "world",
        "unused2", "[CLS]"));
    return file.toFile();
  }

  /** A component with the strategy derived from {@code inferenceOptions}. */
  private static SentenceVectorsDL derived(final Path dir,
      final InferenceOptions inferenceOptions) throws Exception {
    return SentenceVectorsDL.withDerivedPadding(model(dir), vocab(dir), true, Pooling.MEAN, false,
        MAX_LENGTH, inferenceOptions);
  }

  /** A component with the strategy the caller states. */
  private static SentenceVectorsDL stated(final Path dir, final PaddingStrategy padding,
      final InferenceOptions inferenceOptions) throws Exception {
    return new SentenceVectorsDL(model(dir), vocab(dir), true, Pooling.MEAN, false, MAX_LENGTH,
        padding, inferenceOptions);
  }

  private static InferenceOptions requesting(final ExecutionProviderRequest... requests) {
    final InferenceOptions options = new InferenceOptions();
    options.setExecutionProviders(List.of(requests));
    return options;
  }

  private static ExecutionProviderRequest cuda() {
    return ExecutionProviderRequest.of(ExecutionProviders.CUDA);
  }

  private static ExecutionProviderRequest cpu() {
    return ExecutionProviderRequest.of(ExecutionProviders.CPU);
  }

  private static PaddingStrategy derivedFor(final InferenceOptions options) {
    return PaddingStrategy.defaultFor(ExecutionProviders.resolve(options));
  }

  private static int[][] expected(final PaddingStrategy padding) {
    if (padding == PaddingStrategy.EXACT_LENGTH) {
      return EXACT_SHAPES;
    }
    return padding == PaddingStrategy.LONGEST ? LONGEST_SHAPES : MAX_LENGTH_SHAPES;
  }

  // The rule.

  /**
   * A CPU placement derives exact length grouping, the strategy padding runs at 0.66 to 0.81 of the
   * speed of on a CPU. The empty list is the CPU case too: ONNX Runtime puts such a session on the
   * CPU by itself.
   */
  @Test
  void testACpuPlacementDerivesExactLengthGrouping() {
    assertEquals(PaddingStrategy.EXACT_LENGTH, PaddingStrategy.defaultFor(List.of()));
    assertEquals(PaddingStrategy.EXACT_LENGTH, PaddingStrategy.defaultFor(List.of(cpu())));
    assertEquals(PaddingStrategy.EXACT_LENGTH,
        PaddingStrategy.defaultFor(List.of(ExecutionProviderRequest.of(ExecutionProviders.CPU,
            Map.of(CpuExecutionProviderConfigurer.USE_ARENA_OPTION, "false")))));
  }

  /** {@link SentenceVectorsDL#DEFAULT_PADDING} is the value the CPU case derives, as documented. */
  @Test
  void testTheDefaultPaddingConstantIsTheCpuCase() {
    assertEquals(SentenceVectorsDL.DEFAULT_PADDING, PaddingStrategy.defaultFor(List.of()));
    assertEquals(PaddingStrategy.EXACT_LENGTH, SentenceVectorsDL.DEFAULT_PADDING);
  }

  /**
   * An accelerator placement derives padding to the longest row of the batch, the strategy that
   * gives six to eight times the throughput of a GPU. This is the case the stage exists for.
   */
  @Test
  void testAnAcceleratorPlacementDerivesPaddingToTheLongestRow() {
    assertEquals(PaddingStrategy.LONGEST, PaddingStrategy.defaultFor(List.of(cuda())));
    assertEquals(PaddingStrategy.LONGEST, PaddingStrategy.defaultFor(List.of(
        ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "1")))));
    assertTrue(ExecutionProviders.runsOnAccelerator(List.of(cuda())));
  }

  /**
   * A mixed list follows its first request, since ONNX Runtime offers each node to the execution
   * providers in the order they were appended: {@code [cuda, cpu]} runs the encoder on the GPU,
   * while a CPU request in front of CUDA leaves CUDA with no node to run.
   */
  @Test
  void testAMixedListFollowsItsFirstRequest() {
    assertEquals(PaddingStrategy.LONGEST, PaddingStrategy.defaultFor(List.of(cuda(), cpu())));
    assertEquals(PaddingStrategy.EXACT_LENGTH, PaddingStrategy.defaultFor(List.of(cpu(), cuda())));
    assertTrue(ExecutionProviders.runsOnAccelerator(List.of(cuda(), cpu())));
    assertFalse(ExecutionProviders.runsOnAccelerator(List.of(cpu(), cuda())));
  }

  /**
   * An execution provider id this module cannot classify, which is what an addon contributes, is
   * treated as the CPU case. This is the documented decision: the conservative strategy needs no
   * padding token and no assumption about the graph, and it leaves the vectors of such a session
   * where they were.
   */
  @Test
  void testAnUnclassifiedIdIsTreatedAsTheCpuCase() {
    assertEquals(PaddingStrategy.EXACT_LENGTH,
        PaddingStrategy.defaultFor(List.of(ExecutionProviderRequest.of(ADDON_ID))));
    assertEquals(PaddingStrategy.EXACT_LENGTH, PaddingStrategy.defaultFor(
        List.of(ExecutionProviderRequest.of(EchoExecutionProviderConfigurer.ID))));
    // In front of CUDA it is still the request that counts, since it comes first.
    assertEquals(PaddingStrategy.EXACT_LENGTH,
        PaddingStrategy.defaultFor(List.of(ExecutionProviderRequest.of(ADDON_ID), cuda())));
  }

  /**
   * Ids are case-sensitive throughout this package, so a request that misspells the case of a
   * built-in id is not classified either. It would not resolve to a configurer at all.
   */
  @Test
  void testIdsAreClassifiedCaseSensitively() {
    assertEquals(PaddingStrategy.EXACT_LENGTH,
        PaddingStrategy.defaultFor(List.of(ExecutionProviderRequest.of("CUDA"))));
  }

  /**
   * {@link PaddingStrategy#MAX_LENGTH} is not derived at all. It is around twenty times behind the
   * other two on either placement and exists for a graph or a device that needs one fixed shape,
   * which is a property of the model rather than of the execution provider.
   */
  @Test
  void testTheFixedShapeStrategyIsNeverDerived() {
    for (final List<ExecutionProviderRequest> requests : List.of(List.<ExecutionProviderRequest>of(),
        List.of(cpu()), List.of(cuda()), List.of(cuda(), cpu()), List.of(cpu(), cuda()),
        List.of(ExecutionProviderRequest.of(ADDON_ID)))) {
      assertNotEquals(PaddingStrategy.MAX_LENGTH, PaddingStrategy.defaultFor(requests),
          "derived for " + ExecutionProviders.describe(requests));
    }
  }

  /**
   * The deprecated GPU flag represents one CUDA request, so it derives what a CUDA request derives.
   * A caller migrating off the flag keeps the strategy it had.
   */
  @Test
  @SuppressWarnings({"deprecation", "removal"})
  void testTheDeprecatedGpuFlagDerivesWhatACudaRequestDerives() {
    final InferenceOptions flagged = new InferenceOptions();
    flagged.setGpu(true);

    assertEquals(PaddingStrategy.LONGEST, derivedFor(flagged));
    assertEquals(derivedFor(requesting(cuda())), derivedFor(flagged));
  }

  /**
   * The flag is read only while the request list is empty, which
   * {@link ExecutionProviders#resolve(InferenceOptions)} works out, so a list of CPU behind a set
   * flag derives the CPU case. This keeps the derivation and the session placement in agreement:
   * they read the same resolved list.
   */
  @Test
  @SuppressWarnings({"deprecation", "removal"})
  void testARequestListWinsOverTheDeprecatedFlag() {
    final InferenceOptions flagged = requesting(cpu());
    flagged.setGpu(true);

    assertEquals(PaddingStrategy.EXACT_LENGTH, derivedFor(flagged));
  }

  /** A malformed argument is rejected where it is given, not passed on into a session. */
  @Test
  void testNullArgumentsAreRejected() {
    assertThrows(IllegalArgumentException.class, () -> PaddingStrategy.defaultFor(null));
    assertThrows(IllegalArgumentException.class,
        () -> PaddingStrategy.defaultFor(Collections.singletonList(null)));
    assertThrows(IllegalArgumentException.class, () -> ExecutionProviders.runsOnAccelerator(null));
    assertThrows(IllegalArgumentException.class,
        () -> ExecutionProviders.runsOnAccelerator(Collections.singletonList(null)));
  }

  // The constructor that applies the rule.

  /**
   * On the CPU the derived default groups by exact tokenized length and returns the vectors the
   * constructors without an {@link InferenceOptions} have always returned, to the last bit. This is
   * the case that must not move: a caller that upgrades gets the same numbers.
   */
  @Test
  void testTheDefaultOnTheCpuMatchesTheOlderConstructors(@TempDir final Path dir) throws Exception {
    try (SentenceVectorsDL asBefore = new SentenceVectorsDL(model(dir), vocab(dir), true,
             Pooling.MEAN, false, MAX_LENGTH);
         SentenceVectorsDL onTheCpu = derived(dir, new InferenceOptions());
         SentenceVectorsDL onAnExplicitCpu = derived(dir, requesting(cpu()))) {

      assertArrayEquals(EXACT_SHAPES, onTheCpu.batchShapes(DISTINCT));
      assertArrayEquals(EXACT_SHAPES, onAnExplicitCpu.batchShapes(DISTINCT));
      assertArrayEquals(EXACT_SHAPES, asBefore.batchShapes(DISTINCT));

      final float[][] before = asBefore.embedAll(DISTINCT);
      final float[][] after = onTheCpu.embedAll(DISTINCT);
      assertEquals(before.length, after.length);
      for (int i = 0; i < before.length; i++) {
        assertArrayEquals(before[i], after[i], 0f, "row " + i);
      }
    }
  }

  /**
   * An execution provider id from an addon derives the CPU case in a running component too, not only
   * in the rule. {@link EchoExecutionProviderConfigurer} is registered in this module's test
   * resources the way an addon registers one, and it adds a session config entry instead of an
   * execution provider, so the session is a working one.
   */
  @Test
  void testAnAddonProviderDerivesExactLengthGroupingInAComponent(@TempDir final Path dir)
      throws Exception {
    final InferenceOptions options =
        requesting(ExecutionProviderRequest.of(EchoExecutionProviderConfigurer.ID));
    try (SentenceVectorsDL vectors = derived(dir, options)) {
      assertArrayEquals(EXACT_SHAPES, vectors.batchShapes(DISTINCT));
    }
  }

  /**
   * A strategy the caller names is applied as named, whatever the execution providers are. Without
   * this, the derivation would override a choice that is documented to win, and two of the three
   * strategies would be unreachable on the execution provider that derives the third.
   */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testAStatedStrategyWinsOverTheDerivedOne(final PaddingStrategy padding,
      @TempDir final Path dir) throws Exception {
    final InferenceOptions addon =
        requesting(ExecutionProviderRequest.of(EchoExecutionProviderConfigurer.ID));
    try (SentenceVectorsDL onNoRequest = stated(dir, padding, new InferenceOptions());
         SentenceVectorsDL onTheCpu = stated(dir, padding, requesting(cpu()));
         SentenceVectorsDL onAnAddon = stated(dir, padding, addon)) {

      assertArrayEquals(expected(padding), onNoRequest.batchShapes(DISTINCT), "no request");
      assertArrayEquals(expected(padding), onTheCpu.batchShapes(DISTINCT), "an explicit cpu");
      assertArrayEquals(expected(padding), onAnAddon.batchShapes(DISTINCT), "an addon provider");
    }
  }

  /**
   * The SPI path derives as well. A spec without the
   * {@value OnnxTextEmbedderProvider#PADDING_OPTION} option leaves the strategy to the execution
   * provider, so an embedder configured by name on the CPU groups by exact tokenized length, and a
   * spec that names a strategy gets that one.
   */
  @Test
  void testTheProviderSpiDerivesTheSameDefault(@TempDir final Path dir) throws Exception {
    model(dir);
    vocab(dir);
    final Path graph = dir.resolve("tiny-vectors.onnx");
    final Map<String, String> common = Map.of(OnnxTextEmbedderProvider.VOCABULARY_OPTION,
        "vocab.txt", OnnxTextEmbedderProvider.MAX_LENGTH_OPTION, Integer.toString(MAX_LENGTH));
    final Map<String, String> stating = new LinkedHashMap<>(common);
    stating.put(OnnxTextEmbedderProvider.PADDING_OPTION, "longest");
    final OnnxTextEmbedderProvider provider = new OnnxTextEmbedderProvider();
    try (SentenceVectorsDL fromSpi =
             (SentenceVectorsDL) provider.create(ProviderSpec.of(graph, common));
         SentenceVectorsDL stated =
             (SentenceVectorsDL) provider.create(ProviderSpec.of(graph, stating))) {

      assertArrayEquals(EXACT_SHAPES, fromSpi.batchShapes(DISTINCT), "no padding option");
      assertArrayEquals(LONGEST_SHAPES, stated.batchShapes(DISTINCT), "padding=longest");
    }
  }

  /**
   * A CUDA session derives padding to the longest row, and a stated strategy still wins there. This
   * is the wired form of the rule, and it needs a working CUDA runtime, which {@code opennlp-dl}
   * does not depend on; it runs in {@code opennlp-dl-gpu} and on a machine with the GPU runtime on
   * the classpath.
   */
  @Test
  void testACudaSessionDerivesPaddingToTheLongestRow(@TempDir final Path dir) throws Exception {
    Assumptions.assumeTrue(cudaUsable(), "no usable CUDA execution provider here");
    try (SentenceVectorsDL onCuda = derived(dir, requesting(cuda()));
         SentenceVectorsDL stated = stated(dir, PaddingStrategy.EXACT_LENGTH,
             requesting(cuda()))) {

      assertArrayEquals(LONGEST_SHAPES, onCuda.batchShapes(DISTINCT));
      assertArrayEquals(EXACT_SHAPES, stated.batchShapes(DISTINCT));
    }
  }

  // The log entry.

  /**
   * The derived strategy is written to the log at {@code info}, next to the execution providers it
   * came from. The stage that added the derivation added this line with it: a default that cannot
   * be read off a running system is how the wrong one stays in place for years.
   *
   * <p>{@link #logToAFile()} arranges for the entry to be readable, and this is skipped where that
   * did not take, which is a test JVM where an earlier class already initialized the SLF4J provider.
   * A run of this class on its own reads the entry:</p>
   *
   * <pre>{@code ./mvnw test -pl opennlp-core/opennlp-ml/opennlp-dl -am \
   *     -Dtest=PaddingStrategyDefaultTest}</pre>
   */
  @Test
  void testTheDerivedStrategyIsLogged(@TempDir final Path dir) throws Exception {
    Assumptions.assumeTrue(loggingIsObservable(), UNOBSERVABLE_LOGGING);

    final String logged = logged(() -> {
      try (SentenceVectorsDL vectors = derived(dir, requesting(cpu()))) {
        assertArrayEquals(EXACT_SHAPES, vectors.batchShapes(DISTINCT));
      } catch (final Exception e) {
        throw new IllegalStateException(e);
      }
    });

    assertTrue(logged.contains(PaddingStrategy.EXACT_LENGTH.name()), logged);
    assertTrue(logged.contains("derived"), logged);
    assertTrue(logged.contains("cpu[]"), logged);
  }

  /** A stated strategy is logged as well, and is named as the caller's rather than as derived. */
  @Test
  void testAStatedStrategyIsLoggedAsStated(@TempDir final Path dir) throws Exception {
    Assumptions.assumeTrue(loggingIsObservable(), UNOBSERVABLE_LOGGING);

    final String logged = logged(() -> {
      try (SentenceVectorsDL vectors = stated(dir, PaddingStrategy.LONGEST,
               new InferenceOptions())) {
        assertArrayEquals(LONGEST_SHAPES, vectors.batchShapes(DISTINCT));
      } catch (final Exception e) {
        throw new IllegalStateException(e);
      }
    });

    assertTrue(logged.contains(PaddingStrategy.LONGEST.name()), logged);
    assertTrue(logged.contains("caller"), logged);
  }

  /** {@return whether a CUDA session can actually be created on device {@code 0} here} */
  private static boolean cudaUsable() {
    if (!OrtEnvironment.getAvailableProviders().contains(OrtProvider.CUDA)) {
      return false;
    }
    OrtEnvironment.getEnvironment();
    try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
      options.addCUDA(0);
      return true;
    } catch (final OrtException | RuntimeException e) {
      return false;
    }
  }

  /**
   * {@return whether the log file {@link #logToAFile()} set up has the sentinel in it, which is what
   * shows that an {@code info} entry from the logger of {@link SentenceVectorsDL} can be read back
   * here}
   */
  private static boolean loggingIsObservable() throws IOException {
    return Files.readString(logFile, StandardCharsets.UTF_8).contains(SENTINEL);
  }

  /**
   * Runs {@code body} and returns what was logged while it ran.
   *
   * @param body The work to run.
   * @return The entries added to the log file, which is what {@code body} logged.
   */
  private static String logged(final Runnable body) throws IOException {
    final int from = Files.readString(logFile, StandardCharsets.UTF_8).length();
    body.run();
    return Files.readString(logFile, StandardCharsets.UTF_8).substring(from);
  }
}
