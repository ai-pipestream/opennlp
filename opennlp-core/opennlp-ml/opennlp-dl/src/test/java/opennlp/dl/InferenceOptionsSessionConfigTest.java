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

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import ai.onnxruntime.OrtSession;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The session configuration {@link InferenceOptions} carries: the thread counts, the graph
 * optimization level and the execution provider list, what each one accepts, and what each one does
 * to the session options a component builds.
 *
 * <p>onnxruntime 1.29.0 has no getter for any of the three settings and does not put them in
 * {@code getConfigEntries()} either, so what reaches the session is observed through
 * {@link RecordingSessionOptions}, which records the calls instead of making them. A case that
 * asserts a call was made fails if the call is taken out of
 * {@code AbstractDL.configureSession}, and a case that asserts no call was made fails if a default
 * is invented for a setting the caller left unset.</p>
 */
class InferenceOptionsSessionConfigTest {

  private static final String DEVICE_ID = CudaExecutionProviderConfigurer.DEVICE_ID_OPTION;

  private static final String UNOBSERVABLE_LOGGING =
      "the SLF4J provider here writes neither to System.out nor to System.err";

  // Defaults.

  /** Nothing is set on a fresh instance, which is what leaves every ONNX Runtime default alone. */
  @Test
  void testEverythingIsUnsetByDefault() {
    final InferenceOptions options = new InferenceOptions();

    assertNull(options.getIntraOpNumThreads());
    assertNull(options.getInterOpNumThreads());
    assertNull(options.getOptimizationLevel());
    assertEquals(List.of(), options.getExecutionProviders());
  }

  /**
   * A default {@link InferenceOptions} makes no call on the session options at all, so the session a
   * component builds is the plain default one the code built before any of this was configurable.
   */
  @Test
  void testDefaultOptionsConfigureNothing() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      SessionOptionsProbe.configure(recorded, new InferenceOptions());

      assertEquals(List.of(), recorded.calls());
    }
  }

  /**
   * And the session options a default {@link InferenceOptions} produces carry the same configuration
   * as a plain {@code OrtSession.SessionOptions}, which is the part of them ONNX Runtime will read
   * back.
   */
  @Test
  void testDefaultOptionsMatchPlainSessionOptions() throws Exception {
    try (OrtSession.SessionOptions plain = new OrtSession.SessionOptions();
         OrtSession.SessionOptions built = SessionOptionsProbe.of(new InferenceOptions())) {
      assertEquals(plain.getConfigEntries(), built.getConfigEntries());
      assertEquals(Map.of(), built.getConfigEntries());
    }
  }

  // Thread counts: what is accepted.

  @ParameterizedTest
  @ValueSource(ints = {0, 1, 2, 8, 31, 32, 1023, InferenceOptions.MAX_NUM_THREADS})
  void testAcceptedThreadCounts(final int numThreads) {
    final InferenceOptions options = new InferenceOptions();

    options.setIntraOpNumThreads(numThreads);
    options.setInterOpNumThreads(numThreads);

    assertEquals(numThreads, options.getIntraOpNumThreads());
    assertEquals(numThreads, options.getInterOpNumThreads());
  }

  /**
   * A negative count is rejected here because ONNX Runtime accepts it on the options object without
   * complaint and only acts on it when the session is created, which was verified against
   * onnxruntime 1.29.0.
   *
   * @param numThreads The rejected count.
   */
  @ParameterizedTest
  @ValueSource(ints = {-1, -32, Integer.MIN_VALUE, InferenceOptions.MAX_NUM_THREADS + 1,
      100_000, Integer.MAX_VALUE})
  void testRejectedThreadCounts(final int numThreads) {
    final InferenceOptions options = new InferenceOptions();

    assertThrows(IllegalArgumentException.class, () -> options.setIntraOpNumThreads(numThreads));
    assertThrows(IllegalArgumentException.class, () -> options.setInterOpNumThreads(numThreads));
    assertNull(options.getIntraOpNumThreads(), "a rejected value must not be stored");
    assertNull(options.getInterOpNumThreads(), "a rejected value must not be stored");
  }

  @Test
  void testRejectedThreadCountNamesTheSetting() {
    final InferenceOptions options = new InferenceOptions();

    assertTrue(assertThrows(IllegalArgumentException.class,
        () -> options.setIntraOpNumThreads(-1)).getMessage().contains("intraOpNumThreads"));
    assertTrue(assertThrows(IllegalArgumentException.class,
        () -> options.setInterOpNumThreads(-1)).getMessage().contains("interOpNumThreads"));
  }

  // Thread counts and optimization level: what reaches the session.

  /** Every setting that is set reaches the session, with the value that was set. */
  @Test
  void testEverySettingReachesTheSession() throws Exception {
    final InferenceOptions options = new InferenceOptions();
    options.setIntraOpNumThreads(4);
    options.setInterOpNumThreads(2);
    options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);

    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      SessionOptionsProbe.configure(recorded, options);

      assertEquals(List.of("setIntraOpNumThreads(4)", "setInterOpNumThreads(2)",
          "setOptimizationLevel(BASIC_OPT)"), recorded.calls());
    }
  }

  /** A setting that was left unset makes no call, so ONNX Runtime's own default stands. */
  @Test
  void testOnlyTheSettingsThatWereSetReachTheSession() throws Exception {
    final InferenceOptions options = new InferenceOptions();
    options.setIntraOpNumThreads(1);

    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      SessionOptionsProbe.configure(recorded, options);

      assertEquals(List.of("setIntraOpNumThreads(1)"), recorded.calls());
    }
  }

  /** Zero is a value, not an absence: it reaches the session and asks ONNX Runtime to choose. */
  @Test
  void testZeroThreadsReachesTheSession() throws Exception {
    final InferenceOptions options = new InferenceOptions();
    options.setIntraOpNumThreads(0);

    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      SessionOptionsProbe.configure(recorded, options);

      assertEquals(List.of("setIntraOpNumThreads(0)"), recorded.calls());
    }
  }

  /**
   * Every optimization level reaches the session as itself, including the one that takes every
   * optimization away.
   *
   * @param level The level under test.
   */
  @ParameterizedTest
  @EnumSource(OrtSession.SessionOptions.OptLevel.class)
  void testEveryOptimizationLevelReachesTheSession(
      final OrtSession.SessionOptions.OptLevel level) throws Exception {
    final InferenceOptions options = new InferenceOptions();
    options.setOptimizationLevel(level);

    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      SessionOptionsProbe.configure(recorded, options);

      assertEquals(List.of("setOptimizationLevel(" + level + ")"), recorded.calls());
    }
  }

  @Test
  void testNullOptimizationLevelRejected() {
    final InferenceOptions options = new InferenceOptions();

    assertThrows(IllegalArgumentException.class, () -> options.setOptimizationLevel(null));
    assertNull(options.getOptimizationLevel());
  }

  /**
   * The execution providers are appended before the session settings are applied, since ONNX Runtime
   * reads the provider order as the priority order and nothing else may be interleaved with it.
   */
  @Test
  void testExecutionProvidersAreAppendedBeforeTheSessionSettings() throws Exception {
    final InferenceOptions options = new InferenceOptions();
    options.setExecutionProviders(List.of(
        ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "1")),
        ExecutionProviderRequest.of(ExecutionProviders.CPU)));
    options.setIntraOpNumThreads(3);

    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      SessionOptionsProbe.configure(recorded, options);

      assertEquals(List.of("addCUDA(1)", "addCPU(true)", "setIntraOpNumThreads(3)"),
          recorded.calls());
    }
  }

  /**
   * The deprecated GPU flag reaches the session as exactly the calls an explicit CUDA request
   * reaches it as, so the old path is not merely similar to the new one, it is the same
   * configuration.
   */
  @Test
  @SuppressWarnings({"deprecation", "removal"})
  void testTheDeprecatedFlagConfiguresTheSessionLikeAnExplicitCudaRequest() throws Exception {
    final InferenceOptions deprecated = new InferenceOptions();
    deprecated.setGpu(true);
    deprecated.setGpuDeviceId(2);
    final InferenceOptions explicit = new InferenceOptions();
    explicit.setExecutionProviders(List.of(
        ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "2"))));

    try (RecordingSessionOptions fromFlag = new RecordingSessionOptions();
         RecordingSessionOptions fromList = new RecordingSessionOptions()) {
      SessionOptionsProbe.configure(fromFlag, deprecated);
      SessionOptionsProbe.configure(fromList, explicit);

      assertEquals(List.of("addCUDA(2)"), fromFlag.calls());
      assertEquals(fromList.calls(), fromFlag.calls());
    }
  }

  @Test
  void testConfigureRejectsNull() throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      assertThrows(IllegalArgumentException.class,
          () -> SessionOptionsProbe.configure(null, new InferenceOptions()));
      assertThrows(IllegalArgumentException.class,
          () -> SessionOptionsProbe.configure(recorded, null));
    }
  }

  // The execution provider list on InferenceOptions.

  @Test
  void testExecutionProvidersAreCopiedAndUnmodifiable() {
    final List<ExecutionProviderRequest> caller = new ArrayList<>();
    caller.add(ExecutionProviderRequest.of(ExecutionProviders.CPU));
    final InferenceOptions options = new InferenceOptions();
    options.setExecutionProviders(caller);

    caller.add(ExecutionProviderRequest.of(ExecutionProviders.CUDA));

    assertEquals(List.of(ExecutionProviderRequest.of(ExecutionProviders.CPU)),
        options.getExecutionProviders());
    assertThrows(UnsupportedOperationException.class, () -> options.getExecutionProviders()
        .add(ExecutionProviderRequest.of(ExecutionProviders.CUDA)));
  }

  @Test
  void testAddExecutionProviderAppendsInCallOrder() {
    final InferenceOptions options = new InferenceOptions();
    options.addExecutionProvider(ExecutionProviderRequest.of(ExecutionProviders.CUDA));
    options.addExecutionProvider(ExecutionProviderRequest.of(EchoExecutionProviderConfigurer.ID));
    options.addExecutionProvider(ExecutionProviderRequest.of(ExecutionProviders.CPU));

    assertEquals(List.of(ExecutionProviderRequest.of(ExecutionProviders.CUDA),
        ExecutionProviderRequest.of(EchoExecutionProviderConfigurer.ID),
        ExecutionProviderRequest.of(ExecutionProviders.CPU)), options.getExecutionProviders());
  }

  @Test
  void testExecutionProviderListRejectsNullAndNullElements() {
    final InferenceOptions options = new InferenceOptions();
    final List<ExecutionProviderRequest> withNull = new ArrayList<>();
    withNull.add(null);

    assertThrows(IllegalArgumentException.class, () -> options.setExecutionProviders(null));
    assertThrows(IllegalArgumentException.class, () -> options.setExecutionProviders(withNull));
    assertThrows(IllegalArgumentException.class, () -> options.addExecutionProvider(null));
    assertEquals(List.of(), options.getExecutionProviders());
  }

  @Test
  void testAnEmptyListIsAccepted() {
    final InferenceOptions options = new InferenceOptions();
    options.addExecutionProvider(ExecutionProviderRequest.of(ExecutionProviders.CPU));

    assertDoesNotThrow(() -> options.setExecutionProviders(List.of()));
    assertEquals(List.of(), options.getExecutionProviders());
  }

  // The log entry.

  /**
   * One entry is written on the ordinary path, naming the resolved execution providers and the
   * effective session settings. It exists because nothing in this package used to report which
   * execution provider a session ran on, which is how a component that could only ever reach the CPU
   * advertised GPU support for two years.
   *
   * <p>Skipped unless the logger is on and writes to a console stream; see
   * {@link #loggingIsObservable()} for the command that runs it.</p>
   */
  @Test
  void testTheConfigurationIsLogged() throws Exception {
    Assumptions.assumeTrue(loggingIsObservable(), UNOBSERVABLE_LOGGING);
    final InferenceOptions options = new InferenceOptions();
    options.setExecutionProviders(List.of(
        ExecutionProviderRequest.of(ExecutionProviders.CUDA, Map.of(DEVICE_ID, "1")),
        ExecutionProviderRequest.of(ExecutionProviders.CPU)));
    options.setIntraOpNumThreads(4);

    final String logged = captureConsole(options);

    assertTrue(logged.contains("cuda[device_id]"), logged);
    assertTrue(logged.contains("cpu[]"), logged);
    assertTrue(logged.contains("4"), logged);
    assertTrue(logged.contains("ONNX Runtime default"),
        "a setting that was left unset must be named as ONNX Runtime's: " + logged);
  }

  /** The entry is written for a default configuration too, not only where something was chosen. */
  @Test
  void testTheDefaultConfigurationIsLoggedToo() throws Exception {
    Assumptions.assumeTrue(loggingIsObservable(), UNOBSERVABLE_LOGGING);

    final String logged = captureConsole(new InferenceOptions());

    assertTrue(logged.contains("CPU"), logged);
    assertTrue(logged.contains("ONNX Runtime default"), logged);
  }

  /**
   * {@return whether an {@code info} entry from the logger {@link AbstractDL} uses reaches one of
   * the console streams here}
   *
   * <p>The build runs the tests with {@code -Dorg.slf4j.simpleLogger.defaultLogLevel=off}, so the
   * two cases above are skipped in an ordinary run. They are meant to be run on demand, which turns
   * this one logger back on and leaves the rest of the suite quiet:</p>
   *
   * <pre>{@code ./mvnw test -pl opennlp-core/opennlp-ml/opennlp-dl \
   *     -Dtest=InferenceOptionsSessionConfigTest \
   *     -DargLine="-Dorg.slf4j.simpleLogger.log.opennlp.dl.AbstractDL=info"}</pre>
   *
   * <p>Which SLF4J provider wins is a property of the class path, not of this module: both
   * {@code slf4j-simple}, which writes to {@code System.err}, and {@code logback-classic}, which
   * writes to {@code System.out} until it is configured, are reachable from the test class path, and
   * a provider could write somewhere else entirely. So the check is not a level check alone: a line
   * this test emits itself, through the same logger name, has to come back. The two cases are
   * skipped where it does not, and they fail where it does and the configuration writes no entry,
   * which is the behavior they are there for.</p>
   */
  private static boolean loggingIsObservable() {
    final String sentinel = "opennlp-dl session configuration logging probe";
    final Logger logger = LoggerFactory.getLogger(AbstractDL.class);
    if (!logger.isInfoEnabled()) {
      return false;
    }
    return capturingConsole(() -> logger.info(sentinel)).contains(sentinel);
  }

  /**
   * Configures a session with the given options while the console streams are captured, and returns
   * what was written to them.
   *
   * @param options The options to apply.
   * @return The captured output.
   * @throws Exception Thrown if the configuration fails.
   */
  private static String captureConsole(final InferenceOptions options) throws Exception {
    try (RecordingSessionOptions recorded = new RecordingSessionOptions()) {
      final AtomicReference<Exception> failure = new AtomicReference<>();
      final String captured = capturingConsole(() -> {
        try {
          SessionOptionsProbe.configure(recorded, options);
        } catch (final Exception e) {
          failure.set(e);
        }
      });
      if (failure.get() != null) {
        throw failure.get();
      }
      return captured;
    }
  }

  /**
   * Runs {@code body} with {@code System.out} and {@code System.err} redirected into one buffer.
   *
   * @param body The work to run.
   * @return Everything the work wrote to either stream.
   */
  private static String capturingConsole(final Runnable body) {
    final ByteArrayOutputStream captured = new ByteArrayOutputStream();
    final PrintStream originalOut = System.out;
    final PrintStream originalErr = System.err;
    try (PrintStream redirected = new PrintStream(captured, true, StandardCharsets.UTF_8)) {
      System.setOut(redirected);
      System.setErr(redirected);
      try {
        body.run();
      } finally {
        System.setOut(originalOut);
        System.setErr(originalErr);
      }
    }
    return captured.toString(StandardCharsets.UTF_8);
  }
}
