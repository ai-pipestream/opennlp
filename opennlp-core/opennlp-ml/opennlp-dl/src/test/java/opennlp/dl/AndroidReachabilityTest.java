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

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.StandardLocation;
import javax.tools.ToolProvider;

import ai.onnxruntime.OrtSession;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.slf4j.Logger;

import opennlp.tools.commons.Internal;
import opennlp.tools.util.ext.ProviderSpec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Android reachability check: the ONNX Runtime facing sources of this module must keep compiling
 * under Java {@value #RELEASE}, so that a future Android addon stays possible.
 *
 * <h2>What this is and what it is not</h2>
 *
 * <p>It is not an Android build and it does not make this module run on Android. It is the mechanical
 * form of a property {@link OnnxInference} has so far kept by intention and comment only: the ORT
 * interaction stays on the API level Android's desugaring covers, so that the day someone builds an
 * Android addon, the one class that touches ONNX Runtime is not the obstacle. {@code OnnxInference}
 * was written that way on purpose, avoiding {@code HashMap.newHashMap(int)} (19),
 * {@code ByteBuffer.slice(int, int)} (13), {@code LongBuffer.put(int, long[])} (16), records and
 * pattern matching {@code instanceof}, and this is what keeps the next edit honest.</p>
 *
 * <p>What a real Android build would additionally need is recorded in the README of this module, in
 * the section "Android". In short: the module's language level, four sources in this package that use
 * records and newer switch forms, and an ONNX Runtime artifact for Android, which is a different
 * artifact from the one this module compiles against.</p>
 *
 * <h2>Why a compilation and not animal-sniffer</h2>
 *
 * <p>The usual tool for this is animal-sniffer with an Android signature set. It cannot run in this
 * build: neither {@code animal-sniffer-maven-plugin} nor any Android signature artifact resolves
 * offline, and adding a dependency that only an online build can fetch would make the check
 * disappear exactly where it is meant to run. Compiling with {@code --release} needs nothing but the
 * JDK that is already building the module, and a source that compiles under a release by definition
 * uses no API and no language feature newer than it.</p>
 *
 * <p>The trade against animal-sniffer is stated rather than hidden. {@code --release} catches every
 * JDK API and every language feature newer than {@value #RELEASE}, which a signature set for APIs
 * alone does not, and it misses the opposite case: an API that is in Java {@value #RELEASE} and not
 * on Android, such as {@code java.lang.ProcessHandle}. Android's desugaring covers the
 * {@code java.util}, {@code java.util.function} and {@code java.nio.file} surface these classes use,
 * so the level is the property worth pinning here; the exact Android API surface stays a matter for
 * the addon that is built, which can run animal-sniffer where artifacts resolve.</p>
 *
 * <h2>Why this cannot pass vacuously</h2>
 *
 * <p>This branch has already had tests that passed without asserting anything, and a check that
 * cannot fail is worse than none, so three things are asserted rather than assumed. The source of
 * every guarded class is found on disk and the test fails if one is missing, so a renamed or moved
 * file is a failure and not a silent skip. The compiler is required to be present rather than
 * assumed, again a failure and not a skip. And {@link #testTheCheckFailsOnANewerApi()} compiles a
 * source that calls a Java 19 method and requires this same mechanism to reject it, so the day
 * {@code --release} stops being honoured, that test goes red instead of every other one going
 * quietly green.</p>
 *
 * <p>A verification-only {@code maven-compiler-plugin} execution was the first candidate and does not
 * work: its {@code compile} goal sets the project artifact's file to its own output directory, which
 * in a reactor build makes every downstream module compile against the seven classes of the check
 * instead of the module. The in-process compiler has no such effect.</p>
 */
class AndroidReachabilityTest {

  /**
   * The Java release the guarded sources must compile under. It is the level Android's desugaring
   * covers and is unrelated to what this module targets, which is the project baseline.
   */
  private static final int RELEASE = 11;

  /**
   * The ONNX Runtime facing sources that must stay reachable, relative to the main source root.
   *
   * <p>{@link OnnxInference} is the one that matters: it is the single ORT interaction of this
   * package, so it is where a newer API would land. The execution provider SPI is here with it,
   * because an addon that registered an execution provider on Android would have to implement
   * {@link ExecutionProviderConfigurer} and build an {@link InferenceOptions}.</p>
   *
   * <p>{@code Tokens}, {@code AbstractDL}, {@code ExecutionProviderRequest} and
   * {@code ExecutionProviders} are deliberately not here: they hold records, switch rules, a pattern
   * {@code instanceof} and a {@code Stream.toList}, so they do not compile under
   * {@value #RELEASE} today. That is the gap the README section names, and leaving them out states it
   * rather than hiding it. {@link #testTheUnguardedSourcesAreTheDocumentedOnes()} keeps that list
   * honest.</p>
   */
  private static final List<String> GUARDED = List.of(
      "opennlp/dl/OnnxInference.java",
      "opennlp/dl/InferenceOptions.java",
      "opennlp/dl/ExecutionProvider.java",
      "opennlp/dl/ExecutionProviderConfigurer.java",
      "opennlp/dl/ExecutionProviderPlacement.java",
      "opennlp/dl/CpuExecutionProviderConfigurer.java",
      "opennlp/dl/CudaExecutionProviderConfigurer.java");

  /**
   * The sources of this package that are known not to compile under {@value #RELEASE}, each with the
   * newer construct that keeps it out.
   */
  private static final List<String> UNGUARDED = List.of(
      "opennlp/dl/Tokens.java",
      "opennlp/dl/AbstractDL.java",
      "opennlp/dl/ExecutionProviderRequest.java",
      "opennlp/dl/ExecutionProviders.java");

  /**
   * {@return the main source root of this module}
   *
   * <p>Surefire runs a test in the module directory, so the relative path is the normal answer. Where
   * a runner starts somewhere else, the compiled output of this module says where the module is. One
   * of the two has to work, and a source root that cannot be found fails the check rather than
   * letting it find nothing to compile.</p>
   */
  private static Path sourceRoot() {
    final Path relative = Path.of("src", "main", "java");
    if (Files.isDirectory(relative)) {
      return relative;
    }
    final File classes = codeSourceOf(OnnxInference.class);
    assertNotNull(classes, "the source root of this module cannot be located");
    final Path derived = classes.toPath().getParent().getParent().resolve(relative);
    assertTrue(Files.isDirectory(derived), "the source root of this module cannot be located; tried "
        + relative.toAbsolutePath() + " and " + derived);
    return derived;
  }

  /** {@return the guarded sources, one per invocation of the parameterized check} */
  private static List<String> guarded() {
    return GUARDED;
  }

  /**
   * {@return the compiler this build runs on, which must exist}
   *
   * <p>Required rather than assumed: a skip here would turn the whole check into a green no-op.</p>
   */
  private static JavaCompiler compiler() {
    final JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
    assertNotNull(compiler, "the build JDK must supply a compiler for this check to mean anything; "
        + "a JRE without javac cannot run it and must not pass it either");
    return compiler;
  }

  /**
   * {@return the file of {@code source} under the main source root, which must exist}
   *
   * @param source The path of the source, relative to the source root.
   */
  private static Path sourceFile(final String source) {
    final Path file = sourceRoot().resolve(source);
    assertTrue(Files.isRegularFile(file), "the guarded source " + file.toAbsolutePath()
        + " is not there; if it moved or was renamed, update this check rather than losing it");
    return file;
  }

  /**
   * The class path to compile against: what this test runs on, taken from the classes that the
   * guarded sources reference, plus the compiled output of this module for the types they refer to
   * that are not in the guarded set, such as the {@code Tokens} record.
   *
   * <p>Read from the code sources of loaded classes rather than from
   * {@code System.getProperty("java.class.path")}, which under Surefire is a single generated
   * manifest-only jar.</p>
   *
   * @return The class path entries. Never empty.
   */
  private static Collection<File> classPath() {
    final Set<File> entries = new LinkedHashSet<>();
    for (final Class<?> onIt : List.of(OnnxInference.class, OrtSession.class, Internal.class,
        ProviderSpec.class, Logger.class)) {
      final File entry = codeSourceOf(onIt);
      assertNotNull(entry, "the code source of " + onIt.getName() + " must be readable to build the "
          + "class path of this check");
      entries.add(entry);
    }
    return entries;
  }

  /**
   * {@return the jar or directory {@code type} was loaded from, or {@code null} if it has none}
   *
   * @param type The class to locate.
   */
  private static File codeSourceOf(final Class<?> type) {
    if (type.getProtectionDomain() == null || type.getProtectionDomain().getCodeSource() == null) {
      return null;
    }
    try {
      return new File(type.getProtectionDomain().getCodeSource().getLocation().toURI());
    } catch (final URISyntaxException e) {
      return null;
    }
  }

  /**
   * Compiles {@code files} with {@code --release} {@value #RELEASE} and collects what the compiler
   * said.
   *
   * @param files The sources to compile. Must not be empty.
   * @param output The directory the class files go into, which nothing reads afterwards.
   * @return The errors the compiler reported, rendered one per line, empty if it reported none.
   */
  private static List<String> compileAtRelease(final List<Path> files, final Path output) {
    final JavaCompiler compiler = compiler();
    final DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
    try (StandardJavaFileManager fileManager =
             compiler.getStandardFileManager(diagnostics, Locale.ROOT, StandardCharsets.UTF_8)) {
      fileManager.setLocation(StandardLocation.CLASS_PATH, classPath());
      fileManager.setLocation(StandardLocation.CLASS_OUTPUT, List.of(output.toFile()));
      final List<File> sources = new ArrayList<>();
      for (final Path file : files) {
        sources.add(file.toFile());
      }
      final boolean compiled = compiler.getTask(null, fileManager, diagnostics,
          Arrays.asList("--release", Integer.toString(RELEASE), "-proc:none", "-nowarn"), null,
          fileManager.getJavaFileObjectsFromFiles(sources)).call();
      final List<String> errors = new ArrayList<>();
      for (final Diagnostic<? extends JavaFileObject> diagnostic : diagnostics.getDiagnostics()) {
        if (diagnostic.getKind() == Diagnostic.Kind.ERROR) {
          errors.add(diagnostic.getSource() == null ? diagnostic.getMessage(Locale.ROOT)
              : diagnostic.getSource().getName() + ":" + diagnostic.getLineNumber() + ": "
                  + diagnostic.getMessage(Locale.ROOT));
        }
      }
      if (!compiled && errors.isEmpty()) {
        // A refusal the compiler did not attach a diagnostic to is still a refusal, and it must not
        // read as "no errors" to a caller that only looks at the list.
        errors.add("the compiler refused " + files + " without reporting a diagnostic");
      }
      return errors;
    } catch (final IOException e) {
      throw new UncheckedIOException("the check could not read the sources it guards", e);
    }
  }

  /**
   * Every guarded source compiles under Java {@value #RELEASE}, one source at a time so that the
   * failure names the file that broke the property.
   *
   * @param source The source to compile.
   * @param output A directory for the class files, which nothing reads.
   */
  @ParameterizedTest
  @MethodSource("guarded")
  void testTheGuardedSourcesCompileAtTheAndroidReleaseLevel(final String source,
      @TempDir final Path output) {
    final List<String> errors = compileAtRelease(List.of(sourceFile(source)), output);
    assertTrue(errors.isEmpty(), source + " no longer compiles under Java " + RELEASE + ", so it "
        + "uses an API or a language feature an Android build could not desugar. Either use the "
        + "older form, as OnnxInference does for HashMap.newHashMap and ByteBuffer.slice(int, int), "
        + "or move the source out of the guarded set in this test and say so in the README. What "
        + "the compiler reported: " + errors);
  }

  /** The whole guarded set also compiles together, so nothing in it depends on a newer form. */
  @Test
  void testTheGuardedSourcesCompileTogether(@TempDir final Path output) {
    final List<Path> files = new ArrayList<>();
    for (final String source : GUARDED) {
      files.add(sourceFile(source));
    }
    assertEquals(GUARDED.size(), files.size());
    assertTrue(compileAtRelease(files, output).isEmpty());
  }

  /**
   * The check catches a newer API, which is the only thing that makes the assertions above worth
   * anything. The source below calls {@code HashMap.newHashMap(int)}, which is Java 19 and is the
   * exact call {@link OnnxInference} avoids, and this mechanism has to reject it.
   *
   * @param output A directory for the class files, which nothing reads.
   */
  @Test
  void testTheCheckFailsOnANewerApi(@TempDir final Path output) throws Exception {
    final Path source = output.resolve("NewerApi.java");
    Files.writeString(source, "import java.util.HashMap;\n"
        + "import java.util.Map;\n"
        + "final class NewerApi {\n"
        + "  static Map<String, String> of() {\n"
        + "    return HashMap.newHashMap(3);\n"
        + "  }\n"
        + "}\n", StandardCharsets.UTF_8);
    final List<String> errors = compileAtRelease(List.of(source), output);
    assertFalse(errors.isEmpty(), "the check must reject HashMap.newHashMap(int), which is Java 19; "
        + "if it does not, --release is not being honoured and every other assertion here is empty");
    assertTrue(errors.toString().contains("newHashMap"), errors.toString());
  }

  /**
   * The check catches a newer language feature as well, which a signature-based tool would not. The
   * source below is a record, which is the construct that keeps {@code Tokens} out of the guarded
   * set.
   *
   * @param output A directory for the class files, which nothing reads.
   */
  @Test
  void testTheCheckFailsOnANewerLanguageFeature(@TempDir final Path output) throws Exception {
    final Path source = output.resolve("NewerFeature.java");
    Files.writeString(source, "record NewerFeature(int value) {\n}\n", StandardCharsets.UTF_8);
    assertFalse(compileAtRelease(List.of(source), output).isEmpty(),
        "the check must reject a record, which is Java 16");
  }

  /** A source that only uses Java 11 is accepted, so the check is not simply failing everything. */
  @Test
  void testTheCheckAcceptsAnOlderApi(@TempDir final Path output) throws Exception {
    final Path source = output.resolve("OlderApi.java");
    Files.writeString(source, "import java.util.HashMap;\n"
        + "import java.util.Map;\n"
        + "final class OlderApi {\n"
        + "  static Map<String, String> of() {\n"
        + "    return new HashMap<>(4);\n"
        + "  }\n"
        + "}\n", StandardCharsets.UTF_8);
    assertTrue(compileAtRelease(List.of(source), output).isEmpty(),
        "a Java 11 source must pass, or the mechanism rejects everything and proves nothing");
  }

  /**
   * The sources this check leaves out are the ones the README names, and each of them really does
   * fail under Java {@value #RELEASE}. So the guarded set cannot quietly shrink: moving a source out
   * of it means adding it here, which means it has to actually be unreachable.
   */
  @Test
  void testTheUnguardedSourcesAreTheDocumentedOnes(@TempDir final Path output) {
    for (final String source : UNGUARDED) {
      assertFalse(compileAtRelease(List.of(sourceFile(source)), output).isEmpty(),
          source + " compiles under Java " + RELEASE + " now, so it can join the guarded set and "
              + "leave the README section that names it as an obstacle");
    }
  }

  /** Every source of this package is either guarded or documented as unreachable. */
  @Test
  void testEverySourceOfThisPackageIsAccountedFor() throws Exception {
    final Set<String> accounted = new LinkedHashSet<>(GUARDED);
    accounted.addAll(UNGUARDED);
    final List<String> unaccounted = new ArrayList<>();
    try (var sources = Files.list(sourceRoot().resolve("opennlp").resolve("dl"))) {
      sources.filter(Files::isRegularFile)
          .map(file -> "opennlp/dl/" + file.getFileName())
          .filter(name -> name.endsWith(".java") && !accounted.contains(name))
          .forEach(unaccounted::add);
    }
    assertTrue(unaccounted.isEmpty(), "a new source in opennlp.dl has to be added either to the "
        + "guarded set of this check or to the list of sources the README names as unreachable: "
        + unaccounted);
  }
}
