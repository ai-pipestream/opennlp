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

package opennlp.examples;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Assertions;

import opennlp.tools.models.ClassPathModel;
import opennlp.tools.models.ClassPathModelEntry;
import opennlp.tools.models.ClassPathModelFinder;
import opennlp.tools.models.ClassPathModelLoader;
import opennlp.tools.models.simple.SimpleClassPathModelFinder;
import opennlp.tools.util.ParamChecks;

/**
 * Base class for the chapter example tests.
 * <p>
 * A sample printed in the manual names its input files the way a reader would: a plain relative
 * filename such as {@code opennlp-en-ud-ewt-pos-1.3-2.5.4.bin}. The staging methods below put a
 * real file at that name in the directory the tests run in, so the sample can run exactly as
 * printed instead of being rewritten around a test fixture. Staging calls belong outside the
 * {@code docs:begin} and {@code docs:end} markers; only what a reader copies belongs between them.
 */
public abstract class DocExampleSupport {

  /** Marks the line before the region the manual prints, followed by the listing id. */
  public static final String BEGIN_MARKER = "// docs:begin";

  /** Marks the line after the region the manual prints. */
  public static final String END_MARKER = "// docs:end";

  private static final Path WORK_DIR = Paths.get("").toAbsolutePath();

  /**
   * Places a published model at {@code fileName} in the working directory of the test.
   *
   * @param modelJarPattern the model artifact to read, as a jar name pattern, for example
   *                        {@code opennlp-models-pos-*.jar}. Must not be {@code null}.
   * @param fileName        the file name the manual prints. Must not be {@code null}.
   * @return the staged path.
   * @throws IOException if the model cannot be read or written.
   */
  protected static Path stageModel(String modelJarPattern, String fileName) throws IOException {
    ParamChecks.requireNonNullArg(modelJarPattern, "modelJarPattern");
    ParamChecks.requireNonNullArg(fileName, "fileName");
    final ClassPathModelFinder finder = new SimpleClassPathModelFinder(modelJarPattern);
    final Set<ClassPathModelEntry> entries = finder.findModels(false);
    Assertions.assertFalse(entries == null || entries.isEmpty(),
        "No model artifact on the classpath matching " + modelJarPattern
            + "; add it as a test dependency of opennlp-examples");
    final ClassPathModel model = new ClassPathModelLoader().load(entries.iterator().next());
    return write(fileName, model.model());
  }

  /**
   * Places a text fixture at {@code fileName} in the working directory of the test. Use this for
   * the training data and configuration files a sample reads.
   *
   * @param fileName the file name the manual prints. Must not be {@code null}.
   * @param lines    the content, one entry per line.
   * @return the staged path.
   * @throws IOException if the file cannot be written.
   */
  protected static Path stageFile(String fileName, String... lines) throws IOException {
    return stageFile(fileName, 1, lines);
  }

  /**
   * Places a text fixture at {@code fileName} that holds {@code lines} {@code repeat} times.
   * Default training parameters drop features seen fewer than five times, so training fixtures
   * repeat their samples.
   *
   * @param fileName the file name the manual prints. Must not be {@code null}.
   * @param repeat   how often the lines appear. Must not be negative.
   * @param lines    the content, one entry per line.
   * @return the staged path.
   * @throws IOException if the file cannot be written.
   */
  protected static Path stageFile(String fileName, int repeat, String... lines)
      throws IOException {
    ParamChecks.requireNonNullArg(fileName, "fileName");
    ParamChecks.requireNonNullArg(lines, "lines");
    ParamChecks.requireNonNegative(repeat, "repeat");
    final List<String> content = new ArrayList<>(repeat * lines.length);
    for (int i = 0; i < repeat; i++) {
      content.addAll(List.of(lines));
    }
    return write(fileName, String.join(System.lineSeparator(), content)
        .getBytes(StandardCharsets.UTF_8));
  }

  /**
   * Resolves a name against the directory the tests run in.
   *
   * @param fileName the file name. Must not be {@code null}.
   * @return the absolute path, which need not exist.
   */
  protected static Path workFile(String fileName) {
    ParamChecks.requireNonNullArg(fileName, "fileName");
    return WORK_DIR.resolve(fileName);
  }

  /*
   * Written through a temporary file so that two chapter classes staging the same model in
   * parallel forks never observe a half written file.
   */
  private static Path write(String fileName, byte[] content) throws IOException {
    final Path target = WORK_DIR.resolve(fileName);
    final Path temp = Files.createTempFile(WORK_DIR, fileName, ".staging");
    try {
      Files.write(temp, content);
      Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING);
    } finally {
      Files.deleteIfExists(temp);
    }
    return target;
  }
}
