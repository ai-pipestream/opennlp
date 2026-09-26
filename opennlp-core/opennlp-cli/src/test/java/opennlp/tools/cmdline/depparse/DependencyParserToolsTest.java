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

package opennlp.tools.cmdline.depparse;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.slf4j.LoggerFactory;

import opennlp.tools.cmdline.CLI;
import opennlp.tools.cmdline.StreamFactoryRegistry;
import opennlp.tools.cmdline.TerminateToolException;
import opennlp.tools.depparse.DependencyGraph;
import opennlp.tools.depparse.DependencyModel;
import opennlp.tools.depparse.DependencyParserME;
import opennlp.tools.depparse.DependencySample;
import opennlp.tools.util.ObjectStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Exercises CLI registration, CoNLL-U input, training, persistence and inference. */
public class DependencyParserToolsTest {

  /** A two-token sentence in CoNLL-U; the tests repeat it to build a corpus. */
  private static final String SENTENCE = """
      1\tdogs\tdog\tNOUN\tNNS\t_\t2\tnsubj\t_\t_
      2\trun\trun\tVERB\tVBP\t_\t0\troot\t_\t_

      """;

  /** How often {@link #SENTENCE} is repeated in the training data. */
  private static final int REPETITIONS = 20;

  /** The gold graph of {@link #SENTENCE}. */
  private static final DependencyGraph GOLD =
      DependencyGraph.of(new int[] {1, -1}, new String[] {"nsubj", "root"});

  /** The sentence with the heads swapped, which a parser trained on {@link #SENTENCE} misparses. */
  private static final String REVERSED_SENTENCE = """
      1\tdogs\tdog\tNOUN\tNNS\t_\t0\troot\t_\t_
      2\trun\trun\tVERB\tVBP\t_\t1\tobj\t_\t_

      """;

  /** The gold graph of {@link #REVERSED_SENTENCE}. */
  private static final DependencyGraph REVERSED =
      DependencyGraph.of(new int[] {-1, 0}, new String[] {"root", "obj"});

  /** The expected block the error listener prints for {@link #REVERSED_SENTENCE}. */
  private static final String EXPECTED_REVERSED = "Expected: {\n"
      + "1\tdogs\tNOUN\t0\troot" + System.lineSeparator()
      + "2\trun\tVERB\t1\tobj" + System.lineSeparator() + "}";

  /** The predicted block the error listener prints for the parse of {@link #SENTENCE}. */
  private static final String PREDICTED_SENTENCE = "Predicted: {\n"
      + "1\tdogs\tNOUN\t2\tnsubj" + System.lineSeparator()
      + "2\trun\tVERB\t0\troot" + System.lineSeparator() + "}";

  /** The scores logged for a parser that reproduces all 40 tokens of the data. */
  private static final String PERFECT_SCORES = "Tokens: 40; UAS: 1.0; LAS: 1.0";

  /** The punctuation-free scores logged for the same parser; the data has no punctuation. */
  private static final String PERFECT_SCORES_EXCLUDING_PUNCTUATION =
      "Tokens excluding punctuation: 40; UAS: 1.0; LAS: 1.0";

  @TempDir
  private Path dir;

  private Path data;
  private Path model;

  @BeforeEach
  void writeData() throws IOException {
    data = dir.resolve("train.conllu");
    model = dir.resolve("parser.bin");
    Files.writeString(data, SENTENCE.repeat(REPETITIONS));
  }

  /** Trains a model into {@link #model} through the trainer tool. */
  private void train() {
    new DependencyParserTrainerTool().run("conllu", new String[] {
        "-lang", "eng", "-data", data.toString(), "-model", model.toString()});
  }

  /**
   * Runs a tool while capturing the INFO messages its logger emits.
   *
   * @param tool The tool class whose logger is captured.
   * @param run Runs the tool.
   * @return The formatted log messages, in order. Never {@code null}.
   */
  private static List<String> logOf(Class<?> tool, Runnable run) {
    final Logger logger = (Logger) LoggerFactory.getLogger(tool);
    final Level previous = logger.getLevel();
    final ListAppender<ILoggingEvent> appender = new ListAppender<>();
    appender.start();
    logger.addAppender(appender);
    logger.setLevel(Level.INFO);
    try {
      run.run();
    } finally {
      logger.setLevel(previous);
      logger.detachAppender(appender);
    }
    return appender.list.stream().map(ILoggingEvent::getFormattedMessage).toList();
  }

  @Test
  void testToolsAreRegistered() {
    assertTrue(CLI.getToolNames().contains("DependencyParserME"));
    assertTrue(CLI.getToolNames().contains("DependencyParserTrainerME"));
    assertTrue(CLI.getToolNames().contains("DependencyParserEvaluator"));
    assertTrue(CLI.getToolNames().contains("DependencyParserCrossValidator"));
  }

  @Test
  void testTrainerWritesALoadableModel() throws IOException {
    train();
    assertEquals(GOLD, new DependencyParserME(new DependencyModel(model))
        .parse(new String[] {"dogs", "run"}, new String[] {"NOUN", "VERB"}));
  }

  @Test
  void testEvaluatorLogsAttachmentScores() {
    train();
    final List<String> log = logOf(DependencyParserEvaluatorTool.class,
        () -> new DependencyParserEvaluatorTool().run("conllu", new String[] {
            "-model", model.toString(), "-data", data.toString()}));
    assertEquals(List.of(PERFECT_SCORES, PERFECT_SCORES_EXCLUDING_PUNCTUATION), log);
  }

  @Test
  void testCrossValidatorLogsAttachmentScores() {
    final List<String> log = logOf(DependencyParserCrossValidatorTool.class,
        () -> new DependencyParserCrossValidatorTool().run("conllu", new String[] {
            "-lang", "eng", "-data", data.toString(), "-folds", "2"}));
    assertEquals(List.of(PERFECT_SCORES, PERFECT_SCORES_EXCLUDING_PUNCTUATION), log);
  }

  @Test
  void testParserToolPrintsConllu() {
    train();
    final InputStream previousIn = System.in;
    final PrintStream previousOut = System.out;
    final ByteArrayOutputStream output = new ByteArrayOutputStream();
    try {
      System.setIn(new ByteArrayInputStream("dogs_NOUN run_VERB\n".getBytes(StandardCharsets.UTF_8)));
      System.setOut(new PrintStream(output, true, StandardCharsets.UTF_8));
      new DependencyParserMETool().run(new String[] {model.toString()});
    } finally {
      System.setIn(previousIn);
      System.setOut(previousOut);
    }
    assertEquals("1\tdogs\t_\t_\tNOUN\t_\t2\tnsubj\t_\t_\n"
        + "2\trun\t_\t_\tVERB\t_\t0\troot\t_\t_\n\n",
        output.toString(StandardCharsets.UTF_8).replace("\r\n", "\n"));
  }

  @Test
  void testConlluFormatReadsTheSelectedTagset() throws IOException {
    final var factory = StreamFactoryRegistry.getFactory(DependencySample.class, "conllu");
    try (ObjectStream<DependencySample> samples = factory.create(new String[] {
        "-data", data.toString(), "-tagset", "x"})) {
      assertEquals("NNS", samples.read().getTags()[0]);
    }
    try (ObjectStream<DependencySample> samples = factory.create(new String[] {
        "-data", data.toString()})) {
      assertEquals("NOUN", samples.read().getTags()[0]);
    }
  }

  @Test
  void testConlluFormatRejectsUnknownTagset() {
    final var factory = StreamFactoryRegistry.getFactory(DependencySample.class, "conllu");
    assertThrows(TerminateToolException.class, () -> factory.create(new String[] {
        "-data", data.toString(), "-tagset", "invalid"}));
  }

  @Test
  void testCrossValidatorRejectsFoldCountBelowTwo() {
    assertThrows(IllegalArgumentException.class,
        () -> new DependencyParserCrossValidatorTool().run("conllu", new String[] {
            "-lang", "eng", "-data", data.toString(), "-folds", "1"}));
  }

  @Test
  void testCrossValidatorDefaultsToTenFolds() {
    final List<String> log = logOf(DependencyParserCrossValidatorTool.class,
        () -> new DependencyParserCrossValidatorTool().run("conllu", new String[] {
            "-lang", "eng", "-data", data.toString(), "-misclassified", "true"}));
    assertEquals(List.of(PERFECT_SCORES, PERFECT_SCORES_EXCLUDING_PUNCTUATION), log);
  }

  @Test
  void testEvaluatorPrintsMisparsedSamples() throws IOException {
    train();
    final Path test = dir.resolve("test.conllu");
    Files.writeString(test, REVERSED_SENTENCE);
    final List<String> log = logOf(DependencyEvaluationErrorListener.class,
        () -> new DependencyParserEvaluatorTool().run("conllu", new String[] {
            "-model", model.toString(), "-data", test.toString(), "-misclassified", "true"}));
    assertEquals(List.of(EXPECTED_REVERSED + "\n" + PREDICTED_SENTENCE), log);
  }

  @Test
  void testEvaluatorWithoutMisclassifiedFlagPrintsNoSamples() throws IOException {
    train();
    final Path test = dir.resolve("test.conllu");
    Files.writeString(test, REVERSED_SENTENCE);
    final List<String> log = logOf(DependencyEvaluationErrorListener.class,
        () -> new DependencyParserEvaluatorTool().run("conllu", new String[] {
            "-model", model.toString(), "-data", test.toString()}));
    assertEquals(List.of(), log);
  }

  @Test
  void testErrorListenerPrintsBothSamples() {
    final ByteArrayOutputStream output = new ByteArrayOutputStream();
    final DependencySample reference = new DependencySample(
        new String[] {"dogs", "run"}, new String[] {"NOUN", "VERB"}, REVERSED);
    final DependencySample prediction = new DependencySample(
        new String[] {"dogs", "run"}, new String[] {"NOUN", "VERB"}, GOLD);
    new DependencyEvaluationErrorListener(output).misclassified(reference, prediction);
    assertEquals(EXPECTED_REVERSED + "\n" + PREDICTED_SENTENCE + "\n\n",
        output.toString(StandardCharsets.UTF_8).replace("\r\n", "\n"));
  }
}
