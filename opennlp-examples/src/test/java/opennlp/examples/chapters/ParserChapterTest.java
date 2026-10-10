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

package opennlp.examples.chapters;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import opennlp.examples.DocExampleSupport;
import opennlp.tools.cmdline.parser.ParserTool;
import opennlp.tools.parser.HeadRules;
import opennlp.tools.parser.Parse;
import opennlp.tools.parser.ParseSampleStream;
import opennlp.tools.parser.Parser;
import opennlp.tools.parser.ParserCrossValidator;
import opennlp.tools.parser.ParserEvaluator;
import opennlp.tools.parser.ParserFactory;
import opennlp.tools.parser.ParserModel;
import opennlp.tools.parser.ParserType;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.eval.FMeasure;

/**
 * The samples printed in the Parser chapter of the manual.
 * <p>
 * No English parser model is published as a Maven artifact, so the class trains a chunking
 * parser from the small treebank sample that the opennlp-runtime parser tests use, and stages it,
 * the head rules and the training data under the file names the manual prints. The
 * {@code opennlp.parser.resources.dir} system property points at those test resources.
 */
class ParserChapterTest extends DocExampleSupport {

  private static final Path RESOURCES = Paths.get(System.getProperty(
      "opennlp.parser.resources.dir",
      "../opennlp-core/opennlp-runtime/src/test/resources/opennlp/tools/parser"));

  private static final String MODEL_FILE = "en-parser-chunking.bin";

  private static final String HEAD_RULES_FILE = "head_rules";

  private static final String TRAINING_FILE = "parsing.train";

  private static HeadRules headRules;

  private static ParserModel trained;

  @BeforeAll
  static void trainModel() throws IOException {
    Files.copy(RESOURCES.resolve("en_head_rules"), workFile(HEAD_RULES_FILE),
        StandardCopyOption.REPLACE_EXISTING);
    Files.copy(RESOURCES.resolve("parser.train"), workFile(TRAINING_FILE),
        StandardCopyOption.REPLACE_EXISTING);
    try (Reader reader = new InputStreamReader(
        new FileInputStream(workFile(HEAD_RULES_FILE).toFile()), StandardCharsets.UTF_8)) {
      headRules = new opennlp.tools.parser.lang.en.HeadRules(reader);
    }
    try (ObjectStream<Parse> samples = samples()) {
      trained = opennlp.tools.parser.chunking.Parser.train("eng", samples, headRules,
          TrainingParameters.defaultParams());
    }
    try (OutputStream out = new FileOutputStream(workFile(MODEL_FILE).toFile())) {
      trained.serialize(out);
    }
  }

  @Test
  void loadsTheModel() throws IOException {
    // docs:begin example.parser.api.model
    ParserModel model;
    try (InputStream modelIn = new FileInputStream("en-parser-chunking.bin")) {
      model = new ParserModel(modelIn);
    }
    // docs:end
    Assertions.assertNotNull(model);
  }

  @Test
  void createsTheParser() {
    ParserModel model = trained;
    // docs:begin example.parser.api.parser
    Parser parser = ParserFactory.create(model);
    // docs:end
    Assertions.assertNotNull(parser);
  }

  @Test
  void parsesASentence() {
    Parser parser = ParserFactory.create(trained);
    // docs:begin example.parser.api.parse
    String sentence = "The quick brown fox jumps over the lazy dog .";
    Parse[] topParses = ParserTool.parseLine(sentence, parser, 1);
    // docs:end
    Assertions.assertEquals(1, topParses.length);
    final StringBuffer shown = new StringBuffer();
    topParses[0].show(shown);
    Assertions.assertTrue(shown.toString().startsWith("(TOP "), shown::toString);
  }

  @Test
  void readsHeadRules() throws IOException {
    // docs:begin example.parser.training.api.headrules
    HeadRules rules;
    try (Reader rulesReader = new InputStreamReader(
        new FileInputStream("head_rules"), StandardCharsets.UTF_8)) {
      rules = new opennlp.tools.parser.lang.en.HeadRules(rulesReader);
    }
    // docs:end
    Assertions.assertEquals(headRules, rules);
  }

  @Test
  void trainsAndSavesAModel() throws IOException {
    HeadRules rules = headRules;
    // docs:begin example.parser.training.api.train
    InputStreamFactory inputStreamFactory =
        new MarkableFileInputStreamFactory(new File("parsing.train"));
    ObjectStream<String> stringStream =
        new PlainTextByLineStream(inputStreamFactory, StandardCharsets.UTF_8);

    ParserModel model;
    try (ObjectStream<Parse> sampleStream = new ParseSampleStream(stringStream)) {
      // opennlp.tools.parser.treeinsert.Parser.train trains the tree insert parser instead
      model = opennlp.tools.parser.chunking.Parser.train("eng", sampleStream, rules,
          TrainingParameters.defaultParams());
    }

    try (OutputStream modelOut = new BufferedOutputStream(
        new FileOutputStream("en-parser-chunking.bin"))) {
      model.serialize(modelOut);
    }
    // docs:end
    Assertions.assertEquals(ParserType.CHUNKING, model.getParserType());
  }

  @Test
  void evaluatesTheModel() throws IOException {
    ParserModel model = trained;
    try (ObjectStream<Parse> sampleStream = samples()) {
      // docs:begin example.parser.eval.api
      Parser parser = ParserFactory.create(model);
      ParserEvaluator evaluator = new ParserEvaluator(parser);
      evaluator.evaluate(sampleStream);

      FMeasure result = evaluator.getFMeasure();
      System.out.println(result.toString());
      // docs:end
      Assertions.assertTrue(result.getFMeasure() > 0);
    }
  }

  @Test
  void crossValidates() throws IOException {
    HeadRules rules = headRules;
    // docs:begin example.parser.eval.crossval
    InputStreamFactory inputStreamFactory =
        new MarkableFileInputStreamFactory(new File("parsing.train"));
    ObjectStream<String> stringStream =
        new PlainTextByLineStream(inputStreamFactory, StandardCharsets.UTF_8);
    ObjectStream<Parse> sampleStream = new ParseSampleStream(stringStream);
    ParserCrossValidator evaluator = new ParserCrossValidator("eng",
        TrainingParameters.defaultParams(), rules, ParserType.CHUNKING);
    evaluator.evaluate(sampleStream, 10);

    FMeasure result = evaluator.getFMeasure();
    System.out.println(result.toString());
    // docs:end
    sampleStream.close();
    Assertions.assertTrue(result.getFMeasure() > 0);
  }

  private static ObjectStream<Parse> samples() throws IOException {
    return new ParseSampleStream(new PlainTextByLineStream(
        new MarkableFileInputStreamFactory(workFile(TRAINING_FILE).toFile()),
        StandardCharsets.UTF_8));
  }
}
