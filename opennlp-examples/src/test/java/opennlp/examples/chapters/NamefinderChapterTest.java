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
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import opennlp.examples.DocExampleSupport;
import opennlp.tools.namefind.BioCodec;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.namefind.NameSampleDataStream;
import opennlp.tools.namefind.TokenNameFinderCrossValidator;
import opennlp.tools.namefind.TokenNameFinderEvaluator;
import opennlp.tools.namefind.TokenNameFinderFactory;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.eval.FMeasure;
import opennlp.tools.util.featuregen.AdaptiveFeatureGenerator;
import opennlp.tools.util.featuregen.AggregatedFeatureGenerator;
import opennlp.tools.util.featuregen.BigramNameFeatureGenerator;
import opennlp.tools.util.featuregen.BrownCluster;
import opennlp.tools.util.featuregen.BrownTokenFeatureGenerator;
import opennlp.tools.util.featuregen.CachedFeatureGenerator;
import opennlp.tools.util.featuregen.OutcomePriorFeatureGenerator;
import opennlp.tools.util.featuregen.PreviousMapFeatureGenerator;
import opennlp.tools.util.featuregen.SentenceFeatureGenerator;
import opennlp.tools.util.featuregen.TokenClassFeatureGenerator;
import opennlp.tools.util.featuregen.TokenFeatureGenerator;
import opennlp.tools.util.featuregen.WindowFeatureGenerator;

/**
 * The samples printed in the Name Finder chapter of the manual.
 * <p>
 * No English person name model is published as a Maven artifact, so the class trains one from a
 * small fixture in the name finder training format and stages it as {@code en-ner-person.bin}.
 * The two ONNX listings need an ONNX model and vocabulary and are not verified here.
 */
class NamefinderChapterTest extends DocExampleSupport {

  private static final String MODEL_FILE = "en-ner-person.bin";

  private static final String TRAINING_FILE = "en-ner-person.train";

  private static final String[] SENTENCE = {"Pierre", "Vinken", "is", "61", "years", "old", "."};

  private static TokenNameFinderModel trained;

  @BeforeAll
  static void trainModel() throws IOException {
    stageTrainingData();
    try (ObjectStream<NameSample> samples = samples()) {
      trained = NameFinderME.train("eng", "person", samples, TrainingParameters.defaultParams(),
          new TokenNameFinderFactory());
    }
    try (OutputStream out = new FileOutputStream(workFile(MODEL_FILE).toFile())) {
      trained.serialize(out);
    }
  }

  @Test
  void loadsTheModel() throws IOException {
    // docs:begin example.namefinder.api.model
    try (InputStream modelIn = new FileInputStream("en-ner-person.bin")) {
      TokenNameFinderModel model = new TokenNameFinderModel(modelIn);
    }
    // docs:end
  }

  @Test
  void createsTheNameFinder() {
    TokenNameFinderModel model = trained;
    // docs:begin example.namefinder.api.namefinder
    NameFinderME nameFinder = new NameFinderME(model);
    // docs:end
    Assertions.assertNotNull(nameFinder);
  }

  @Test
  void clearsAdaptiveDataPerDocument() {
    NameFinderME nameFinder = new NameFinderME(trained);
    String[][][] documents = {{SENTENCE, {"Mr", ".", "Vinken", "is", "chairman", "."}},
        {SENTENCE}};
    // docs:begin example.namefinder.api.documents
    for (String[][] document : documents) {
      for (String[] sentence : document) {
        Span[] nameSpans = nameFinder.find(sentence);
        // do something with the names
      }
      nameFinder.clearAdaptiveData();
    }
    // docs:end
  }

  @Test
  void findsNames() {
    NameFinderME nameFinder = new NameFinderME(trained);
    // docs:begin example.namefinder.api.find
    String[] sentence = new String[]{
        "Pierre",
        "Vinken",
        "is",
        "61",
        "years",
        "old",
        "."
        };

    Span[] nameSpans = nameFinder.find(sentence);
    // docs:end
    Assertions.assertArrayEquals(new Span[] {new Span(0, 2, "person")}, nameSpans);
    Assertions.assertEquals("person", nameSpans[0].getType());
  }

  @Test
  void trainsAndSavesAModel() throws IOException {
    // docs:begin example.namefinder.training.api
    TokenNameFinderFactory factory = TokenNameFinderFactory.create(null, null,
        Collections.emptyMap(), new BioCodec());
    File trainingFile = new File("en-ner-person.train");
    ObjectStream<String> lineStream = new PlainTextByLineStream(
        new MarkableFileInputStreamFactory(trainingFile), StandardCharsets.UTF_8);

    TokenNameFinderModel trainedModel;
    try (ObjectStream<NameSample> sampleStream = new NameSampleDataStream(lineStream)) {
      trainedModel = NameFinderME.train("eng", "person", sampleStream,
          TrainingParameters.defaultParams(), factory);
    }

    File modelFile = new File("en-ner-person-custom.bin");
    try (OutputStream modelOut = new BufferedOutputStream(new FileOutputStream(modelFile))) {
      trainedModel.serialize(modelOut);
    }
    // docs:end
    Assertions.assertTrue(modelFile.length() > 0);
  }

  @Test
  void buildsAFeatureGenerator() throws IOException {
    BrownCluster dictResource = new BrownCluster(new ByteArrayInputStream(
        "Pierre\t0110\nVinken\t0111\n".getBytes(StandardCharsets.UTF_8)));
    // docs:begin example.namefinder.featuregen.api
    AdaptiveFeatureGenerator featureGenerator = new CachedFeatureGenerator(
        new AggregatedFeatureGenerator(
            new WindowFeatureGenerator(new TokenFeatureGenerator(), 2, 2),
            new WindowFeatureGenerator(new TokenClassFeatureGenerator(true), 2, 2),
            new OutcomePriorFeatureGenerator(),
            new PreviousMapFeatureGenerator(),
            new BigramNameFeatureGenerator(),
            new SentenceFeatureGenerator(true, false),
            new BrownTokenFeatureGenerator(dictResource)));
    // docs:end
    final List<String> features = new ArrayList<>();
    featureGenerator.createFeatures(features, SENTENCE, 0, null);
    Assertions.assertTrue(features.stream().anyMatch(f -> f.contains("0110")), features::toString);
  }

  @Test
  void createsANameFinderWithCustomFeatures() {
    TokenNameFinderModel model = trained;
    // docs:begin example.namefinder.featuregen.detector
    new NameFinderME(model);
    // docs:end
  }

  @Test
  void evaluatesTheModel() throws IOException {
    TokenNameFinderModel model = trained;
    try (ObjectStream<NameSample> sampleStream = samples()) {
      // docs:begin example.namefinder.eval.api
      TokenNameFinderEvaluator evaluator = new TokenNameFinderEvaluator(new NameFinderME(model));
      evaluator.evaluate(sampleStream);

      FMeasure result = evaluator.getFMeasure();
      System.out.println(result.toString());
      // docs:end
      Assertions.assertTrue(result.getFMeasure() > 0);
    }
  }

  @Test
  void crossValidates() throws IOException {
    // docs:begin example.namefinder.eval.crossval
    InputStreamFactory dataIn = new MarkableFileInputStreamFactory(new File("en-ner-person.train"));
    ObjectStream<NameSample> sampleStream = new NameSampleDataStream(
        new PlainTextByLineStream(dataIn, StandardCharsets.UTF_8));
    TokenNameFinderCrossValidator evaluator = new TokenNameFinderCrossValidator("eng",
        null, TrainingParameters.defaultParams(), new TokenNameFinderFactory());
    evaluator.evaluate(sampleStream, 10);

    FMeasure result = evaluator.getFMeasure();
    System.out.println(result.toString());
    // docs:end
    sampleStream.close();
    Assertions.assertTrue(result.getFMeasure() >= 0);
  }

  private static ObjectStream<NameSample> samples() throws IOException {
    return new NameSampleDataStream(new PlainTextByLineStream(
        new MarkableFileInputStreamFactory(workFile(TRAINING_FILE).toFile()),
        StandardCharsets.UTF_8));
  }

  /* The sentences the chapter prints, in the name finder training format. */
  private static void stageTrainingData() throws IOException {
    stageFile(TRAINING_FILE, 10,
        "<START:person> Pierre Vinken <END> , 61 years old , will join the board as a "
            + "nonexecutive director Nov. 29 .",
        "Mr . <START:person> Vinken <END> is chairman of Elsevier N.V. , the Dutch publishing "
            + "group .",
        "<START:person> Rudolph Agnew <END> , 55 years old and former chairman of Consolidated "
            + "Gold Fields PLC , was named a director of this British industrial conglomerate .",
        "<START:person> Pierre Vinken <END> is 61 years old .",
        "");
  }
}
