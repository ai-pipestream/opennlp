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
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

import de.hhn.mi.configuration.KernelType;
import de.hhn.mi.configuration.SvmConfigurationImpl;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import opennlp.examples.DocExampleSupport;
import opennlp.tools.doccat.BagOfWordsFeatureGenerator;
import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizer;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.doccat.FeatureGenerator;
import opennlp.tools.ml.libsvm.doccat.DocumentCategorizerSVM;
import opennlp.tools.ml.libsvm.doccat.FeatureSelectionStrategy;
import opennlp.tools.ml.libsvm.doccat.SvmDoccatConfiguration;
import opennlp.tools.ml.libsvm.doccat.SvmDoccatModel;
import opennlp.tools.ml.libsvm.doccat.TermWeightingStrategy;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.StringUtil;
import opennlp.tools.util.TrainingParameters;

/**
 * The samples printed in the Document Categorizer chapter of the manual.
 * <p>
 * The training data is staged as {@code en-doccat.train} in the format the chapter describes.
 * The two listings that use {@code ...} as a placeholder and the two ONNX listings are not
 * verified here.
 */
class DoccatChapterTest extends DocExampleSupport {

  private static final String TRAINING_FILE = "en-doccat.train";

  private static final String[] DECREASE = StringUtil.split("Major acquisitions that have a "
      + "lower gross margin than the existing network also had a negative impact on the overall "
      + "gross margin , but it should improve following the implementation of its integration "
      + "strategies .", ' ');

  @BeforeAll
  static void stageTrainingData() throws IOException {
    stageFile(TRAINING_FILE, 10,
        "GMDecrease " + String.join(" ", DECREASE),
        "GMDecrease The gross margin fell because of lower prices .",
        "GMIncrease The upward movement of gross margin resulted from amounts pursuant to "
            + "adjustments to obligations towards dealers .",
        "GMIncrease The gross margin rose because of higher prices .");
  }

  @Test
  void trainsAnSvmModel() throws IOException {
    try (ObjectStream<DocumentSample> sampleStream = samples()) {
      // docs:begin example.doccat.api.svm.train
      // Configure the SVM categorizer
      SvmDoccatConfiguration config = new SvmDoccatConfiguration.Builder()
          .setSvmConfiguration(new SvmConfigurationImpl.Builder()
              .setKernelType(KernelType.LINEAR)
              .setProbability(true)
              .build())
          .setTermWeightingStrategy(TermWeightingStrategy.TF_IDF)
          .setFeatureSelectionStrategy(FeatureSelectionStrategy.INFORMATION_GAIN)
          .setMaxFeatures(1000)
          .setScaleFeatures(true)
          .setScaleRange(0.0, 1.0)
          .build();

      // Train from a DocumentSample stream with a bag-of-words feature generator
      FeatureGenerator featureGenerator = new BagOfWordsFeatureGenerator();
      SvmDoccatModel model = DocumentCategorizerSVM.train("eng", sampleStream, config, featureGenerator);

      // Serialize the model
      try (OutputStream out = new FileOutputStream("doccat-svm.bin")) {
        model.serialize(out);
      }
      // docs:end
    }
    Assertions.assertTrue(workFile("doccat-svm.bin").toFile().length() > 0);
  }

  @Test
  void classifiesWithAnSvmModel() throws Exception {
    try (ObjectStream<DocumentSample> samples = samples();
         OutputStream out = new FileOutputStream(workFile("doccat-svm.bin").toFile())) {
      DocumentCategorizerSVM.train("eng", samples, new BagOfWordsFeatureGenerator())
          .serialize(out);
    }
    // docs:begin example.doccat.api.svm.classify
    // Load the model
    SvmDoccatModel model;
    try (InputStream in = new FileInputStream("doccat-svm.bin")) {
      model = SvmDoccatModel.deserialize(in);
    }

    // Classify
    DocumentCategorizer categorizer = new DocumentCategorizerSVM(model, new BagOfWordsFeatureGenerator());
    double[] outcomes = categorizer.categorize(
        new String[]{"The", "gross", "margin", "rose", "because", "of", "higher", "prices", "."});
    String bestCategory = categorizer.getBestCategory(outcomes);
    // docs:end
    Assertions.assertEquals("GMIncrease", bestCategory);
  }

  @Test
  void trainsAMaxentModel() {
    // docs:begin example.doccat.training.api
    DoccatModel model = null;
    try {
      ObjectStream<String> lineStream = new PlainTextByLineStream(
          new MarkableFileInputStreamFactory(new File("en-doccat.train")), StandardCharsets.UTF_8);

      ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);

      model = DocumentCategorizerME.train("eng", sampleStream,
          TrainingParameters.defaultParams(), new DoccatFactory());
    } catch (IOException e) {
      e.printStackTrace();
    }
    // docs:end
    Assertions.assertNotNull(model);
    final DocumentCategorizerME categorizer = new DocumentCategorizerME(model);
    Assertions.assertEquals("GMDecrease",
        categorizer.getBestCategory(categorizer.categorize(DECREASE)));
  }

  @Test
  void serializesAMaxentModel() throws IOException {
    DoccatModel model;
    try (ObjectStream<DocumentSample> samples = samples()) {
      model = DocumentCategorizerME.train("eng", samples, TrainingParameters.defaultParams(),
          new DoccatFactory());
    }
    File modelFile = workFile("en-doccat.bin").toFile();
    // docs:begin example.doccat.training.api.serialize
    try (OutputStream modelOut = new BufferedOutputStream(new FileOutputStream(modelFile))) {
      model.serialize(modelOut);
    }
    // docs:end
    Assertions.assertTrue(modelFile.length() > 0);
  }

  @Test
  void configuresSvmTraining() throws IOException {
    try (ObjectStream<DocumentSample> sampleStream = samples()) {
      // docs:begin example.doccat.training.svm
      SvmDoccatConfiguration config = new SvmDoccatConfiguration.Builder()
          .setSvmConfiguration(new SvmConfigurationImpl.Builder()
              .setKernelType(KernelType.LINEAR)
              .setProbability(true)
              .build())
          .setTermWeightingStrategy(TermWeightingStrategy.TF_IDF)
          .setFeatureSelectionStrategy(FeatureSelectionStrategy.INFORMATION_GAIN)
          .setMaxFeatures(1000)
          .setScaleFeatures(true)
          .build();

      SvmDoccatModel model = DocumentCategorizerSVM.train("eng", sampleStream, config,
          new BagOfWordsFeatureGenerator());
      // docs:end
      Assertions.assertEquals(2, model.getNumberOfCategories());
    }
  }

  @Test
  void serializesAnSvmModel() throws Exception {
    SvmDoccatModel model;
    try (ObjectStream<DocumentSample> samples = samples()) {
      model = DocumentCategorizerSVM.train("eng", samples, new BagOfWordsFeatureGenerator());
    }
    // docs:begin example.doccat.training.svm.serialize
    // Save
    try (OutputStream out = new FileOutputStream("svm-doccat.bin")) {
      model.serialize(out);
    }

    // Load
    SvmDoccatModel loaded;
    try (InputStream in = new FileInputStream("svm-doccat.bin")) {
      loaded = SvmDoccatModel.deserialize(in);
    }
    // docs:end
    Assertions.assertEquals(model.getNumberOfCategories(), loaded.getNumberOfCategories());
  }

  private static ObjectStream<DocumentSample> samples() throws IOException {
    return new DocumentSampleStream(new PlainTextByLineStream(
        new MarkableFileInputStreamFactory(workFile(TRAINING_FILE).toFile()),
        StandardCharsets.UTF_8));
  }
}
