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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.examples.DocExampleSupport;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSSample;
import opennlp.tools.postag.POSTaggerFactory;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.postag.WordTagSampleStream;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Sequence;
import opennlp.tools.util.TrainingParameters;

/**
 * The samples printed in the Part-of-Speech Tagger chapter of the manual.
 * <p>
 * Each method runs one listing. What sits between the {@code docs:begin} and {@code docs:end}
 * markers is the text the manual prints, character for character; everything before and after
 * exists so that text can run.
 */
class PostaggerChapterTest extends DocExampleSupport {

  private static final String MODEL_JAR = "opennlp-models-pos-*.jar";

  private static final String MODEL_FILE = "opennlp-en-ud-ewt-pos-1.3-2.5.4.bin";

  private static final String TRAINING_FILE = "en-custom-pos.train";

  private static final String[] SHORT_SENTENCE = {"Most", "large", "cities", "."};

  @Test
  void postagger_tagging_api_model() throws IOException {
    stageModel(MODEL_JAR, MODEL_FILE);
    // docs:begin
    try (InputStream modelIn = new FileInputStream("opennlp-en-ud-ewt-pos-1.3-2.5.4.bin")) {
      POSModel model = new POSModel(modelIn);
    }
    // docs:end
  }

  @Test
  void postagger_tagging_api_tagger() throws IOException {
    POSModel model = loadModel();
    // docs:begin
    POSTaggerME tagger = new POSTaggerME(model);
    // docs:end
    Assertions.assertNotNull(tagger);
  }

  @Test
  void postagger_tagging_api_tag() throws IOException {
    POSTaggerME tagger = new POSTaggerME(loadModel());
    // docs:begin
    String[] sent = new String[]{"Most", "large", "cities", "in", "the", "US", "had",
                                 "morning", "and", "afternoon", "newspapers", "."};
    String[] tags = tagger.tag(sent);
    // docs:end
    Assertions.assertEquals(sent.length, tags.length);
    for (String tag : tags) {
      Assertions.assertFalse(tag == null || tag.isBlank());
    }
  }

  @Test
  void postagger_tagging_api_probs() throws IOException {
    POSTaggerME tagger = new POSTaggerME(loadModel());
    tagger.tag(SHORT_SENTENCE);
    // docs:begin
    double[] probs = tagger.probs();
    // docs:end
    Assertions.assertEquals(SHORT_SENTENCE.length, probs.length);
  }

  @Test
  void postagger_tagging_api_topk() throws IOException {
    POSTaggerME tagger = new POSTaggerME(loadModel());
    String[] sent = SHORT_SENTENCE;
    // docs:begin
    Sequence[] topSequences = tagger.topKSequences(sent);
    // docs:end
    Assertions.assertTrue(topSequences.length > 0);
  }

  @Test
  void postagger_training_api_train() throws IOException {
    stageTrainingData();
    // docs:begin
    POSModel model = null;

    try {
      ObjectStream<String> lineStream = new PlainTextByLineStream(
          new MarkableFileInputStreamFactory(new File("en-custom-pos.train")), StandardCharsets.UTF_8);

      ObjectStream<POSSample> sampleStream = new WordTagSampleStream(lineStream);

      model = POSTaggerME.train("eng", sampleStream, TrainingParameters.defaultParams(),
          new POSTaggerFactory());
    } catch (IOException e) {
      e.printStackTrace();
    }
    // docs:end
    Assertions.assertNotNull(model);
  }

  @Test
  void postagger_training_api_serialize() throws IOException {
    POSModel model = loadModel();
    File modelFile = workFile("en-custom-pos-maxent.bin").toFile();
    // docs:begin
    try (OutputStream modelOut = new BufferedOutputStream(new FileOutputStream(modelFile))) {
      model.serialize(modelOut);
    }
    // docs:end
    Assertions.assertTrue(modelFile.length() > 0);
  }

  private static POSModel loadModel() throws IOException {
    final Path staged = stageModel(MODEL_JAR, MODEL_FILE);
    try (InputStream modelIn = Files.newInputStream(staged)) {
      return new POSModel(modelIn);
    }
  }

  /*
   * Default training parameters drop features seen fewer than five times, so the fixture repeats
   * its sentences often enough for a model to come out of it.
   */
  private static void stageTrainingData() throws IOException {
    final List<String> lines = new ArrayList<>();
    for (int i = 0; i < 10; i++) {
      lines.add("It_PRON is_AUX Spring_PROPN ._PUNCT");
      lines.add("The_DET flowers_NOUN are_AUX red_ADJ ._PUNCT");
      lines.add("Yellow_ADJ is_AUX my_PRON favourite_ADJ colour_NOUN ._PUNCT");
    }
    stageFile(TRAINING_FILE, lines.toArray(new String[0]));
  }
}
