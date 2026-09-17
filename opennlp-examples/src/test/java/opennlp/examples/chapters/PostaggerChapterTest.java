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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
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
import opennlp.tools.util.StringUtil;
import opennlp.tools.util.TrainingParameters;

/**
 * The samples printed in the Part-of-Speech Tagger chapter of the manual.
 * <p>
 * What sits between a {@code docs:begin} and {@code docs:end} marker is the text the manual
 * prints, character for character; everything before and after exists so that text can run.
 */
class PostaggerChapterTest extends DocExampleSupport {

  private static final String[] SHORT_SENTENCE = {"Most", "large", "cities", "."};

  private static POSModel published;

  @BeforeAll
  static void loadModel() throws IOException {
    try (InputStream in = Files.newInputStream(stageModel("opennlp-models-pos-*.jar",
        "opennlp-en-ud-ewt-pos-1.3-2.5.4.bin"))) {
      published = new POSModel(in);
    }
    stageFile("en-custom-pos.train", 10,
        "It_PRON is_AUX Spring_PROPN ._PUNCT",
        "The_DET flowers_NOUN are_AUX red_ADJ ._PUNCT",
        "Yellow_ADJ is_AUX my_PRON favourite_ADJ colour_NOUN ._PUNCT");
  }

  @Test
  void loadsTheModel() throws IOException {
    // docs:begin example.postagger.tagging.api.model
    try (InputStream modelIn = new FileInputStream("opennlp-en-ud-ewt-pos-1.3-2.5.4.bin")) {
      POSModel model = new POSModel(modelIn);
    }
    // docs:end
  }

  @Test
  void createsTheTagger() {
    POSModel model = published;
    // docs:begin example.postagger.tagging.api.tagger
    POSTaggerME tagger = new POSTaggerME(model);
    // docs:end
    Assertions.assertNotNull(tagger);
  }

  @Test
  void tagsASentence() {
    POSTaggerME tagger = new POSTaggerME(published);
    // docs:begin example.postagger.tagging.api.tag
    String[] sent = new String[]{"Most", "large", "cities", "in", "the", "US", "had",
                                 "morning", "and", "afternoon", "newspapers", "."};
    String[] tags = tagger.tag(sent);
    // docs:end
    Assertions.assertEquals(sent.length, tags.length);
    for (String tag : tags) {
      Assertions.assertFalse(StringUtil.isUnicodeBlank(tag));
    }
  }

  @Test
  void returnsTagProbabilities() {
    POSTaggerME tagger = new POSTaggerME(published);
    tagger.tag(SHORT_SENTENCE);
    // docs:begin example.postagger.tagging.api.probs
    double[] probs = tagger.probs();
    // docs:end
    Assertions.assertEquals(SHORT_SENTENCE.length, probs.length);
  }

  @Test
  void returnsTopSequences() {
    POSTaggerME tagger = new POSTaggerME(published);
    String[] sent = SHORT_SENTENCE;
    // docs:begin example.postagger.tagging.api.topk
    Sequence[] topSequences = tagger.topKSequences(sent);
    // docs:end
    Assertions.assertTrue(topSequences.length > 0);
  }

  @Test
  void trainsAModel() {
    // docs:begin example.postagger.training.api.train
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
  void serializesTheModel() throws IOException {
    POSModel model = published;
    File modelFile = workFile("en-custom-pos-maxent.bin").toFile();
    // docs:begin example.postagger.training.api.serialize
    try (OutputStream modelOut = new BufferedOutputStream(new FileOutputStream(modelFile))) {
      model.serialize(modelOut);
    }
    // docs:end
    Assertions.assertTrue(modelFile.length() > 0);
  }
}
