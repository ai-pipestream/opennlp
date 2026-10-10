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
import opennlp.tools.sentdetect.SentenceDetectorFactory;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.sentdetect.SentenceSample;
import opennlp.tools.sentdetect.SentenceSampleStream;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;
import opennlp.tools.util.TrainingParameters;

/**
 * The samples printed in the Sentence Detector chapter of the manual.
 */
class SentdetectChapterTest extends DocExampleSupport {

  private static SentenceModel published;

  @BeforeAll
  static void loadModel() throws IOException {
    try (InputStream in = Files.newInputStream(stageModel("opennlp-models-sentdetect-*.jar",
        "opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin"))) {
      published = new SentenceModel(in);
    }
    stageFile("en-custom-sent.train", 10,
        "Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29.",
        "Mr. Vinken is chairman of Elsevier N.V., the Dutch publishing group.",
        "Rudolph Agnew, 55 years old, was named a director of this British industrial group.",
        "");
  }

  @Test
  void loadsTheModel() throws IOException {
    // docs:begin example.sentdetect.api.model
    try (InputStream modelIn = new FileInputStream("opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin")) {
      SentenceModel model = new SentenceModel(modelIn);
    }
    // docs:end
  }

  @Test
  void createsTheDetector() {
    SentenceModel model = published;
    // docs:begin example.sentdetect.api.detector
    SentenceDetectorME sentenceDetector = new SentenceDetectorME(model);
    // docs:end
    Assertions.assertNotNull(sentenceDetector);
  }

  @Test
  void detectsSentences() {
    SentenceDetectorME sentenceDetector = new SentenceDetectorME(published);
    // docs:begin example.sentdetect.api.sentences
    String[] sentences = sentenceDetector.sentDetect("  First sentence. Second sentence. ");
    // docs:end
    Assertions.assertArrayEquals(new String[] {"First sentence.", "Second sentence."}, sentences);
  }

  @Test
  void detectsSentenceSpans() {
    SentenceDetectorME sentenceDetector = new SentenceDetectorME(published);
    // docs:begin example.sentdetect.api.spans
    Span[] sentences = sentenceDetector.sentPosDetect("  First sentence. Second sentence. ");
    // docs:end
    Assertions.assertArrayEquals(new Span[] {new Span(2, 17), new Span(18, 34)}, sentences);
  }

  @Test
  void trainsAndSavesAModel() throws IOException {
    File modelFile = workFile("en-custom-sent.bin").toFile();
    // docs:begin example.sentdetect.training.api
    ObjectStream<String> lineStream = new PlainTextByLineStream(
        new MarkableFileInputStreamFactory(new File("en-custom-sent.train")), StandardCharsets.UTF_8);

    SentenceModel model;
    try (ObjectStream<SentenceSample> sampleStream = new SentenceSampleStream(lineStream)) {
      model = SentenceDetectorME.train("eng", sampleStream,
          new SentenceDetectorFactory("eng", true, null, null), TrainingParameters.defaultParams());
    }

    try (OutputStream modelOut = new BufferedOutputStream(new FileOutputStream(modelFile))) {
      model.serialize(modelOut);
    }
    // docs:end
    Assertions.assertTrue(modelFile.length() > 0);
  }
}
