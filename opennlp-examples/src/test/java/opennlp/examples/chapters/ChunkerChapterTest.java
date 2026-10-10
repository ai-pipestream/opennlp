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
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import opennlp.examples.DocExampleSupport;
import opennlp.tools.chunker.ChunkSample;
import opennlp.tools.chunker.ChunkSampleStream;
import opennlp.tools.chunker.ChunkerFactory;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Sequence;
import opennlp.tools.util.TrainingParameters;

/**
 * The samples printed in the Chunker chapter of the manual.
 * <p>
 * No English chunker model is published as a Maven artifact, so the class trains one from a small
 * fixture in the CoNLL-2000 format the chapter describes and stages it under the file name the
 * manual prints.
 */
class ChunkerChapterTest extends DocExampleSupport {

  private static final String MODEL_FILE = "en-chunker.bin";

  private static final String TRAINING_FILE = "en-chunker.train";

  private static final String[] SENT = {"Rockwell", "International", "Corp.", "'s",
      "Tulsa", "unit", "said", "it", "signed", "a", "tentative", "agreement",
      "extending", "its", "contract", "with", "Boeing", "Co.", "to",
      "provide", "structural", "parts", "for", "Boeing", "'s", "747",
      "jetliners", "."};

  private static final String[] POS = {"NNP", "NNP", "NNP", "POS", "NNP", "NN",
      "VBD", "PRP", "VBD", "DT", "JJ", "NN", "VBG", "PRP$", "NN", "IN",
      "NNP", "NNP", "TO", "VB", "JJ", "NNS", "IN", "NNP", "POS", "CD", "NNS",
      "."};

  private static final String[] CHUNKS = {"B-NP", "I-NP", "I-NP", "B-NP", "I-NP", "I-NP",
      "B-VP", "B-NP", "B-VP", "B-NP", "I-NP", "I-NP", "B-VP", "B-NP", "I-NP", "B-PP",
      "B-NP", "I-NP", "B-VP", "I-VP", "B-NP", "I-NP", "B-PP", "B-NP", "B-NP", "I-NP", "I-NP",
      "O"};

  private static ChunkerModel trained;

  @BeforeAll
  static void trainModel() throws IOException {
    stageTrainingData();
    try (ObjectStream<ChunkSample> samples = new ChunkSampleStream(new PlainTextByLineStream(
        new MarkableFileInputStreamFactory(workFile(TRAINING_FILE).toFile()),
        StandardCharsets.UTF_8))) {
      trained = ChunkerME.train("eng", samples, TrainingParameters.defaultParams(),
          new ChunkerFactory());
    }
    try (OutputStream out = new FileOutputStream(workFile(MODEL_FILE).toFile())) {
      trained.serialize(out);
    }
  }

  @Test
  void loadsTheModel() throws IOException {
    // docs:begin example.chunker.api.model
    ChunkerModel model;

    try (InputStream modelIn = new FileInputStream("en-chunker.bin")) {
      model = new ChunkerModel(modelIn);
    }
    // docs:end
    Assertions.assertNotNull(model);
  }

  @Test
  void createsTheChunker() {
    ChunkerModel model = trained;
    // docs:begin example.chunker.api.chunker
    ChunkerME chunker = new ChunkerME(model);
    // docs:end
    Assertions.assertNotNull(chunker);
  }

  @Test
  void chunksASentence() {
    ChunkerME chunker = new ChunkerME(trained);
    // docs:begin example.chunker.api.chunk
    String[] sent = new String[] { "Rockwell", "International", "Corp.", "'s",
        "Tulsa", "unit", "said", "it", "signed", "a", "tentative", "agreement",
        "extending", "its", "contract", "with", "Boeing", "Co.", "to",
        "provide", "structural", "parts", "for", "Boeing", "'s", "747",
        "jetliners", "." };

    String[] pos = new String[] { "NNP", "NNP", "NNP", "POS", "NNP", "NN",
        "VBD", "PRP", "VBD", "DT", "JJ", "NN", "VBG", "PRP$", "NN", "IN",
        "NNP", "NNP", "TO", "VB", "JJ", "NNS", "IN", "NNP", "POS", "CD", "NNS",
        "." };

    String[] tag = chunker.chunk(sent, pos);
    // docs:end
    Assertions.assertArrayEquals(CHUNKS, tag);
  }

  @Test
  void returnsChunkProbabilities() {
    ChunkerME chunker = new ChunkerME(trained);
    chunker.chunk(SENT, POS);
    // docs:begin example.chunker.api.probs
    double[] probs = chunker.probs();
    // docs:end
    Assertions.assertEquals(SENT.length, probs.length);
  }

  @Test
  void returnsTopSequences() {
    ChunkerME chunker = new ChunkerME(trained);
    String[] sent = SENT;
    String[] pos = POS;
    // docs:begin example.chunker.api.topk
    Sequence[] topSequences = chunker.topKSequences(sent, pos);
    // docs:end
    Assertions.assertTrue(topSequences.length > 0);
    Assertions.assertEquals(SENT.length, topSequences[0].getOutcomes().size());
  }

  @Test
  void trainsAndSavesAModel() throws IOException {
    File modelFile = workFile("en-chunker-custom.bin").toFile();
    // docs:begin example.chunker.training.api
    ObjectStream<String> lineStream = new PlainTextByLineStream(
        new MarkableFileInputStreamFactory(new File("en-chunker.train")), StandardCharsets.UTF_8);

    ChunkerModel model;
    try (ObjectStream<ChunkSample> sampleStream = new ChunkSampleStream(lineStream)) {
      model = ChunkerME.train("eng", sampleStream, TrainingParameters.defaultParams(),
          new ChunkerFactory());
    }

    try (OutputStream modelOut = new BufferedOutputStream(new FileOutputStream(modelFile))) {
      model.serialize(modelOut);
    }
    // docs:end
    Assertions.assertTrue(modelFile.length() > 0);
  }

  /* The first sentence and the training sample the chapter prints, in CoNLL-2000 format. */
  private static void stageTrainingData() throws IOException {
    final List<String> lines = new ArrayList<>();
    for (int t = 0; t < SENT.length; t++) {
      lines.add(SENT[t] + " " + POS[t] + " " + CHUNKS[t]);
    }
    lines.addAll(List.of("", "He PRP B-NP", "reckons VBZ B-VP", "the DT B-NP",
        "current JJ I-NP", "account NN I-NP", "deficit NN I-NP", "will MD B-VP",
        "narrow VB I-VP", "to TO B-PP", "only RB B-NP", "# # I-NP", "1.8 CD I-NP",
        "billion CD I-NP", "in IN B-PP", "September NNP B-NP", ". . O", ""));
    stageFile(TRAINING_FILE, 10, lines.toArray(new String[0]));
  }
}
