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
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import opennlp.examples.DocExampleSupport;
import opennlp.tools.sentiment.SentimentEvaluator;
import opennlp.tools.sentiment.SentimentFactory;
import opennlp.tools.sentiment.SentimentME;
import opennlp.tools.sentiment.SentimentModel;
import opennlp.tools.sentiment.SentimentSample;
import opennlp.tools.sentiment.SentimentSampleStream;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;

/**
 * The samples printed in the Sentiment Analysis chapter of the manual.
 * <p>
 * The training data is staged as {@code en-sentiment.train} from the four samples the chapter
 * prints. The model loading listing uses {@code ...} as a placeholder and is not verified here.
 */
class SentimentChapterTest extends DocExampleSupport {

  private static final String TRAINING_FILE = "en-sentiment.train";

  private static SentimentModel trained;

  @BeforeAll
  static void trainModel() throws IOException {
    stageFile(TRAINING_FILE, 10,
        "positive I love this movie it is absolutely wonderful and amazing",
        "positive This product is great and I am very happy with it",
        "negative I hate this product it broke after one day of use",
        "negative Terrible experience the worst customer service I have ever had");
    try (ObjectStream<SentimentSample> samples = samples()) {
      trained = SentimentME.train("eng", samples, TrainingParameters.defaultParams(),
          new SentimentFactory());
    }
  }

  @Test
  void predictsSentiment() {
    SentimentModel model = trained;
    // docs:begin example.sentiment.api.predict
    SentimentME sentiment = new SentimentME(model);

    // Predict from a raw sentence string (tokenized internally)
    String result = sentiment.predict("I love this product");

    // Or predict from pre-tokenized input
    String[] tokens = new String[]{"I", "love", "this", "product"};
    String result2 = sentiment.predict(tokens);

    // Access the probability distribution over sentiment categories
    double[] probabilities = sentiment.probabilities(tokens);
    String bestSentiment = sentiment.getBestSentiment(probabilities);
    // docs:end
    Assertions.assertEquals("positive", result);
    Assertions.assertEquals(result, result2);
    Assertions.assertEquals(result, bestSentiment);
    Assertions.assertEquals(2, probabilities.length);
  }

  @Test
  void trainsAModel() throws IOException {
    // docs:begin example.sentiment.training.api
    SentimentModel model;

    InputStreamFactory dataIn = new MarkableFileInputStreamFactory(
        new File("en-sentiment.train"));

    ObjectStream<String> lineStream =
        new PlainTextByLineStream(dataIn, StandardCharsets.UTF_8);
    ObjectStream<SentimentSample> sampleStream =
        new SentimentSampleStream(lineStream);

    model = SentimentME.train("eng", sampleStream,
        TrainingParameters.defaultParams(), new SentimentFactory());
    // docs:end
    sampleStream.close();
    Assertions.assertNotNull(model);
  }

  @Test
  void serializesTheModel() throws IOException {
    SentimentModel model = trained;
    // docs:begin example.sentiment.training.api.serialize
    try (OutputStream modelOut = new BufferedOutputStream(
        new FileOutputStream("en-sentiment.bin"))) {
      model.serialize(modelOut);
    }
    // docs:end
    Assertions.assertTrue(workFile("en-sentiment.bin").toFile().length() > 0);
  }

  @Test
  void evaluatesTheModel() throws IOException {
    SentimentModel model = trained;
    try (ObjectStream<SentimentSample> testSampleStream = samples()) {
      // docs:begin example.sentiment.evaluation.api
      SentimentME sentiment = new SentimentME(model);
      SentimentEvaluator evaluator = new SentimentEvaluator(sentiment);
      evaluator.evaluate(testSampleStream);

      System.out.println(evaluator.getFMeasure());
      // docs:end
      Assertions.assertTrue(evaluator.getFMeasure().getFMeasure() > 0);
    }
  }

  private static ObjectStream<SentimentSample> samples() throws IOException {
    return new SentimentSampleStream(new PlainTextByLineStream(
        new MarkableFileInputStreamFactory(workFile(TRAINING_FILE).toFile()),
        StandardCharsets.UTF_8));
  }
}
