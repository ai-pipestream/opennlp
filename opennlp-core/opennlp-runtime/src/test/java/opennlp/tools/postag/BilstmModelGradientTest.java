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

package opennlp.tools.postag;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Finite-difference gradient checks for the full training wiring of
 * {@link BilstmPOSTrainer}: the cross-entropy loss of one sentence, backpropagated
 * through the tag scorer, both sentence BiLSTM directions, both character BiLSTM
 * directions, and the embedding tables, must match central-difference numerical
 * gradients. This is the hard gate that extends {@link LstmLayerGradientTest} from
 * the bare cell to everything training actually runs; dropout is pinned to zero so
 * the loss is deterministic.
 */
class BilstmModelGradientTest {

  private static final double EPSILON = 1e-5d;
  private static final double TOLERANCE = 1e-5d;

  private BilstmPOSTrainer.TrainingContext context;
  private BilstmPOSTrainer.TrainingContext.Worker worker;
  private POSSample sample;

  @BeforeEach
  void setUp() {
    final List<POSSample> corpus = List.of(
        new POSSample(new String[] {"The", "cat", "sat"},
            new String[] {"DET", "NOUN", "VERB"}),
        new POSSample(new String[] {"A", "dog", "ran"},
            new String[] {"DET", "NOUN", "VERB"}),
        new POSSample(new String[] {"Cats", "sleep"},
            new String[] {"NOUN", "VERB"}));
    final BilstmPOSTrainer.Settings settings = new BilstmPOSTrainer.Settings(
        4, 3, 3, 4, 1, 2, 1e-3d, 5.0d, 0.0d, 1, 10, 7L, 1);
    context = BilstmPOSTrainer.TrainingContext.build(corpus, settings, null, null);
    worker = context.newWorker();
    sample = corpus.get(0);
  }

  @Test
  void testOutputWeightGradients() {
    final double[][] analytic = analyticGradients(14);
    final double[][] weights = context.testingOutputWeights();
    for (int o = 0; o < weights.length; o++) {
      for (int j = 0; j < weights[o].length; j++) {
        assertClose(analytic[o][j], numerical(weights[o], j),
            "outputWeights[" + o + "][" + j + "]");
      }
    }
  }

  @Test
  void testOutputBiasGradients() {
    final double[][] analytic = analyticGradients(15);
    final double[] bias = context.testingOutputBias();
    for (int o = 0; o < bias.length; o++) {
      assertClose(analytic[0][o], numerical(bias, o), "outputBias[" + o + "]");
    }
  }

  @Test
  void testWordEmbeddingGradients() {
    final double[][] analytic = analyticGradients(0);
    final double[][] embeddings = context.testingWordEmbeddings();
    // every training word of the sample is in the vocabulary (cutoff 1)
    for (int row = 0; row < embeddings.length; row++) {
      for (int i = 0; i < embeddings[row].length; i++) {
        assertClose(analytic[row][i], numerical(embeddings[row], i),
            "wordEmbeddings[" + row + "][" + i + "]");
      }
    }
  }

  @Test
  void testCharEmbeddingGradients() {
    final double[][] analytic = analyticGradients(1);
    final double[][] embeddings = context.testingCharEmbeddings();
    for (int row = 0; row < embeddings.length; row++) {
      for (int i = 0; i < embeddings[row].length; i++) {
        assertClose(analytic[row][i], numerical(embeddings[row], i),
            "charEmbeddings[" + row + "][" + i + "]");
      }
    }
  }

  @Test
  void testWordLstmGradients() {
    final LstmLayer layer = context.testingWordForward();
    final double[][] analyticW = analyticGradients(8);
    final double[][] analyticU = analyticGradients(9);
    final double[][] analyticB = analyticGradients(10);
    for (int r = 0; r < 4 * layer.hiddenSize(); r++) {
      for (int k = 0; k < layer.inputSize(); k++) {
        assertClose(analyticW[r][k], numerical(layer.w()[r], k),
            "wordForward.w[" + r + "][" + k + "]");
      }
      for (int k = 0; k < layer.hiddenSize(); k++) {
        assertClose(analyticU[r][k], numerical(layer.u()[r], k),
            "wordForward.u[" + r + "][" + k + "]");
      }
      assertClose(analyticB[0][r], numerical(layer.b(), r), "wordForward.b[" + r + "]");
    }
  }

  private double[][] analyticGradients(int index) {
    AdamOptimizer.zero(worker.buffers());
    context.sentenceGradients(sample, new Random(0L), worker);
    final double[][] source = worker.buffers().get(index);
    final double[][] copy = new double[source.length][];
    for (int r = 0; r < source.length; r++) {
      copy[r] = source[r].clone();
    }
    return copy;
  }

  private double numerical(double[] row, int i) {
    final double original = row[i];
    row[i] = original + EPSILON;
    final double plus = context.sentenceGradients(sample, new Random(0L), worker);
    row[i] = original - EPSILON;
    final double minus = context.sentenceGradients(sample, new Random(0L), worker);
    row[i] = original;
    return (plus - minus) / (2.0d * EPSILON);
  }

  private static void assertClose(double analytic, double numerical, String what) {
    final double scale = Math.max(1e-3d, Math.abs(analytic) + Math.abs(numerical));
    assertTrue(Math.abs(analytic - numerical) / scale < TOLERANCE,
        () -> what + " analytic " + analytic + " vs numerical " + numerical);
  }
}
