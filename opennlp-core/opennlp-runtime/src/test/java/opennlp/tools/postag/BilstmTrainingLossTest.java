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

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** Checks softmax losses and gradients for all BiLSTM tagging heads. */
class BilstmTrainingLossTest {

  private enum Head {
    UPOS, XPOS, FEATS
  }

  /** {@return finite scores and their analytically known loss and gold-label gradient} */
  private static Stream<Arguments> finiteScores() {
    return Stream.of(Head.values()).flatMap(head -> Stream.of(
        Arguments.of(head, 0.0, 0.0, Math.log(2), -0.5),
        Arguments.of(head, -1000.0, 0.0, 1000.0, -1.0),
        Arguments.of(head, -1e300, 1e300, 2e300, -1.0),
        Arguments.of(head, 1e300, 1e300, Math.log(2), -0.5),
        Arguments.of(head, -1e300, -1e300, Math.log(2), -0.5),
        Arguments.of(head, 1000.0, 0.0, 0.0, 0.0)));
  }

  /**
   * A rounded-to-zero probability still has a finite log loss when the score gap is finite.
   *
   * @param head The head to exercise.
   * @param gold The correct label's score.
   * @param other The other label's score.
   * @param headLoss The selected head's loss.
   * @param gradient The selected head's correct-label gradient.
   * @throws ReflectiveOperationException If the auxiliary bias cannot be accessed.
   */
  @ParameterizedTest
  @MethodSource("finiteScores")
  void testFiniteLoss(Head head, double gold, double other, double headLoss,
      double gradient) throws ReflectiveOperationException {
    final Fixture fixture = fixture();
    fixture.biases()[head.ordinal()][0] = gold;
    fixture.biases()[head.ordinal()][1] = other;
    final double expected = headLoss + 2 * Math.log(2);
    assertEquals(expected, fixture.context().sentenceGradients(fixture.sample(),
        new Random(7), fixture.worker()), 4 * Math.ulp(expected));
    final double[] actual = fixture.worker().buffers().get(15 + 2 * head.ordinal())[0];
    assertArrayEquals(new double[] {gradient, -gradient}, actual, 0.0);
  }

  /** {@return non-finite scores in every label position and head} */
  private static Stream<Arguments> invalidScores() {
    return Stream.of(Head.values()).flatMap(head ->
        Stream.of(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY)
            .flatMap(value -> Stream.of(0, 1).map(label -> Arguments.of(head, label, value))));
  }

  /**
   * Non-finite scores stop the sentence before invalid loss can enter the epoch total.
   *
   * @param head The head to exercise.
   * @param label The score position.
   * @param value The invalid score.
   * @throws ReflectiveOperationException If the auxiliary bias cannot be accessed.
   */
  @ParameterizedTest
  @MethodSource("invalidScores")
  void testInvalidScore(Head head, int label, double value) throws ReflectiveOperationException {
    final Fixture fixture = fixture();
    fixture.biases()[head.ordinal()][label] = value;
    assertThrows(IllegalStateException.class, () -> fixture.context().sentenceGradients(
        fixture.sample(), new Random(7), fixture.worker()));
  }

  /**
   * An unrepresentable loss fails instead of reporting infinity as an epoch result.
   *
   * @param head The head to exercise.
   * @throws ReflectiveOperationException If the auxiliary bias cannot be accessed.
   */
  @ParameterizedTest
  @EnumSource(Head.class)
  void testLossOverflow(Head head) throws ReflectiveOperationException {
    final Fixture fixture = fixture();
    fixture.biases()[head.ordinal()][0] = -Double.MAX_VALUE;
    fixture.biases()[head.ordinal()][1] = Double.MAX_VALUE;
    assertThrows(IllegalStateException.class, () -> fixture.context().sentenceGradients(
        fixture.sample(), new Random(7), fixture.worker()));
  }

  /**
   * Finite token losses can overflow when summed for a sentence.
   *
   * @param head The head to exercise.
   * @throws ReflectiveOperationException If the auxiliary bias cannot be accessed.
   */
  @ParameterizedTest
  @EnumSource(Head.class)
  void testSentenceLossOverflow(Head head) throws ReflectiveOperationException {
    final Fixture fixture = fixture();
    fixture.biases()[head.ordinal()][0] = -1e308;
    final BilstmPOSTrainer.MultiTaskSample sample = new BilstmPOSTrainer.MultiTaskSample(
        new String[] {"one", "one"}, new String[] {"X", "X"},
        new String[] {"X0", "X0"}, new String[] {"F0", "F0"});
    assertThrows(IllegalStateException.class, () -> fixture.context().sentenceGradients(
        sample, new Random(7), fixture.worker()));
  }

  /**
   * Creates a single-token example with two labels per head and zero scoring weights.
   *
   * @return The training context and mutable head biases.
   * @throws ReflectiveOperationException If the auxiliary bias cannot be accessed.
   */
  private Fixture fixture() throws ReflectiveOperationException {
    final List<BilstmPOSTrainer.MultiTaskSample> corpus = List.of(
        new BilstmPOSTrainer.MultiTaskSample(new String[] {"one"},
            new String[] {"X"}, new String[] {"X0"}, new String[] {"F0"}),
        new BilstmPOSTrainer.MultiTaskSample(new String[] {"two"},
            new String[] {"Y"}, new String[] {"X1"}, new String[] {"F1"}));
    final BilstmPOSTrainer.Settings settings = new BilstmPOSTrainer.Settings(
        4, 3, 3, 4, 1, 2, 1e-3, 5, 0, 1, 10, 7L, 1, 0, 0, false, 1,
        0, 0, 1, 0, false);
    final BilstmPOSTrainer.TrainingContext context =
        BilstmPOSTrainer.TrainingContext.build(corpus, settings, null, null);
    for (final double[][] weights : List.of(context.testingOutputWeights(),
        context.testingXposWeights(), context.testingFeatsWeights())) {
      for (final double[] row : weights) {
        Arrays.fill(row, 0.0);
      }
    }
    final Field field = context.getClass().getDeclaredField("featsBias");
    field.setAccessible(true);
    final double[][] biases = {context.testingOutputBias(), context.testingXposBias(),
        (double[]) field.get(context)};
    return new Fixture(context, context.newWorker(), corpus.get(0), biases);
  }

  /** Holds a context whose output scores depend only on the supplied biases. */
  private record Fixture(BilstmPOSTrainer.TrainingContext context,
      BilstmPOSTrainer.TrainingContext.Worker worker,
      BilstmPOSTrainer.MultiTaskSample sample, double[][] biases) {
  }
}
