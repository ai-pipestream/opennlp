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

package opennlp.tools.depparse;

import java.io.IOException;
import java.util.List;
import java.util.function.Predicate;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.tools.ml.perceptron.SimplePerceptronSequenceTrainer;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Parameters;
import opennlp.tools.util.TrainingParameters;

import static opennlp.tools.depparse.DependencyTestSamples.CORPUS_SENTENCES;
import static opennlp.tools.depparse.DependencyTestSamples.CORPUS_WORDS;
import static opennlp.tools.depparse.DependencyTestSamples.LANGUAGE;
import static opennlp.tools.depparse.DependencyTestSamples.corpus;
import static opennlp.tools.depparse.DependencyTestSamples.trainingParameters;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests {@link DependencyCrossValidator}: the tokens it counts, its punctuation handling,
 * its argument checks, and its agreement with a single {@link DependencyEvaluator} on a
 * corpus every fold can memorize.
 */
public class DependencyCrossValidatorTest {

  /** The number of determiner tokens in {@link DependencyTestSamples#corpus()}. */
  private static final int CORPUS_DETERMINERS = CORPUS_SENTENCES / 3;

  /**
   * @return A validator over the test corpus language with a zero feature cutoff.
   */
  private static DependencyCrossValidator validator() {
    return new DependencyCrossValidator(LANGUAGE, trainingParameters());
  }

  @ParameterizedTest(name = "folds = {0}")
  @ValueSource(ints = {2, 3, 5, 8})
  void testCountsEveryTokenOnce(int folds) throws IOException {
    final DependencyCrossValidator validator = validator();
    validator.evaluate(ObjectStreamUtils.createObjectStream(corpus()), folds);
    assertEquals(CORPUS_WORDS, validator.getWordCount());
    assertEquals(CORPUS_WORDS, validator.getWordCountExcludingPunctuation(),
        "the corpus has no punctuation, so no token is left out");
  }

  @Test
  void testAgreesWithSingleEvaluatorOnMemorizableCorpus() throws IOException {
    final DependencyCrossValidator validator = validator();
    validator.evaluate(ObjectStreamUtils.createObjectStream(corpus()), 4);

    final DependencyEvaluator evaluator =
        new DependencyEvaluator(new DependencyParserME(DependencyTestSamples.train(corpus())));
    evaluator.evaluate(ObjectStreamUtils.createObjectStream(corpus()));

    assertEquals(evaluator.getWordCount(), validator.getWordCount());
    assertEquals(evaluator.getUas(), validator.getUas());
    assertEquals(evaluator.getLas(), validator.getLas());
    assertEquals(evaluator.getUasExcludingPunctuation(),
        validator.getUasExcludingPunctuation());
    assertEquals(evaluator.getLasExcludingPunctuation(),
        validator.getLasExcludingPunctuation());
    assertEquals(1.0d, validator.getUas());
    assertEquals(1.0d, validator.getLas());
  }

  @Test
  void testAccumulatesAcrossRuns() throws IOException {
    final DependencyCrossValidator validator = validator();
    validator.evaluate(ObjectStreamUtils.createObjectStream(corpus()), 2);
    validator.evaluate(ObjectStreamUtils.createObjectStream(corpus()), 2);
    assertEquals(2 * CORPUS_WORDS, validator.getWordCount());
  }

  @Test
  void testLeavesUniversalPunctuationOutByDefault() throws IOException {
    final DependencySample punctuated = DependencyTestSamples.sample(
        new String[] {"dogs", "bark", "."}, new String[] {"NOUN", "VERB", "PUNCT"},
        new int[] {1, -1, 1}, new String[] {"nsubj", "root", "punct"});
    final List<DependencySample> samples = DependencyTestSamples.repeat(List.of(punctuated));
    final DependencyCrossValidator validator = validator();
    validator.evaluate(ObjectStreamUtils.createObjectStream(samples), 2);
    assertEquals(3L * samples.size(), validator.getWordCount());
    assertEquals(2L * samples.size(), validator.getWordCountExcludingPunctuation());
    assertEquals(1.0d, validator.getUasExcludingPunctuation());
    assertEquals(1.0d, validator.getLasExcludingPunctuation());
  }

  @Test
  void testCustomPunctuationPredicate() throws IOException {
    final DependencyCrossValidator validator =
        new DependencyCrossValidator(LANGUAGE, trainingParameters(), "DT"::equals);
    validator.evaluate(ObjectStreamUtils.createObjectStream(corpus()), 2);
    assertEquals(CORPUS_WORDS, validator.getWordCount());
    assertEquals(CORPUS_WORDS - CORPUS_DETERMINERS,
        validator.getWordCountExcludingPunctuation());
  }

  @Test
  void testListenersHearEveryHeldOutSample() throws IOException {
    final RecordingMonitor monitor = new RecordingMonitor();
    final DependencyCrossValidator validator =
        new DependencyCrossValidator(LANGUAGE, trainingParameters(), monitor);
    validator.evaluate(ObjectStreamUtils.createObjectStream(corpus()), 4);
    assertEquals(CORPUS_SENTENCES, monitor.correct.size(),
        "each sentence is held out once and the corpus is memorizable");
    assertEquals(List.of(), monitor.wrong);
  }

  @Test
  void testListenersWithCustomPunctuation() throws IOException {
    final RecordingMonitor monitor = new RecordingMonitor();
    final DependencyCrossValidator validator =
        new DependencyCrossValidator(LANGUAGE, trainingParameters(), "DT"::equals, monitor);
    validator.evaluate(ObjectStreamUtils.createObjectStream(corpus()), 2);
    assertEquals(CORPUS_SENTENCES, monitor.correct.size() + monitor.wrong.size());
    assertEquals(CORPUS_WORDS - CORPUS_DETERMINERS,
        validator.getWordCountExcludingPunctuation());
  }

  @Test
  void testEmptyValidatorScoresZero() {
    final DependencyCrossValidator validator = validator();
    assertEquals(0, validator.getWordCount());
    assertEquals(0.0d, validator.getUas());
    assertEquals(0.0d, validator.getLas());
  }

  @ParameterizedTest(name = "folds = {0}")
  @ValueSource(ints = {Integer.MIN_VALUE, -1, 0, 1})
  void testRejectsFoldCountBelowTwo(int folds) {
    final DependencyCrossValidator validator = validator();
    assertThrows(IllegalArgumentException.class,
        () -> validator.evaluate(ObjectStreamUtils.createObjectStream(corpus()), folds));
    assertEquals(0, validator.getWordCount());
  }

  @Test
  void testRejectsNullArguments() {
    final TrainingParameters parameters = trainingParameters();
    assertThrows(IllegalArgumentException.class,
        () -> new DependencyCrossValidator(null, parameters));
    assertThrows(IllegalArgumentException.class,
        () -> new DependencyCrossValidator(LANGUAGE, null));
    assertThrows(IllegalArgumentException.class,
        () -> new DependencyCrossValidator(LANGUAGE, parameters, (Predicate<String>) null));
    assertThrows(IllegalArgumentException.class, () -> validator().evaluate(null, 2));
  }

  @Test
  void testRejectsSequenceTrainer() {
    final TrainingParameters parameters = trainingParameters();
    parameters.put(Parameters.ALGORITHM_PARAM,
        SimplePerceptronSequenceTrainer.PERCEPTRON_SEQUENCE_VALUE);
    final DependencyCrossValidator validator =
        new DependencyCrossValidator(LANGUAGE, parameters);
    assertThrows(IllegalArgumentException.class,
        () -> validator.evaluate(ObjectStreamUtils.createObjectStream(corpus()), 2));
  }
}
