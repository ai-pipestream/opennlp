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

import java.io.IOException;
import java.util.List;
import java.util.function.Function;

import org.junit.jupiter.api.Test;

import opennlp.tools.util.CollectionObjectStream;
import opennlp.tools.util.ObjectStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Pins the {@link BilstmPOSTrainer} contract: fail-loud validation, deterministic
 * training under one seed, and actually learning a separable pattern end to end.
 */
class BilstmPOSTrainerTest {

  private static final List<POSSample> CORPUS = List.of(
      new POSSample(new String[] {"The", "cat", "sat"}, new String[] {"D", "N", "V"}),
      new POSSample(new String[] {"A", "dog", "ran"}, new String[] {"D", "N", "V"}),
      new POSSample(new String[] {"The", "bird", "flew"}, new String[] {"D", "N", "V"}),
      new POSSample(new String[] {"A", "fish", "swam"}, new String[] {"D", "N", "V"}));

  private static final BilstmPOSTrainer.Settings TINY = new BilstmPOSTrainer.Settings(
      8, 4, 4, 8, 40, 2, 5e-3d, 5.0d, 0.1d, 1, 12, 7L, 1, 0.0d);

  private static ObjectStream<POSSample> stream(List<POSSample> samples) {
    return new CollectionObjectStream<>(samples);
  }

  @Test
  void testRejectsNullArguments() {
    assertThrows(IllegalArgumentException.class, () -> BilstmPOSTrainer.train(null, TINY));
    assertThrows(IllegalArgumentException.class,
        () -> BilstmPOSTrainer.train(stream(CORPUS), null));
    assertThrows(IllegalArgumentException.class,
        () -> BilstmPOSTrainer.train(stream(CORPUS), TINY, (Function<CharSequence, float[]>) null));
    assertThrows(IllegalArgumentException.class,
        () -> BilstmPOSTrainer.train(stream(CORPUS), TINY, w -> null, null));
  }

  @Test
  void testRejectsEmptyCorpus() {
    assertThrows(IllegalArgumentException.class,
        () -> BilstmPOSTrainer.train(stream(List.of()), TINY));
  }

  @Test
  void testRejectsBrokenVectorContracts() {
    final Function<CharSequence, float[]> inconsistent = w -> {
      final String word = w.toString();
      return new float["the".equals(BilstmPOSModel.normalize(word)) ? 3 : 2];
    };
    assertThrows(IllegalArgumentException.class,
        () -> BilstmPOSTrainer.train(stream(CORPUS), TINY, inconsistent));
    assertThrows(IllegalArgumentException.class,
        () -> BilstmPOSTrainer.train(stream(CORPUS), TINY, w -> null));
    final List<String> withNull = new java.util.ArrayList<>();
    withNull.add("cat");
    withNull.add(null);
    assertThrows(IllegalArgumentException.class,
        () -> BilstmPOSTrainer.train(stream(CORPUS), TINY, w -> new float[2], withNull));
  }

  @Test
  void testLearnsSeparablePattern() throws IOException {
    final BilstmPOSModel model = BilstmPOSTrainer.train(stream(CORPUS), TINY);
    final BilstmPOSTagger tagger = new BilstmPOSTagger(model);
    for (final POSSample sample : CORPUS) {
      assertArrayEquals(sample.getTags(), tagger.tag(sample.getSentence()));
    }
  }

  @Test
  void testSameSeedTrainsIdenticalModel() throws IOException {
    final BilstmPOSModel first = BilstmPOSTrainer.train(stream(CORPUS), TINY);
    final BilstmPOSModel second = BilstmPOSTrainer.train(stream(CORPUS), TINY);
    final String[] sentence = {"The", "cat", "ran"};
    final double[][] firstScores = first.score(sentence);
    final double[][] secondScores = second.score(sentence);
    for (int t = 0; t < sentence.length; t++) {
      assertArrayEquals(firstScores[t], secondScores[t], 0.0d);
    }
  }

  @Test
  void testParallelTrainingMatchesSequential() throws IOException {
    // strided batch assignment plus ordered reduction gives the same per-element
    // accumulation order for any thread count; the only wobble is last-ulp noise
    // from the JIT contracting multiply-adds differently between compilation paths,
    // so the comparison is tight but not bit-exact
    final BilstmPOSModel sequential = BilstmPOSTrainer.train(stream(CORPUS), TINY);
    final BilstmPOSModel parallel = BilstmPOSTrainer.train(stream(CORPUS),
        new BilstmPOSTrainer.Settings(8, 4, 4, 8, 40, 2, 5e-3d, 5.0d, 0.1d, 1, 12, 7L,
            3, 0.0d));
    final String[] sentence = {"The", "cat", "ran"};
    final double[][] expected = sequential.score(sentence);
    final double[][] actual = parallel.score(sentence);
    for (int t = 0; t < sentence.length; t++) {
      assertArrayEquals(expected[t], actual[t], 1e-6d);
    }
  }

  @Test
  void testTrainsWithPretrainedTable() throws IOException {
    final Function<CharSequence, float[]> vectors =
        w -> new float[] {w.length(), w.charAt(0)};
    final BilstmPOSModel model =
        BilstmPOSTrainer.train(stream(CORPUS), TINY, vectors, List.of("unseen"));
    // a lexicon-only word resolves through the table at tagging time
    final String[] tags = new BilstmPOSTagger(model).tag(new String[] {"unseen"});
    assertEquals(1, tags.length);
  }
}
