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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.jupiter.api.Test;

import opennlp.tools.util.CollectionObjectStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Pins {@link BilstmPOSModel} serialization and the thread-safety of tagging.
 */
class BilstmPOSModelTest {

  private static final List<POSSample> CORPUS = List.of(
      new POSSample(new String[] {"The", "cat", "sat"}, new String[] {"D", "N", "V"}),
      new POSSample(new String[] {"A", "dog", "ran"}, new String[] {"D", "N", "V"}));

  private static BilstmPOSModel tinyModel() throws IOException {
    return BilstmPOSTrainer.train(new CollectionObjectStream<>(CORPUS),
        new BilstmPOSTrainer.Settings(8, 4, 4, 8, 3, 2, 5e-3d, 5.0d, 0.1d, 1, 12, 7L, 2, 0.0d));
  }

  @Test
  void testSerializationRoundTripPreservesScores() throws IOException {
    final BilstmPOSModel model = tinyModel();
    final ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    model.serialize(buffer);
    final BilstmPOSModel loaded =
        BilstmPOSModel.load(new ByteArrayInputStream(buffer.toByteArray()));
    assertArrayEquals(model.tags(), loaded.tags());
    final String[] sentence = {"The", "dog", "sat"};
    final double[][] expected = model.score(sentence);
    final double[][] actual = loaded.score(sentence);
    assertEquals(expected.length, actual.length);
    for (int t = 0; t < expected.length; t++) {
      assertArrayEquals(expected[t], actual[t], 0.0d);
    }
  }

  @Test
  void testSerializationRoundTripPreservesPretrainedTable() throws IOException {
    final BilstmPOSModel model = BilstmPOSTrainer.train(
        new CollectionObjectStream<>(CORPUS),
        new BilstmPOSTrainer.Settings(8, 4, 4, 8, 3, 2, 5e-3d, 5.0d, 0.1d, 1, 12, 7L, 2, 0.0d),
        w -> new float[] {w.length(), 1.0f}, List.of("unseen"));
    final ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    model.serialize(buffer);
    final BilstmPOSModel loaded =
        BilstmPOSModel.load(new ByteArrayInputStream(buffer.toByteArray()));
    assertArrayEquals(model.wordRepresentation("unseen"),
        loaded.wordRepresentation("unseen"), 0.0d);
  }

  @Test
  void testRejectsForeignMagic() {
    final byte[] junk = "ONLP-FFPT-2\0\0\0\0".getBytes();
    assertThrows(IOException.class,
        () -> BilstmPOSModel.load(new ByteArrayInputStream(junk)));
  }

  @Test
  void testRejectsNullAndEmpty() {
    final BilstmPOSTagger tagger;
    try {
      tagger = new BilstmPOSTagger(tinyModel());
    }
    catch (IOException e) {
      throw new AssertionError(e);
    }
    assertThrows(IllegalArgumentException.class, () -> tagger.tag(null));
    assertEquals(0, tagger.tag(new String[0]).length);
  }

  @Test
  void testSharedTaggerIsThreadSafe()
      throws IOException, InterruptedException, ExecutionException {
    final BilstmPOSTagger tagger = new BilstmPOSTagger(tinyModel());
    final String[][] sentences = {
        {"The", "cat", "sat"}, {"A", "dog", "ran"}, {"The", "dog", "sat", "quickly"}};
    final String[][] expected = new String[sentences.length][];
    for (int i = 0; i < sentences.length; i++) {
      expected[i] = tagger.tag(sentences[i]);
    }
    final ExecutorService pool = Executors.newFixedThreadPool(8);
    try {
      final List<Callable<Boolean>> tasks = new ArrayList<>();
      for (int i = 0; i < sentences.length; i++) {
        final int index = i;
        tasks.add(() -> {
          for (int iteration = 0; iteration < 200; iteration++) {
            final String[] tagged = tagger.tag(sentences[index]);
            for (int t = 0; t < tagged.length; t++) {
              if (!tagged[t].equals(expected[index][t])) {
                return false;
              }
            }
          }
          return true;
        });
      }
      final List<Future<Boolean>> results = pool.invokeAll(tasks);
      for (final Future<Boolean> result : results) {
        assertEquals(Boolean.TRUE, result.get());
      }
    }
    finally {
      pool.shutdownNow();
    }
  }
}
