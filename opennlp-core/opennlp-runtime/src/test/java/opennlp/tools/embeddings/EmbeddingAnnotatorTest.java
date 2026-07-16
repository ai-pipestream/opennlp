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

package opennlp.tools.embeddings;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.Layers;
import opennlp.tools.util.Span;

public class EmbeddingAnnotatorTest {

  /** Embeds text as its length and first character code, enough to observe alignment. */
  private static final TextEmbedder FIXTURE = new TextEmbedder() {
    @Override
    public float[] embed(CharSequence text) {
      return new float[] {text.length(), text.isEmpty() ? 0 : text.charAt(0)};
    }

    @Override
    public int dimension() {
      return 2;
    }
  };

  @Test
  void testTokenAndSentenceEmbeddingsCoexist() {
    final String text = "Dogs bark. Cats nap.";
    final Document base = Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 10), "s"),
            new Annotation<>(new Span(11, 20), "s")))
        .with(Layers.TOKENS, List.of(
            new Annotation<>(new Span(0, 4), "Dogs"),
            new Annotation<>(new Span(5, 9), "bark"),
            new Annotation<>(new Span(9, 10), "."),
            new Annotation<>(new Span(11, 15), "Cats"),
            new Annotation<>(new Span(16, 19), "nap"),
            new Annotation<>(new Span(19, 20), ".")));

    final EmbeddingAnnotator overTokens = new EmbeddingAnnotator(FIXTURE, Layers.TOKENS);
    final EmbeddingAnnotator overSentences =
        new EmbeddingAnnotator(FIXTURE, Layers.SENTENCES);
    final Document document = overSentences.annotate(overTokens.annotate(base));

    final List<Annotation<float[]>> tokens = document.get(overTokens.layer());
    final List<Annotation<float[]>> sentences = document.get(overSentences.layer());
    Assertions.assertEquals(6, tokens.size());
    Assertions.assertEquals(2, sentences.size());
    Assertions.assertEquals(4.0f, tokens.get(0).value()[0]);
    Assertions.assertEquals('D', (int) tokens.get(0).value()[1]);
    Assertions.assertEquals(10.0f, sentences.get(0).value()[0]);
    Assertions.assertEquals(new Span(0, 4), tokens.get(0).span());
    Assertions.assertNotEquals(overTokens.layer(), overSentences.layer());
  }

  @Test
  void testInvalidArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new EmbeddingAnnotator(null, Layers.TOKENS));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new EmbeddingAnnotator(FIXTURE, null));
    final EmbeddingAnnotator annotator = new EmbeddingAnnotator(FIXTURE, Layers.TOKENS);
    Assertions.assertThrows(IllegalArgumentException.class, () -> annotator.annotate(null));
  }

  /**
   * Asserts that a document whose source layer is present but empty still receives the
   * provided layer, carrying no annotations.
   */
  @Test
  void testEmptySourceLayerYieldsEmptyProvidedLayer() {
    final Document base = Document.of("no tokens were found").with(Layers.TOKENS, List.of());
    final EmbeddingAnnotator annotator = new EmbeddingAnnotator(FIXTURE, Layers.TOKENS);
    final Document document = annotator.annotate(base);
    Assertions.assertTrue(document.layers().contains(annotator.layer()));
    Assertions.assertTrue(document.get(annotator.layer()).isEmpty());
  }

  /**
   * Asserts that a document without the source layer is treated like one with an empty
   * source layer: annotation succeeds and the provided layer is present and empty,
   * because the document container reads an absent layer as an empty list.
   */
  @Test
  void testAbsentSourceLayerYieldsEmptyProvidedLayer() {
    final EmbeddingAnnotator annotator = new EmbeddingAnnotator(FIXTURE, Layers.TOKENS);
    final Document document = annotator.annotate(Document.of("never tokenized"));
    Assertions.assertTrue(document.layers().contains(annotator.layer()));
    Assertions.assertTrue(document.get(annotator.layer()).isEmpty());
  }

  /**
   * Asserts that vectors of differing lengths are stored exactly as the embedder
   * returned them: the annotator applies no dimension check, so an embedder whose
   * output length varies per input produces a layer with mixed vector lengths.
   */
  @Test
  void testInconsistentVectorDimensionsAreStoredAsReturned() {
    final Document base = Document.of("ab cdef").with(Layers.TOKENS, List.of(
        new Annotation<>(new Span(0, 2), "ab"),
        new Annotation<>(new Span(3, 7), "cdef")));
    final EmbeddingAnnotator annotator =
        new EmbeddingAnnotator(new TextLengthEmbedder(), Layers.TOKENS);
    final List<Annotation<float[]>> vectors = annotator.annotate(base).get(annotator.layer());
    Assertions.assertEquals(2, vectors.size());
    Assertions.assertArrayEquals(new float[] {1, 1}, vectors.get(0).value());
    Assertions.assertArrayEquals(new float[] {1, 1, 1, 1}, vectors.get(1).value());
  }

  /**
   * Asserts that an embedder returning {@code null} fails the annotation loudly with an
   * {@link IllegalArgumentException}, raised when the {@code null} vector is rejected as
   * an annotation value, instead of storing a {@code null} silently.
   */
  @Test
  void testNullVectorFromEmbedderFailsLoud() {
    final Document base = Document.of("boom").with(Layers.TOKENS,
        List.of(new Annotation<>(new Span(0, 4), "boom")));
    final EmbeddingAnnotator annotator =
        new EmbeddingAnnotator(new NullVectorEmbedder(), Layers.TOKENS);
    final IllegalArgumentException e = Assertions.assertThrows(
        IllegalArgumentException.class, () -> annotator.annotate(base));
    Assertions.assertEquals("value must not be null", e.getMessage());
  }

  /**
   * Asserts that annotating a document that already carries the provided layer is
   * rejected: the immutable document refuses duplicate layers, so running the same
   * annotator twice over one document chain fails with an
   * {@link IllegalArgumentException} naming the conflicting layer.
   */
  @Test
  void testAnnotatingSameDocumentTwiceIsRejected() {
    final Document base = Document.of("once").with(Layers.TOKENS,
        List.of(new Annotation<>(new Span(0, 4), "once")));
    final EmbeddingAnnotator annotator = new EmbeddingAnnotator(FIXTURE, Layers.TOKENS);
    final Document annotated = annotator.annotate(base);
    final IllegalArgumentException e = Assertions.assertThrows(
        IllegalArgumentException.class, () -> annotator.annotate(annotated));
    Assertions.assertTrue(e.getMessage().contains("already present"));
    Assertions.assertTrue(e.getMessage().contains("embeddings:tokens"));
  }

  /**
   * Verifies the concurrency claim: one annotator instance with a stateless embedder is
   * driven from several threads over two different documents at once, and every result
   * must be identical to the sequentially computed reference, span by span and vector
   * component by vector component.
   *
   * @throws Exception Propagated if a worker fails or does not finish in time; a
   *         propagated {@link java.util.concurrent.ExecutionException} carries the
   *         worker's assertion failure and fails the test.
   */
  @Test
  void testConcurrentAnnotationYieldsIdenticalResults() throws Exception {
    final Document docA = Document.of("Dogs bark.").with(Layers.TOKENS, List.of(
        new Annotation<>(new Span(0, 4), "Dogs"),
        new Annotation<>(new Span(5, 9), "bark"),
        new Annotation<>(new Span(9, 10), ".")));
    final Document docB = Document.of("Cats nap.").with(Layers.TOKENS, List.of(
        new Annotation<>(new Span(0, 4), "Cats"),
        new Annotation<>(new Span(5, 8), "nap"),
        new Annotation<>(new Span(8, 9), ".")));
    final EmbeddingAnnotator annotator = new EmbeddingAnnotator(FIXTURE, Layers.TOKENS);
    final List<Annotation<float[]>> expectedA = annotator.annotate(docA).get(annotator.layer());
    final List<Annotation<float[]>> expectedB = annotator.annotate(docB).get(annotator.layer());

    final int threads = 8;
    final ExecutorService pool = Executors.newFixedThreadPool(threads);
    try {
      final CountDownLatch start = new CountDownLatch(1);
      final List<Future<?>> workers = new ArrayList<>(threads);
      for (int t = 0; t < threads; t++) {
        final Document input = t % 2 == 0 ? docA : docB;
        final List<Annotation<float[]>> expected = t % 2 == 0 ? expectedA : expectedB;
        workers.add(pool.submit(() -> {
          start.await();
          for (int round = 0; round < 50; round++) {
            final List<Annotation<float[]>> actual =
                annotator.annotate(input).get(annotator.layer());
            Assertions.assertEquals(expected.size(), actual.size());
            for (int i = 0; i < expected.size(); i++) {
              Assertions.assertEquals(expected.get(i).span(), actual.get(i).span());
              Assertions.assertArrayEquals(expected.get(i).value(), actual.get(i).value());
            }
          }
          return null;
        }));
      }
      start.countDown();
      for (final Future<?> worker : workers) {
        worker.get(30, TimeUnit.SECONDS);
      }
    } finally {
      pool.shutdownNow();
    }
  }

  /**
   * A misbehaving {@link TextEmbedder} whose vector length equals the input text length
   * instead of the two dimensions it declares, with every component set to one. It
   * exists to observe how the annotator treats vectors of inconsistent dimensions.
   */
  private static final class TextLengthEmbedder implements TextEmbedder {

    /**
     * Embeds a text as a vector of ones whose length equals the text length.
     *
     * @param text The text to embed. Must not be {@code null}.
     * @return A vector of length {@code text.length()} with every component one.
     * @throws IllegalArgumentException Thrown if {@code text} is {@code null}.
     */
    @Override
    public float[] embed(CharSequence text) {
      if (text == null) {
        throw new IllegalArgumentException("text must not be null");
      }
      final float[] vector = new float[text.length()];
      for (int i = 0; i < vector.length; i++) {
        vector[i] = 1;
      }
      return vector;
    }

    /**
     * @return The declared dimension of two, which {@link #embed(CharSequence)}
     *         deliberately does not honor.
     */
    @Override
    public int dimension() {
      return 2;
    }
  }

  /**
   * A broken {@link TextEmbedder} that returns {@code null} for every input. It exists
   * to observe how the annotator reacts to an embedder violating its contract.
   */
  private static final class NullVectorEmbedder implements TextEmbedder {

    /**
     * Violates the embedder contract on purpose.
     *
     * @param text The text to embed; ignored.
     * @return Always {@code null}.
     */
    @Override
    public float[] embed(CharSequence text) {
      return null;
    }

    /**
     * @return The declared dimension of two, never observable through
     *         {@link #embed(CharSequence)}.
     */
    @Override
    public int dimension() {
      return 2;
    }
  }
}
