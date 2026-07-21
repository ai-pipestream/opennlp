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
import java.util.Set;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;
import opennlp.tools.util.Span;

/**
 * Demonstrates the realistic {@link EmbeddingAnnotator} flow with fully deterministic
 * collaborators: a stub tokenizer annotator produces the token layer, a stub embedder
 * turns each covered text into a fixed three-dimensional vector, and the tests assert
 * the exact vectors, spans, and derived layer identifiers that come out of the pipeline.
 */
public class EmbeddingAnnotatorUsageTest {

  /**
   * A deterministic {@link TextEmbedder} that folds the character codes of a text into a
   * fixed three-dimensional vector: the code of the character at index {@code i} is
   * added to vector component {@code i % 3}. Equal texts always map to equal vectors,
   * which makes every expected value in this class computable by hand.
   */
  private static final class CharSumEmbedder implements TextEmbedder {

    /**
     * Embeds a text by summing its character codes into three components.
     *
     * @param text The text to embed. Must not be {@code null}.
     * @return A vector of length three; the zero vector for empty text.
     * @throws IllegalArgumentException Thrown if {@code text} is {@code null}.
     */
    @Override
    public float[] embed(CharSequence text) {
      if (text == null) {
        throw new IllegalArgumentException("text must not be null");
      }
      final float[] vector = new float[3];
      for (int i = 0; i < text.length(); i++) {
        vector[i % 3] += text.charAt(i);
      }
      return vector;
    }

    /**
     * @return The constant vector length three.
     */
    @Override
    public int dimension() {
      return 3;
    }
  }

  /**
   * A stub {@link DocumentAnnotator} that provides {@link Layers#TOKENS} by splitting
   * the document text on the literal space character. Each token annotation covers one
   * maximal run of non-space characters and carries its covered text as the value. This
   * keeps the pipeline example self-contained without loading a tokenizer model.
   */
  private static final class SpaceTokenizerAnnotator implements DocumentAnnotator {

    /**
     * Tokenizes the document text on spaces and adds the token layer.
     *
     * @param document The document to annotate. Must not be {@code null}.
     * @return A new {@link Document} carrying {@link Layers#TOKENS} in addition to the
     *         input layers. Never {@code null}.
     * @throws IllegalArgumentException Thrown if {@code document} is {@code null}.
     */
    @Override
    public Document annotate(Document document) {
      if (document == null) {
        throw new IllegalArgumentException("document must not be null");
      }
      final CharSequence text = document.text();
      final List<Annotation<String>> tokens = new ArrayList<>();
      int start = -1;
      for (int i = 0; i <= text.length(); i++) {
        final boolean boundary = i == text.length() || text.charAt(i) == ' ';
        if (boundary && start >= 0) {
          tokens.add(new Annotation<>(new Span(start, i), text.subSequence(start, i).toString()));
          start = -1;
        } else if (!boundary && start < 0) {
          start = i;
        }
      }
      return document.with(Layers.TOKENS, tokens);
    }

    /**
     * @return The single-element set holding the token layer. Never {@code null}.
     */
    @Override
    public Set<LayerKey<?>> provides() {
      return Set.of(Layers.TOKENS);
    }
  }

  /**
   * Runs the two-step pipeline of tokenizing and then embedding the token layer, and
   * asserts the exact vector of every token, that every vector sits on the span of its
   * source token, and that the provided layer identifier derives from the token layer.
   */
  @Test
  void testPipelineEmbedsEveryToken() {
    final Document tokenized = new SpaceTokenizerAnnotator().annotate(
        Document.of("Dogs bark loudly"));

    final EmbeddingAnnotator annotator =
        new EmbeddingAnnotator(new CharSumEmbedder(), Layers.TOKENS);
    final Document embedded = annotator.annotate(tokenized);

    Assertions.assertEquals("embeddings:opennlp:tokens", annotator.layer().id());
    Assertions.assertEquals(float[].class, annotator.layer().type());

    final List<Annotation<String>> tokens = embedded.get(Layers.TOKENS);
    final List<Annotation<float[]>> vectors = embedded.get(annotator.layer());
    Assertions.assertEquals(3, tokens.size());
    Assertions.assertEquals(tokens.size(), vectors.size());
    for (int i = 0; i < tokens.size(); i++) {
      Assertions.assertEquals(tokens.get(i).span(), vectors.get(i).span());
    }
    // "Dogs" = D(68) o(111) g(103) s(115): components (68+115, 111, 103).
    Assertions.assertArrayEquals(new float[] {183, 111, 103}, vectors.get(0).value());
    // "bark" = b(98) a(97) r(114) k(107): components (98+107, 97, 114).
    Assertions.assertArrayEquals(new float[] {205, 97, 114}, vectors.get(1).value());
    // "loudly" = l(108) o(111) u(117) d(100) l(108) y(121): (108+100, 111+108, 117+121).
    Assertions.assertArrayEquals(new float[] {208, 219, 238}, vectors.get(2).value());
  }

  /**
   * Adds one annotator over the token layer and one over the sentence layer to the same
   * document and asserts that both provided layers coexist, that each carries its own
   * derived identifier, and that the exact sentence and token vectors are present.
   */
  @Test
  void testTokenAndSentenceAnnotatorsCoexist() {
    final String text = "Dogs bark. Cats nap.";
    final Document base = Document.of(text)
        .with(Layers.SENTENCES, List.of(
            new Annotation<>(new Span(0, 10), "Dogs bark."),
            new Annotation<>(new Span(11, 20), "Cats nap.")))
        .with(Layers.TOKENS, List.of(
            new Annotation<>(new Span(0, 4), "Dogs"),
            new Annotation<>(new Span(5, 9), "bark"),
            new Annotation<>(new Span(9, 10), "."),
            new Annotation<>(new Span(11, 15), "Cats"),
            new Annotation<>(new Span(16, 19), "nap"),
            new Annotation<>(new Span(19, 20), ".")));

    final CharSumEmbedder embedder = new CharSumEmbedder();
    final EmbeddingAnnotator overTokens = new EmbeddingAnnotator(embedder, Layers.TOKENS);
    final EmbeddingAnnotator overSentences = new EmbeddingAnnotator(embedder, Layers.SENTENCES);
    final Document document = overSentences.annotate(overTokens.annotate(base));

    Assertions.assertEquals("embeddings:opennlp:tokens", overTokens.layer().id());
    Assertions.assertEquals("embeddings:opennlp:sentences", overSentences.layer().id());
    Assertions.assertTrue(document.layers().contains(overTokens.layer()));
    Assertions.assertTrue(document.layers().contains(overSentences.layer()));

    final List<Annotation<float[]>> sentences = document.get(overSentences.layer());
    Assertions.assertEquals(2, sentences.size());
    // "Dogs bark." = 68 111 103 115 32 98 97 114 107 46 folded by index modulo three.
    Assertions.assertArrayEquals(new float[] {326, 257, 308}, sentences.get(0).value());
    // "Cats nap." = 67 97 116 115 32 110 97 112 46 folded by index modulo three.
    Assertions.assertArrayEquals(new float[] {279, 241, 272}, sentences.get(1).value());
    Assertions.assertEquals(new Span(0, 10), sentences.get(0).span());
    Assertions.assertEquals(new Span(11, 20), sentences.get(1).span());

    final List<Annotation<float[]>> tokens = document.get(overTokens.layer());
    Assertions.assertEquals(6, tokens.size());
    // "Cats" = C(67) a(97) t(116) s(115): components (67+115, 97, 116).
    Assertions.assertArrayEquals(new float[] {182, 97, 116}, tokens.get(3).value());
    // "." is a single character, code 46, landing in the first component only.
    Assertions.assertArrayEquals(new float[] {46, 0, 0}, tokens.get(5).value());
  }
}
