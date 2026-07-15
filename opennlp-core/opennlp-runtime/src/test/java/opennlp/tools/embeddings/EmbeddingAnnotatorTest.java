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

import java.util.List;

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
}
