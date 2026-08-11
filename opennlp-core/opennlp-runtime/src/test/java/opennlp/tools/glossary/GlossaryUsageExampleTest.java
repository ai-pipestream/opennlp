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

package opennlp.tools.glossary;

import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnalyzer;
import opennlp.tools.document.Layers;
import opennlp.tools.document.TokenizerAnnotator;
import opennlp.tools.stemmer.snowball.SnowballStemmer;
import opennlp.tools.stemmer.snowball.SnowballStemmerFactory;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.normalizer.GermanUmlautCharSequenceNormalizer;
import opennlp.tools.util.normalizer.TermAnalyzer;

/**
 * Demonstrates the intended end-to-end use of the glossary components: a glossary of
 * multiword terms is compiled into an {@link AhoCorasickGlossaryMatcher}, the matcher is
 * mounted into a {@link DocumentAnalyzer} pipeline through a {@link GlossaryAnnotator},
 * and the glossary layer is read back from the resulting {@link Document} with every
 * span expressed in the coordinates of the original text.
 */
public class GlossaryUsageExampleTest {

  /**
   * Walks the realistic flow from glossary construction to layer readout: three entries,
   * one of them matched with a case difference, are found by a two-step pipeline, and
   * every hit is checked for its exact span, identifier, registered term, and covered
   * text in original text coordinates.
   */
  @Test
  void testGlossaryPipelineEndToEnd() {
    // A small glossary: each surface form resolves to a stable identifier, and a
    // multiword form is one entry, not a token sequence.
    final List<GlossaryEntry> glossary = List.of(
        new GlossaryEntry("LIB-1", "Apache OpenNLP"),
        new GlossaryEntry("ALG-42", "maximum entropy model"),
        new GlossaryEntry("ALG-7", "perceptron model"));

    // The glossary annotator requires no other layer, so it can sit anywhere in the
    // pipeline; here it runs after a model-free tokenizer to show layer coexistence.
    final DocumentAnalyzer analyzer = DocumentAnalyzer.builder()
        .add(new TokenizerAnnotator(WhitespaceTokenizer.INSTANCE))
        .add(new GlossaryAnnotator(new AhoCorasickGlossaryMatcher(glossary, true)))
        .build();

    final String text =
        "Apache OpenNLP ships a Maximum Entropy model and a perceptron model.";
    final Document document = analyzer.analyze(text);

    // Both pipeline steps contributed their layer to the same document.
    Assertions.assertTrue(document.layers().contains(Layers.TOKENS));
    Assertions.assertTrue(document.layers().contains(GlossaryAnnotator.GLOSSARY));
    final List<Annotation<String>> tokens = document.get(Layers.TOKENS);
    Assertions.assertEquals(11, tokens.size());
    Assertions.assertEquals("OpenNLP", tokens.get(1).value());

    final List<Annotation<GlossaryMatch>> hits = document.get(GlossaryAnnotator.GLOSSARY);
    Assertions.assertEquals(3, hits.size());
    assertHit(hits.get(0), 0, 14, "LIB-1", "Apache OpenNLP", "Apache OpenNLP", text);
    // The matcher ignores case, so the registered term and the covered text differ in
    // case while the span still points at the original characters.
    assertHit(hits.get(1), 23, 44, "ALG-42", "maximum entropy model",
        "Maximum Entropy model", text);
    assertHit(hits.get(2), 51, 67, "ALG-7", "perceptron model", "perceptron model", text);
  }

  /**
   * Mirrors the offset-aware normalizer example in {@code glossary.xml}: a German
   * umlaut fold lets a registered ASCII term cover the original eszett surface.
   */
  @Test
  void testOffsetAwareNormalizerManualExample() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ST", "strasse")),
        true,
        GermanUmlautCharSequenceNormalizer.getInstance());

    final String text = "Die Stra\u00DFe ist frei.";
    final List<GlossaryMatch> hits = matcher.match(text);

    Assertions.assertEquals(1, hits.size());
    Assertions.assertEquals("ST", hits.get(0).id());
    Assertions.assertEquals("strasse", hits.get(0).term());
    Assertions.assertEquals("Stra\u00DFe",
        text.substring(hits.get(0).span().getStart(), hits.get(0).span().getEnd()));
  }

  /**
   * Mirrors the inflection example in {@code glossary.xml}: {@code hot dog} matches
   * {@code hot dogs} through {@link TermAnalyzingGlossaryMatcher}, and the span covers
   * the plural surface in original coordinates.
   */
  @Test
  void testInflectedMultiwordManualExample() {
    final TermAnalyzer terms = TermAnalyzer.builder()
        .caseFold()
        .stem(new SnowballStemmerFactory(SnowballStemmer.ALGORITHM.ENGLISH))
        .build();
    final TermAnalyzingGlossaryMatcher matcher = new TermAnalyzingGlossaryMatcher(
        List.of(new GlossaryEntry("FOOD", "hot dog")), terms);

    final String text = "the hot dogs were cold";
    final List<GlossaryMatch> hits = matcher.match(text);

    Assertions.assertEquals(1, hits.size());
    Assertions.assertEquals("FOOD", hits.get(0).id());
    Assertions.assertEquals("hot dog", hits.get(0).term());
    Assertions.assertEquals("hot dogs",
        text.substring(hits.get(0).span().getStart(), hits.get(0).span().getEnd()));
  }

  /**
   * Shows the read side of the layer contract for consumers that cannot know which
   * annotators ran: a document that never saw a glossary annotator reads as an empty
   * glossary layer rather than failing, and the key is absent from the layer set.
   */
  @Test
  void testReadingTheGlossaryLayerOfAnUnannotatedDocumentIsEmpty() {
    final Document document = Document.of("No glossary annotator ran over this text.");

    // An absent layer reads as an empty list, and the key does not appear in layers().
    Assertions.assertTrue(document.get(GlossaryAnnotator.GLOSSARY).isEmpty());
    Assertions.assertFalse(document.layers().contains(GlossaryAnnotator.GLOSSARY));
  }

  /**
   * Asserts one glossary hit in full: its exact span offsets, the identifier and
   * registered term of the matched entry, the text the span covers in the original
   * document text, and that the annotation span and the match's own span agree.
   *
   * @param hit The glossary annotation to check. Must not be {@code null}.
   * @param expectedStart The expected span start, inclusive, in original text chars.
   * @param expectedEnd The expected span end, exclusive, in original text chars.
   * @param expectedId The expected identifier of the matched entry.
   * @param expectedTerm The expected registered term of the matched entry.
   * @param expectedCovered The expected text between the span offsets.
   * @param text The original document text the span refers to. Must not be {@code null}.
   */
  private void assertHit(Annotation<GlossaryMatch> hit, int expectedStart,
      int expectedEnd, String expectedId, String expectedTerm, String expectedCovered,
      String text) {
    Assertions.assertEquals(expectedStart, hit.span().getStart());
    Assertions.assertEquals(expectedEnd, hit.span().getEnd());
    Assertions.assertEquals(hit.span().getStart(), hit.value().span().getStart());
    Assertions.assertEquals(hit.span().getEnd(), hit.value().span().getEnd());
    Assertions.assertEquals(expectedId, hit.value().id());
    Assertions.assertEquals(expectedTerm, hit.value().term());
    Assertions.assertEquals(expectedCovered,
        text.substring(hit.span().getStart(), hit.span().getEnd()));
  }
}
