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

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import opennlp.examples.DocExampleSupport;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnalyzer;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.DocumentAnnotators;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTagger;
import opennlp.tools.postag.POSTaggerAnnotator;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetector;
import opennlp.tools.sentdetect.SentenceDetectorAnnotator;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.termvector.TermVector;
import opennlp.tools.termvector.TermVectorAnnotator;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerAnnotator;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.Span;
import opennlp.tools.util.normalizer.CharSequenceNormalizer;
import opennlp.tools.util.normalizer.TextNormalizer;

/**
 * The samples printed in the Document chapter of the manual.
 * <p>
 * The pipeline samples run the published English sentence and part-of-speech models with a
 * whitespace tokenizer, the tokenizer for which the chapter's stated spans hold. The custom
 * annotator sample declares class members, so its region sits at class level.
 */
class DocumentChapterTest extends DocExampleSupport {

  private static final String TEXT = "The dog barks. It naps.";

  private static SentenceModel sentenceModel;

  private static POSModel posModel;

  @BeforeAll
  static void loadModels() throws IOException {
    try (InputStream in = Files.newInputStream(stageModel("opennlp-models-sentdetect-*.jar",
        "opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin"))) {
      sentenceModel = new SentenceModel(in);
    }
    try (InputStream in = Files.newInputStream(stageModel("opennlp-models-pos-*.jar",
        "opennlp-en-ud-ewt-pos-1.3-2.5.4.bin"))) {
      posModel = new POSModel(in);
    }
  }

  // docs:begin example.document.custom
  static final LayerKey<Integer> TOKEN_LENGTHS =
      LayerKey.of("token-lengths", Integer.class);

  class TokenLengthAnnotator implements DocumentAnnotator {

    @Override
    public Document annotate(Document document) {
      DocumentAnnotators.requireLayers(document, Layers.TOKENS);
      List<Annotation<String>> tokens = document.get(Layers.TOKENS);
      List<Annotation<Integer>> lengths = new ArrayList<>(tokens.size());
      for (Annotation<String> token : tokens) {
        lengths.add(new Annotation<>(token.span(), token.value().length()));
      }
      return document.with(TOKEN_LENGTHS, lengths);
    }

    @Override
    public Set<LayerKey<?>> requires() {
      return Set.of(Layers.TOKENS);
    }

    @Override
    public Set<LayerKey<?>> provides() {
      return Set.of(TOKEN_LENGTHS);
    }
  }
  // docs:end

  @Test
  void attachesADocumentLevelValue() {
    Document document = Document.of(TEXT);
    // docs:begin example.document.introduction.scope
    LayerKey<String> LANGUAGE = LayerKey.document("app:language", String.class);

    Document tagged = document.with(LANGUAGE, List.of(Annotation.of("eng")));
    String language = tagged.get(LANGUAGE).get(0).value(); // "eng", span is null
    // docs:end
    Assertions.assertEquals("eng", language);
    Assertions.assertNull(tagged.get(LANGUAGE).get(0).span());
  }

  @Test
  void runsAPipeline() {
    SentenceDetector sentenceDetector = sentenceDetector();
    Tokenizer tokenizer = WhitespaceTokenizer.INSTANCE;
    POSTagger tagger = tagger();
    // docs:begin example.document.pipeline
    DocumentAnalyzer analyzer = DocumentAnalyzer.builder()
        .add(new SentenceDetectorAnnotator(sentenceDetector))
        .add(new TokenizerAnnotator(tokenizer))
        .add(new POSTaggerAnnotator(tagger))
        .add(new TokenLengthAnnotator())
        .build();

    Document document = analyzer.analyze("The dog barks. It naps.");
    // docs:end
    Assertions.assertEquals(List.of(new Span(0, 14), new Span(15, 23)),
        document.get(Layers.SENTENCES).stream().map(Annotation::span).toList());
    Assertions.assertEquals(List.of(new Span(0, 3), new Span(4, 7), new Span(8, 14),
            new Span(15, 17), new Span(18, 23)),
        document.get(Layers.TOKENS).stream().map(Annotation::span).toList());
    Assertions.assertEquals(List.of("DET", "NOUN", "PUNCT", "PRON", "PUNCT"),
        document.get(Layers.POS_TAGS).stream().map(Annotation::value).toList());
  }

  @Test
  void alignsLayersWithTokens() {
    Document document = analyze();
    final List<String> coveredTexts = new ArrayList<>();
    // docs:begin example.document.pipeline.layers
    List<Annotation<String>> tokens = document.get(Layers.TOKENS);
    List<Annotation<String>> tags = document.get(Layers.POS_TAGS);
    for (int i = 0; i < tags.size(); i++) {
      // each tag sits on its token's span, e.g. "DET" on [0..3) for "The"
      Span span = tags.get(i).span();
    }

    // every span refers to the original text, so covered text round-trips
    for (Annotation<String> token : tokens) {
      CharSequence covered = token.span().getCoveredText(document.text());
    }
    // docs:end
    for (int i = 0; i < tags.size(); i++) {
      Assertions.assertEquals(tokens.get(i).span(), tags.get(i).span());
    }
    for (Annotation<String> token : tokens) {
      coveredTexts.add(token.span().getCoveredText(document.text()).toString());
    }
    Assertions.assertEquals(List.of("The", "dog", "barks.", "It", "naps."), coveredTexts);
    Assertions.assertEquals("DET", tags.get(0).value());
    Assertions.assertEquals(new Span(0, 3), tags.get(0).span());
  }

  @Test
  void computesTokenLengths() {
    Assertions.assertEquals(List.of(3, 3, 6, 2, 5),
        analyze().get(TOKEN_LENGTHS).stream().map(Annotation::value).toList());
  }

  @Test
  void readsTokenLengths() {
    Document document = analyze();
    // docs:begin example.document.custom.lengths
    List<Annotation<Integer>> lengths = document.get(TOKEN_LENGTHS);
    int firstTokenLength = lengths.get(0).value(); // 3, for "The" at [0..3)
    // docs:end
    Assertions.assertEquals(3, firstTokenLength);
    Assertions.assertEquals(new Span(0, 3), lengths.get(0).span());
  }

  @Test
  void buildsTermVectors() {
    Tokenizer tokenizer = WhitespaceTokenizer.INSTANCE;
    // docs:begin example.document.termvectors
    DocumentAnalyzer analyzer = DocumentAnalyzer.builder()
        .add(new TokenizerAnnotator(tokenizer))
        .add(new TermVectorAnnotator())
        .build();

    Document document = analyzer.analyze("The dog barks. The dog naps.");
    List<Annotation<TermVector>> vectors = document.get(TermVectorAnnotator.TERM_VECTORS);
    // vectors.get(1).value() is ("dog", 2, [[4..7), [19..22)])
    // docs:end
    Assertions.assertEquals(new TermVector("dog", 2, List.of(new Span(4, 7), new Span(19, 22))),
        vectors.get(1).value());
  }

  @Test
  void foldsTermVectors() {
    // docs:begin example.document.termvectors.normalized
    CharSequenceNormalizer folder = TextNormalizer.builder().caseFold().build();
    DocumentAnalyzer analyzer = DocumentAnalyzer.builder()
        .add(new TokenizerAnnotator(WhitespaceTokenizer.INSTANCE))
        .add(new TermVectorAnnotator(folder))
        .build();

    Document document = analyzer.analyze("Word word WORD");
    List<Annotation<TermVector>> vectors = document.get(TermVectorAnnotator.TERM_VECTORS);
    // vectors.get(0).value() is ("word", 3, [[0..4), [5..9), [10..14)])
    // docs:end
    Assertions.assertEquals(new TermVector("word", 3,
        List.of(new Span(0, 4), new Span(5, 9), new Span(10, 14))), vectors.get(0).value());
  }

  private Document analyze() {
    return DocumentAnalyzer.builder()
        .add(new SentenceDetectorAnnotator(sentenceDetector()))
        .add(new TokenizerAnnotator(WhitespaceTokenizer.INSTANCE))
        .add(new POSTaggerAnnotator(tagger()))
        .add(new TokenLengthAnnotator())
        .build()
        .analyze(TEXT);
  }

  private static SentenceDetector sentenceDetector() {
    return new SentenceDetectorME(sentenceModel);
  }

  private static POSTagger tagger() {
    return new POSTaggerME(posModel);
  }
}
