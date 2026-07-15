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

package opennlp.tools.document;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.postag.POSTagger;
import opennlp.tools.sentdetect.SentenceDetector;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.util.Sequence;
import opennlp.tools.util.Span;

/**
 * Runs the code from the "The Document Model" manual chapter (document.xml) and asserts
 * the behavior the chapter states. Each test mirrors an example in that chapter; if the
 * chapter and the API disagree, a test here fails.
 */
public class ManualExamplesTest {

  private static final String TEXT =
      "Rockwell International Corp. said it signed an agreement.";

  @Test
  void documentStartsWithNoLayersAndKeepsItsText() {
    final Document document = Document.of(TEXT);
    Assertions.assertTrue(document.layers().isEmpty());
    Assertions.assertEquals(TEXT, document.text().toString());
  }

  @Test
  void annotationPairsASpanWithAValue() {
    final Annotation<String> annotation = new Annotation<>(new Span(0, 8), "Rockwell");
    Assertions.assertEquals(0, annotation.span().getStart());
    Assertions.assertEquals(8, annotation.span().getEnd());
    Assertions.assertEquals("Rockwell", annotation.value());
    // the span indexes the original text
    Assertions.assertEquals("Rockwell",
        annotation.span().getCoveredText(TEXT).toString());
  }

  @Test
  void withAddsALayerAndLeavesTheOriginalUnchanged() {
    final Document document = Document.of(TEXT);
    final List<Annotation<String>> tokens = List.of(
        new Annotation<>(new Span(0, 8), "Rockwell"));

    final Document tokenized = document.with(Layers.TOKENS, tokens);

    Assertions.assertTrue(tokenized.layers().contains(Layers.TOKENS));
    Assertions.assertEquals(tokens, tokenized.get(Layers.TOKENS));
    // the original is unchanged
    Assertions.assertTrue(document.layers().isEmpty());
  }

  @Test
  void getReturnsAnEmptyListForAnAbsentLayer() {
    final Document document = Document.of(TEXT);
    Assertions.assertTrue(document.get(Layers.TOKENS).isEmpty());
  }

  @Test
  void withRejectsALayerThatIsAlreadyPresent() {
    final List<Annotation<String>> tokens = List.of(
        new Annotation<>(new Span(0, 8), "Rockwell"));
    final Document tokenized = Document.of(TEXT).with(Layers.TOKENS, tokens);
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> tokenized.with(Layers.TOKENS, tokens));
  }

  @Test
  void layerKeysAreEqualByIdentifierAndType() {
    final LayerKey<String> reconstructed = LayerKey.of("tokens", String.class);
    Assertions.assertEquals(Layers.TOKENS, reconstructed);
    // a reconstructed key addresses the same layer
    final List<Annotation<String>> tokens = List.of(
        new Annotation<>(new Span(0, 8), "Rockwell"));
    final Document tokenized = Document.of(TEXT).with(Layers.TOKENS, tokens);
    Assertions.assertEquals(tokens, tokenized.get(reconstructed));
  }

  @Test
  void theBuiltInLayerKeysCarryStringValues() {
    for (final LayerKey<String> key : List.of(
        Layers.SENTENCES, Layers.TOKENS, Layers.POS_TAGS, Layers.ENTITIES)) {
      Assertions.assertEquals(String.class, key.type());
    }
  }

  @Test
  void theBuiltInAnnotatorsDeclareTheLayersTheChapterLists() {
    final DocumentAnnotator sentences = new SentenceDetectorAnnotator(new StubSentenceDetector());
    Assertions.assertEquals(Set.of(), sentences.requires());
    Assertions.assertEquals(Set.of(Layers.SENTENCES), sentences.provides());

    final DocumentAnnotator tokens = new TokenizerAnnotator(new StubTokenizer());
    Assertions.assertEquals(Set.of(), tokens.requires());
    Assertions.assertEquals(Set.of(Layers.TOKENS), tokens.provides());

    final DocumentAnnotator tags = new POSTaggerAnnotator(new StubPOSTagger());
    Assertions.assertEquals(Set.of(Layers.TOKENS), tags.requires());
    Assertions.assertEquals(Set.of(Layers.POS_TAGS), tags.provides());
  }

  @Test
  void theBuilderRejectsAnUnsatisfiedRequirement() {
    // POSTaggerAnnotator requires TOKENS, which nothing before it provides
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> DocumentAnalyzer.builder()
            .add(new POSTaggerAnnotator(new StubPOSTagger()))
            .build());
  }

  @Test
  void aPipelineProducesTokenAndTagLayersThatLineUp() {
    final DocumentAnalyzer analyzer = DocumentAnalyzer.builder()
        .add(new SentenceDetectorAnnotator(new StubSentenceDetector()))
        .add(new TokenizerAnnotator(new StubTokenizer()))
        .add(new POSTaggerAnnotator(new StubPOSTagger()))
        .build();

    final Document document = analyzer.analyze(TEXT);

    final List<Annotation<String>> tokens = document.get(Layers.TOKENS);
    final List<Annotation<String>> tags = document.get(Layers.POS_TAGS);
    Assertions.assertFalse(tokens.isEmpty());
    Assertions.assertEquals(tokens.size(), tags.size());
    for (int i = 0; i < tokens.size(); i++) {
      // a tag shares the span of the token it describes
      Assertions.assertEquals(tokens.get(i).span(), tags.get(i).span());
    }
    // the token spans index the original text
    Assertions.assertEquals("Rockwell",
        tokens.get(0).span().getCoveredText(TEXT).toString());
  }

  @Test
  void theCustomAcronymAnnotatorFlagsAllCapsTokens() {
    // build a token layer by hand: "NASA and IBM met the US team"
    final String text = "NASA and IBM met the US team";
    final List<Annotation<String>> tokens = tokenLayer(text);
    final Document document = Document.of(text).with(Layers.TOKENS, tokens);

    final AcronymAnnotator annotator = new AcronymAnnotator();
    Assertions.assertEquals(Set.of(Layers.TOKENS), annotator.requires());
    Assertions.assertEquals(Set.of(AcronymAnnotator.ACRONYMS), annotator.provides());

    final Document typed = annotator.annotate(document);
    final List<Annotation<String>> acronyms = typed.get(AcronymAnnotator.ACRONYMS);

    final List<String> flagged = new ArrayList<>();
    for (final Annotation<String> acronym : acronyms) {
      flagged.add(acronym.value());
      // acronyms keep the span of the token they came from
      Assertions.assertEquals(acronym.value(),
          acronym.span().getCoveredText(text).toString());
    }
    Assertions.assertEquals(List.of("NASA", "IBM", "US"), flagged);
  }

  /** Splits on single spaces into token annotations over the original text. */
  private static List<Annotation<String>> tokenLayer(String text) {
    final List<Annotation<String>> tokens = new ArrayList<>();
    int start = 0;
    for (int i = 0; i <= text.length(); i++) {
      if (i == text.length() || text.charAt(i) == ' ') {
        if (i > start) {
          tokens.add(new Annotation<>(new Span(start, i), text.substring(start, i)));
        }
        start = i + 1;
      }
    }
    return tokens;
  }

  // ---- the AcronymAnnotator exactly as printed in document.xml ----

  static final class AcronymAnnotator implements DocumentAnnotator {

    public static final LayerKey<String> ACRONYMS =
        LayerKey.of("acronyms", String.class);

    @Override
    public Set<LayerKey<?>> requires() {
      return Set.of(Layers.TOKENS);
    }

    @Override
    public Set<LayerKey<?>> provides() {
      return Set.of(ACRONYMS);
    }

    @Override
    public Document annotate(Document document) {
      List<Annotation<String>> acronyms = new ArrayList<>();
      for (Annotation<String> token : document.get(Layers.TOKENS)) {
        if (isAcronym(token.value())) {
          acronyms.add(new Annotation<>(token.span(), token.value()));
        }
      }
      return document.with(ACRONYMS, acronyms);
    }

    private static boolean isAcronym(String form) {
      if (form.length() < 2) {
        return false;
      }
      for (int i = 0; i < form.length(); i++) {
        if (!Character.isUpperCase(form.charAt(i))) {
          return false;
        }
      }
      return true;
    }
  }

  // ---- deterministic stub components, so the pipeline example actually runs ----

  private static final class StubSentenceDetector implements SentenceDetector {
    @Override
    public String[] sentDetect(CharSequence s) {
      return new String[] {s.toString()};
    }

    @Override
    public Span[] sentPosDetect(CharSequence s) {
      return new Span[] {new Span(0, s.length())};
    }
  }

  private static final class StubTokenizer implements Tokenizer {
    @Override
    public String[] tokenize(String s) {
      final Span[] spans = tokenizePos(s);
      final String[] out = new String[spans.length];
      for (int i = 0; i < spans.length; i++) {
        out[i] = spans[i].getCoveredText(s).toString();
      }
      return out;
    }

    @Override
    public Span[] tokenizePos(String s) {
      final List<Span> spans = new ArrayList<>();
      int start = 0;
      for (int i = 0; i <= s.length(); i++) {
        if (i == s.length() || s.charAt(i) == ' ') {
          if (i > start) {
            spans.add(new Span(start, i));
          }
          start = i + 1;
        }
      }
      return spans.toArray(new Span[0]);
    }
  }

  private static final class StubPOSTagger implements POSTagger {
    @Override
    public String[] tag(String[] sentence) {
      final String[] tags = new String[sentence.length];
      for (int i = 0; i < tags.length; i++) {
        tags[i] = "X";
      }
      return tags;
    }

    @Override
    public String[] tag(String[] sentence, Object[] additionalContext) {
      return tag(sentence);
    }

    @Override
    public Sequence[] topKSequences(String[] sentence) {
      return new Sequence[0];
    }

    @Override
    public Sequence[] topKSequences(String[] sentence, Object[] additionalContext) {
      return new Sequence[0];
    }
  }
}
