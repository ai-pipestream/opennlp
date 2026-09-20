/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package opennlp.dl.vectors;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;

import opennlp.tools.eval.AbstractEvalTest;

/**
 * The three {@link PaddingStrategy} choices of {@link SentenceVectorsDL} against the real
 * sentence-transformers MiniLM model, whose only output is {@code last_hidden_state} of shape
 * {@code [batch_size, sequence_length, 384]}, so the sentence vector is pooled in Java.
 *
 * <p>The invariant under test is that padding changes no output vector: a padded position is
 * {@code 0} in the attention mask, the encoder gives it no weight, and {@link Pooling} does not
 * read it. Every assertion compares a padded row against the same input embedded by an
 * {@link PaddingStrategy#EXACT_LENGTH} instance, which runs it at {@code [1, its own length]} with
 * no padded position anywhere. Comparing against the same instance would not do: under
 * {@link PaddingStrategy#MAX_LENGTH} even a single-input call is padded, so both sides would carry
 * the same padding and a wrong padding implementation would cancel out. That was confirmed by
 * temporarily padding the attention mask with {@code 1}: against a separate unpadded reference the
 * deviation is around {@code 1.4} in absolute value, against the same instance it is {@code 0}.</p>
 *
 * <p>{@link #TOLERANCE} is the stated bound. With onnxruntime 1.29.0 on the CPU execution provider
 * the measured deviation over these inputs is exactly {@code 0}; the bound leaves room for a build
 * whose kernel blocking depends on the tensor width.</p>
 */
public class SentenceVectorsDLPaddedBatchEval extends AbstractEvalTest {

  /** The stated bound on how far a padded row may sit from the same row run unpadded. */
  private static final float TOLERANCE = 1e-5f;

  private static final String MODEL = "onnx/sentence-transformers/model.onnx";
  private static final String VOCABULARY = "onnx/sentence-transformers/vocab.txt";
  private static final String CORPUS = "leipzig/eng_news_2010_300K-sentences.txt";

  private static final int DIMENSION = 384;

  /** How many natural sentences the length-varied cases run over. */
  private static final int CORPUS_SENTENCES = 48;

  private static File model() throws IOException {
    return new File(getOpennlpDataDir(), MODEL);
  }

  private static File vocabulary() throws IOException {
    return new File(getOpennlpDataDir(), VOCABULARY);
  }

  private static SentenceVectorsDL vectors(final PaddingStrategy padding, final Pooling pooling,
      final boolean normalize, final int maxLength) throws Exception {
    return new SentenceVectorsDL(model(), vocabulary(), true, pooling, normalize, maxLength,
        padding);
  }

  /**
   * Natural sentences of widely varying length, read from the Leipzig English news corpus, plus
   * short and degenerate inputs so the shortest and the longest share every batch.
   */
  private static List<String> naturalInputs() throws IOException {
    final Path corpus = new File(getOpennlpDataDir(), CORPUS).toPath();
    Assumptions.assumeTrue(Files.isReadable(corpus),
        "the Leipzig English news sentences are needed: " + CORPUS);
    final List<String> texts = new ArrayList<>();
    try (Stream<String> lines = Files.lines(corpus, StandardCharsets.UTF_8)) {
      final Iterator<String> iterator = lines.iterator();
      while (iterator.hasNext() && texts.size() < CORPUS_SENTENCES) {
        final String line = iterator.next();
        // The corpus is "<id><tab><sentence>"; no regular expressions in this project.
        final int tab = line.indexOf('\t');
        final String sentence = (tab >= 0 ? line.substring(tab + 1) : line).strip();
        if (!sentence.isEmpty() && !texts.contains(sentence)) {
          texts.add(sentence);
        }
      }
    }
    Assertions.assertEquals(CORPUS_SENTENCES, texts.size(), "the corpus should be long enough");
    // The extremes, in the same call as the natural text.
    texts.add("");
    texts.add(" ");
    texts.add("a");
    texts.add("hi");
    texts.add(String.join(" ", Collections.nCopies(400, "the quick brown fox jumped")));
    return texts;
  }

  /** The sentence vectors of {@link #naturalInputs()} with no padded position anywhere. */
  private static float[][] unpaddedReference(final List<String> texts, final Pooling pooling,
      final boolean normalize, final int maxLength) throws Exception {
    final float[][] reference = new float[texts.size()][];
    try (SentenceVectorsDL exact = vectors(PaddingStrategy.EXACT_LENGTH, pooling, normalize,
        maxLength)) {
      for (int i = 0; i < texts.size(); i++) {
        reference[i] = exact.embed(texts.get(i));
      }
    }
    return reference;
  }

  private static List<String> inputs;
  private static float[][] meanReference;

  @BeforeAll
  static void loadInputs() throws Exception {
    inputs = naturalInputs();
    meanReference = unpaddedReference(inputs, Pooling.MEAN, false,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH);
  }

  private static Stream<PaddingStrategy> paddingStrategies() {
    return Arrays.stream(PaddingStrategy.values());
  }

  /**
   * Every strategy reproduces the unpadded vector of every input, within {@link #TOLERANCE}, over
   * natural sentences of widely varying length with the shortest and the longest in the same
   * batch. A failure names the input.
   */
  @ParameterizedTest
  @MethodSource("paddingStrategies")
  public void paddedRowsEqualUnpaddedRows(final PaddingStrategy padding) throws Exception {
    try (SentenceVectorsDL sv = vectors(padding, Pooling.MEAN, false,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH)) {
      final float[][] batch = sv.embedAll(inputs);
      Assertions.assertEquals(inputs.size(), batch.length);
      for (int i = 0; i < inputs.size(); i++) {
        Assertions.assertArrayEquals(meanReference[i], batch[i], TOLERANCE,
            padding + ": row " + i + " of " + inputs.size() + " differs from its unpadded vector, "
                + "input [" + shorten(inputs.get(i)) + "]");
      }
    }
  }

  /**
   * The same holds with unit-length scaling and with {@link Pooling#CLS}, the two other shapes the
   * pooling step can take. Mean pooling unscaled is covered by
   * {@link #paddedRowsEqualUnpaddedRows(PaddingStrategy)}.
   */
  @ParameterizedTest
  @MethodSource("paddingStrategies")
  public void paddedRowsEqualUnpaddedRowsForEveryPooling(final PaddingStrategy padding)
      throws Exception {
    for (final Pooling pooling : Pooling.values()) {
      final boolean normalize = pooling == Pooling.MEAN;
      final float[][] reference = unpaddedReference(inputs, pooling, normalize,
          SentenceVectorsDL.DEFAULT_MAX_LENGTH);
      try (SentenceVectorsDL sv = vectors(padding, pooling, normalize,
          SentenceVectorsDL.DEFAULT_MAX_LENGTH)) {
        final float[][] batch = sv.embedAll(inputs);
        for (int i = 0; i < inputs.size(); i++) {
          Assertions.assertArrayEquals(reference[i], batch[i], TOLERANCE,
              padding + ", " + pooling + ", normalize=" + normalize + ": row " + i + ", input ["
                  + shorten(inputs.get(i)) + "]");
        }
      }
    }
  }

  /**
   * The extreme pair on its own: the shortest and the longest input of the set in one batch of two,
   * where the short row is padded by the whole difference in length. Their vectors are far apart,
   * so the equality assertions are not trivially satisfied.
   */
  @ParameterizedTest
  @MethodSource("paddingStrategies")
  public void shortestAndLongestInTheSameBatch(final PaddingStrategy padding) throws Exception {
    final String shortest = "hi";
    final String longest = String.join(" ", Collections.nCopies(200, "the quick brown fox"));
    final List<String> pair = List.of(shortest, longest);
    final float[][] reference = unpaddedReference(pair, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH);
    try (SentenceVectorsDL sv = vectors(padding, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH)) {
      final float[][] batch = sv.embedAll(pair);
      Assertions.assertArrayEquals(reference[0], batch[0], TOLERANCE,
          padding + ": the shortest input beside the longest");
      Assertions.assertArrayEquals(reference[1], batch[1], TOLERANCE,
          padding + ": the longest input beside the shortest");
      Assertions.assertTrue(cosine(batch[0], batch[1]) < 0.9,
          padding + ": the two must be clearly different vectors, otherwise the equality "
              + "assertions above prove nothing");
    }
  }

  /**
   * A vector is independent of how many inputs travel with it: alone, in a batch of two, and in a
   * large mixed batch all give the same vector.
   */
  @ParameterizedTest
  @MethodSource("paddingStrategies")
  public void vectorIsIndependentOfBatchSize(final PaddingStrategy padding) throws Exception {
    final String companion = String.join(" ", Collections.nCopies(30, "a longer companion"));
    try (SentenceVectorsDL sv = vectors(padding, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH)) {
      final float[][] large = sv.embedAll(inputs);
      for (int i = 0; i < inputs.size(); i++) {
        final String text = inputs.get(i);
        final float[] alone = sv.embedAll(List.of(text))[0];
        Assertions.assertArrayEquals(alone, sv.embed(text), TOLERANCE,
            padding + ": embed and a batch of one, [" + shorten(text) + "]");
        Assertions.assertArrayEquals(alone, sv.embedAll(List.of(text, companion))[0], TOLERANCE,
            padding + ": a batch of two, [" + shorten(text) + "]");
        Assertions.assertArrayEquals(alone, sv.embedAll(List.of(companion, text))[1], TOLERANCE,
            padding + ": a batch of two, reversed, [" + shorten(text) + "]");
        Assertions.assertArrayEquals(alone, large[i], TOLERANCE,
            padding + ": a batch of " + inputs.size() + ", [" + shorten(text) + "]");
      }
    }
  }

  /**
   * Row {@code i} is the vector of input {@code i}. The inputs are sentences on clearly unrelated
   * subjects, so each batch row is nearest, by cosine similarity, to its own single-input vector
   * and not to any other. A test over similar inputs could not prove the association.
   */
  @ParameterizedTest
  @MethodSource("paddingStrategies")
  public void orderIsPreserved(final PaddingStrategy padding) throws Exception {
    // Unrelated subjects, deliberately of very different lengths so the rows are reordered
    // internally before the batch runs.
    final List<String> texts = List.of(
        "the cat sat on the mat",
        "quantum chromodynamics describes the strong interaction between quarks and gluons, and "
            + "its coupling constant grows at long distances",
        "bake the bread at two hundred degrees for forty minutes",
        "he was arrested in Marseille on suspicion of tax fraud",
        "photosynthesis converts light into chemical energy",
        "the Dow Jones industrial average closed up three hundred points on Friday after the "
            + "Federal Reserve signalled that it would hold interest rates steady for the rest of "
            + "the year",
        "she scored the winning goal in the ninety fourth minute",
        "install the driver, reboot, and check the kernel log for errors",
        "Mozart wrote the Jupiter symphony in seventeen eighty eight",
        "volcanic ash grounded flights across northern Europe");
    final float[][] reference = unpaddedReference(texts, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH);
    try (SentenceVectorsDL sv = vectors(padding, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH)) {
      final float[][] batch = sv.embedAll(texts);
      for (int i = 0; i < texts.size(); i++) {
        int nearest = -1;
        double best = -2;
        for (int j = 0; j < texts.size(); j++) {
          final double similarity = cosine(batch[i], reference[j]);
          if (similarity > best) {
            best = similarity;
            nearest = j;
          }
        }
        Assertions.assertEquals(i, nearest, padding + ": batch row " + i + " is nearest to input "
            + nearest + " [" + shorten(texts.get(nearest)) + "] rather than to its own input ["
            + shorten(texts.get(i)) + "]");
        Assertions.assertArrayEquals(reference[i], batch[i], TOLERANCE,
            padding + ": row " + i + ", [" + shorten(texts.get(i)) + "]");
      }
    }
  }

  /**
   * Boundary inputs a batch has to carry alongside natural text: no inputs at all, one input, an
   * empty string, whitespace only, a one-character input, and duplicates. An empty and a
   * whitespace-only input both encode to the wrapped {@code [CLS] [SEP]}, so both get that
   * vector rather than a zero vector.
   */
  @ParameterizedTest
  @MethodSource("paddingStrategies")
  public void boundaryInputs(final PaddingStrategy padding) throws Exception {
    final List<String> edges = List.of("", " ", "\t\r\n", " ", "a", "é", "hi", "hi",
        "george washington was president", "george washington was president");
    final float[][] reference = unpaddedReference(edges, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH);
    try (SentenceVectorsDL sv = vectors(padding, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH)) {
      Assertions.assertEquals(0, sv.embedAll(List.of()).length, "an empty call");
      Assertions.assertArrayEquals(reference[4], sv.embedAll(List.of("a"))[0], TOLERANCE,
          "a call of one");
      final float[][] batch = sv.embedAll(edges);
      Assertions.assertEquals(edges.size(), batch.length);
      for (int i = 0; i < edges.size(); i++) {
        Assertions.assertEquals(DIMENSION, batch[i].length);
        Assertions.assertArrayEquals(reference[i], batch[i], TOLERANCE,
            padding + ": row " + i + ", [" + edges.get(i) + "]");
      }
      // Duplicates get equal vectors in separate arrays.
      Assertions.assertArrayEquals(batch[6], batch[7], 0f, "duplicate short inputs");
      Assertions.assertArrayEquals(batch[8], batch[9], 0f, "duplicate sentences");
      Assertions.assertNotSame(batch[6], batch[7]);
      // The empty and whitespace-only inputs are the [CLS] [SEP] vector, which is not a zero
      // vector and not the vector of "a".
      Assertions.assertArrayEquals(batch[0], batch[1], TOLERANCE, "empty and a space");
      Assertions.assertTrue(cosine(batch[0], batch[4]) < 0.999,
          "the empty input must not give the same vector as a one-character input");
    }
  }

  /**
   * An input longer than the maximum sequence length is truncated to it, keeping the final
   * {@code [SEP]}, whatever the strategy. The truncated form is what every strategy embeds, and
   * {@link PaddingStrategy#MAX_LENGTH} therefore leaves such a row with no padded position at all.
   */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  public void inputLongerThanTheMaximumSequenceLength(final PaddingStrategy padding)
      throws Exception {
    final String body = String.join(" ", Collections.nCopies(600, "the quick brown fox jumped"));
    // Over 512 wordpieces, so the encoding is cut to 512 whatever follows.
    final List<String> texts = List.of("hi", body, body + " and a different tail entirely",
        "george washington was president");
    final float[][] reference = unpaddedReference(texts, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH);
    try (SentenceVectorsDL sv = vectors(padding, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH)) {
      final float[][] batch = sv.embedAll(texts);
      for (int i = 0; i < texts.size(); i++) {
        Assertions.assertArrayEquals(reference[i], batch[i], TOLERANCE,
            padding + ": row " + i + ", [" + shorten(texts.get(i)) + "]");
      }
      // Both long inputs truncate to the same first 512 tokens, so they get the same vector.
      Assertions.assertArrayEquals(batch[1], batch[2], TOLERANCE,
          padding + ": two inputs that truncate to the same 512 tokens must agree");
    }
    // A smaller maximum truncates further, and the short rows beside it are unaffected.
    final float[][] shortReference = unpaddedReference(texts, Pooling.MEAN, true, 32);
    try (SentenceVectorsDL sv = vectors(padding, Pooling.MEAN, true, 32)) {
      final float[][] batch = sv.embedAll(texts);
      for (int i = 0; i < texts.size(); i++) {
        Assertions.assertArrayEquals(shortReference[i], batch[i], TOLERANCE,
            padding + ", maxLength 32: row " + i + ", [" + shorten(texts.get(i)) + "]");
      }
    }
  }

  /**
   * Unicode inputs against the real vocabulary: supplementary code points with and without a
   * combining mark, the precomposed and decomposed forms of the same accented text, and four
   * non-Latin scripts. Each shares a batch with ASCII text of very different length and must still
   * reproduce its unpadded vector.
   *
   * <p>The model is uncased, so tokenization strips accents; the precomposed and decomposed forms
   * of one word therefore also have to agree with each other, which is asserted separately.</p>
   */
  @ParameterizedTest
  @MethodSource("paddingStrategies")
  public void unicodeInputs(final PaddingStrategy padding) throws Exception {
    final List<String> texts = new ArrayList<>(List.of(
        "𐐒",                                 // U+10412, DESERET CAPITAL LETTER EF
        "𐐒́",                           // the same with a combining acute
        "𐐒𐐓𐐔 text",
        "café crème",                         // precomposed
        "café crème",                       // decomposed, same text
        "été à Paris",
        "中文句子测试",         // Han
        "اللغة العربية",  // Arabic
        "हिन्दी वाक्य",        // Devanagari
        "Алексей был здес"
            + "ь",                                                               // Cyrillic
        "😀 🚀"));                  // emoji, supplementary and unmapped
    texts.add("hi");
    texts.add(String.join(" ", Collections.nCopies(150, "a long ASCII companion sentence")));
    final float[][] reference = unpaddedReference(texts, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH);
    try (SentenceVectorsDL sv = vectors(padding, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH)) {
      final float[][] batch = sv.embedAll(texts);
      for (int i = 0; i < texts.size(); i++) {
        Assertions.assertEquals(DIMENSION, batch[i].length);
        Assertions.assertArrayEquals(reference[i], batch[i], TOLERANCE,
            padding + ": row " + i + ", [" + texts.get(i) + "]");
      }
      // Precomposed and decomposed forms of the same words agree after accent stripping.
      Assertions.assertArrayEquals(batch[3], batch[4], TOLERANCE,
          padding + ": precomposed and decomposed forms must give the same vector");
    }
  }

  /**
   * A call large enough to be cut into sub-batches by
   * {@link SentenceVectorsDL#MAX_BATCH_TOKEN_POSITIONS} still returns every row correctly and in
   * order. Sized from the bound rather than hardcoded.
   */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  public void callLargeEnoughToBeChunked(final PaddingStrategy padding) throws Exception {
    final int maxLength = 128;
    // Enough rows at the fixed width to need three sub-batches under MAX_LENGTH.
    final int rows = 2 * SentenceVectorsDL.MAX_BATCH_TOKEN_POSITIONS / maxLength + 3;
    final List<String> texts = new ArrayList<>(rows);
    for (int i = 0; i < rows; i++) {
      texts.add(inputs.get(i % inputs.size()));
    }
    final float[][] reference = unpaddedReference(inputs, Pooling.MEAN, true, maxLength);
    try (SentenceVectorsDL sv = vectors(padding, Pooling.MEAN, true, maxLength)) {
      final float[][] batch = sv.embedAll(texts);
      Assertions.assertEquals(rows, batch.length);
      for (int i = 0; i < rows; i++) {
        Assertions.assertArrayEquals(reference[i % inputs.size()], batch[i], TOLERANCE,
            padding + ": row " + i + " of " + rows + ", [" + shorten(texts.get(i)) + "]");
      }
    }
  }

  /** {@code null} is rejected at the public boundary, the same way under every strategy. */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  public void nullHandling(final PaddingStrategy padding) throws Exception {
    try (SentenceVectorsDL sv = vectors(padding, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH)) {
      Assertions.assertEquals("texts must not be null",
          Assertions.assertThrows(IllegalArgumentException.class,
              () -> sv.embedAll(null)).getMessage());
      Assertions.assertEquals("texts[1] must not be null",
          Assertions.assertThrows(IllegalArgumentException.class,
              () -> sv.embedAll(Arrays.asList("hi", null))).getMessage());
      Assertions.assertEquals("text must not be null",
          Assertions.assertThrows(IllegalArgumentException.class,
              () -> sv.embed(null)).getMessage());
      Assertions.assertEquals("sentence must not be null",
          Assertions.assertThrows(IllegalArgumentException.class,
              () -> sv.getVectors(null)).getMessage());
    }
    Assertions.assertEquals("padding must not be null",
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> new SentenceVectorsDL(model(), vocabulary(), true, Pooling.MEAN, true,
                SentenceVectorsDL.DEFAULT_MAX_LENGTH, null)).getMessage());
  }

  /** No strategy changes the reported dimension, and the default is exact-length grouping. */
  @Test
  public void dimensionAndDefault() throws Exception {
    Assertions.assertEquals(PaddingStrategy.EXACT_LENGTH, SentenceVectorsDL.DEFAULT_PADDING);
    for (final PaddingStrategy padding : PaddingStrategy.values()) {
      try (SentenceVectorsDL sv = vectors(padding, Pooling.MEAN, true,
          SentenceVectorsDL.DEFAULT_MAX_LENGTH)) {
        Assertions.assertEquals(DIMENSION, sv.dimension(), padding.toString());
        sv.embedAll(inputs);
        Assertions.assertEquals(DIMENSION, sv.dimension(), padding + " after a batch");
        Assertions.assertEquals(DIMENSION, sv.embed("hi").length, padding.toString());
      }
    }
    try (SentenceVectorsDL sv = new SentenceVectorsDL(model(), vocabulary())) {
      Assertions.assertEquals(DIMENSION, sv.dimension());
    }
  }

  /**
   * The strategies differ in the tensor shapes and the inference count of one call, which is the
   * only thing that distinguishes them. The tokenized lengths of the natural sentences decide the
   * expected numbers, so they are computed from the call rather than written down.
   */
  @Test
  public void strategiesShapeTheTensorsDifferently() throws Exception {
    final int maxLength = SentenceVectorsDL.DEFAULT_MAX_LENGTH;
    try (SentenceVectorsDL exact = vectors(PaddingStrategy.EXACT_LENGTH, Pooling.MEAN, true,
             maxLength);
         SentenceVectorsDL longest = vectors(PaddingStrategy.LONGEST, Pooling.MEAN, true,
             maxLength);
         SentenceVectorsDL fixed = vectors(PaddingStrategy.MAX_LENGTH, Pooling.MEAN, true,
             maxLength)) {

      final int[][] exactShapes = exact.batchShapes(inputs);
      final int[][] longestShapes = longest.batchShapes(inputs);
      final int[][] fixedShapes = fixed.batchShapes(inputs);

      // Exact-length grouping fragments on natural text: many inferences, most of them narrow.
      Assertions.assertTrue(exactShapes.length > inputs.size() / 2,
          "exact-length grouping should fragment on natural sentences, it produced only "
              + exactShapes.length + " inferences for " + inputs.size() + " inputs");
      for (final int[] shape : exactShapes) {
        Assertions.assertTrue(shape[0] >= 1);
      }

      // Padding to the longest row collapses the same call into far fewer inferences.
      Assertions.assertTrue(longestShapes.length < exactShapes.length,
          "LONGEST must run fewer inferences than EXACT_LENGTH: " + longestShapes.length
              + " against " + exactShapes.length);

      // Every row of a fixed-width call is exactly maxLength wide.
      for (final int[] shape : fixedShapes) {
        Assertions.assertEquals(maxLength, shape[1]);
      }

      // Every strategy accounts for every input exactly once and respects the bound.
      for (final int[][] shapes : List.of(exactShapes, longestShapes, fixedShapes)) {
        int rows = 0;
        for (final int[] shape : shapes) {
          rows += shape[0];
          Assertions.assertTrue(
              shape[0] == 1 || shape[0] * shape[1] <= SentenceVectorsDL.MAX_BATCH_TOKEN_POSITIONS,
              "a batch of " + shape[0] + " by " + shape[1] + " exceeds the bound");
        }
        Assertions.assertEquals(inputs.size(), rows);
      }
    }
  }

  private static double cosine(final float[] a, final float[] b) {
    double dot = 0;
    double na = 0;
    double nb = 0;
    for (int d = 0; d < a.length; d++) {
      dot += (double) a[d] * b[d];
      na += (double) a[d] * a[d];
      nb += (double) b[d] * b[d];
    }
    return na == 0 || nb == 0 ? 0 : dot / Math.sqrt(na * nb);
  }

  private static String shorten(final String text) {
    return text.length() <= 60 ? text : text.substring(0, 57) + "...";
  }
}
