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
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The three {@link PaddingStrategy} choices of
 * {@link SentenceVectorsDL#embedAll(java.util.List)}, against the deterministic
 * {@code tiny-vectors.onnx} and {@code tiny-pooled.onnx} graphs described by
 * {@link SentenceVectorsDLEmbedderTest}.
 *
 * <p>{@code tiny-vectors.onnx} computes {@code output[b][t] = float(input_ids[b][t]) * W} with
 * {@code W = [0.5, -1, 2]}, so with mean pooling and no scaling the vector of an input is the mean
 * of its vocabulary ids times {@code W}, hand-computable per input.</p>
 *
 * <p>{@code tiny-pooled.onnx} sums the token vectors of a row inside the graph and deliberately
 * ignores {@code attention_mask}, which makes the padding added here directly observable: the sum
 * of a padded row includes its pad ids. Combined with a vocabulary whose pad token is not id
 * {@code 0} that turns the sum into a readout of the tensor width one row actually ran at, which
 * is what {@link #testStrategiesRunAtDifferentTensorWidths()} and the sub-batch tests assert. A
 * real encoder masks the padded positions out instead, so for it the pad id is unobservable and
 * only the mask matters; those cases use the pad-at-zero vocabulary, where the in-graph sum is
 * unaffected by padding just as a mask-honoring graph would be.</p>
 */
class SentenceVectorsDLPaddedBatchTest {

  /**
   * Batch rows are assembled into wider tensors than a single-input call uses, so the model may
   * pick a different kernel blocking for the same arithmetic. Vectors are compared to this
   * tolerance rather than bit for bit.
   */
  private static final float DELTA = 1e-6f;

  private static final float[] W = {0.5f, -1f, 2f};

  /** Inputs of varied tokenized length whose mean-pooled vectors are all distinct. */
  private static final List<String> DISTINCT = List.of(
      "x",                  // [CLS]=7 [UNK]=2 [SEP]=3,   length 3, mean 4
      "hello hello",        // 7 4 4 3,                    length 4, mean 4.5
      "hello hello world",  // 7 4 4 5 3,                  length 5, mean 4.6
      "hello",              // 7 4 3,                      length 3, mean 14/3
      "hello world",        // 7 4 5 3,                    length 4, mean 4.75
      "world");             // 7 5 3,                      length 3, mean 5

  private static final float[] DISTINCT_MEANS = {4f, 4.5f, 4.6f, 14 / 3f, 4.75f, 5f};

  private static float[] scale(final float factor) {
    return new float[] {W[0] * factor, W[1] * factor, W[2] * factor};
  }

  // Copied out of the classpath rather than resolved in place, for the same reason as in
  // SentenceVectorsDLEmbedderTest: the resource may live inside a test-jar.
  private static File model(final Path dir, final String name) throws IOException {
    final Path file = dir.resolve(name);
    try (InputStream is = Objects.requireNonNull(SentenceVectorsDLPaddedBatchTest.class
        .getResourceAsStream("/opennlp/dl/vectors/" + name))) {
      Files.copy(is, file, StandardCopyOption.REPLACE_EXISTING);
    }
    return file.toFile();
  }

  /** The vocabulary of {@link SentenceVectorsDLEmbedderTest}, whose pad token is id {@code 0}. */
  private static File vocab(final Path dir) throws IOException {
    final Path file = dir.resolve("vocab.txt");
    Files.write(file, List.of("[PAD]", "unused1", "[UNK]", "[SEP]", "hello", "world",
        "unused2", "[CLS]"));
    return file.toFile();
  }

  /**
   * The same vocabulary with the pad token moved to id {@code 1}, so a padded position is
   * distinguishable from the {@code 0} an unwritten tensor position holds.
   */
  private static File padVocab(final Path dir) throws IOException {
    final Path file = dir.resolve("pad-vocab.txt");
    Files.write(file, List.of("unused0", "[PAD]", "[UNK]", "[SEP]", "hello", "world",
        "unused6", "[CLS]"));
    return file.toFile();
  }

  private static SentenceVectorsDL tinyVectors(final Path dir, final PaddingStrategy padding,
      final int maxLength) throws Exception {
    return new SentenceVectorsDL(model(dir, "tiny-vectors.onnx"), vocab(dir), true, Pooling.MEAN,
        false, maxLength, padding);
  }

  private static SentenceVectorsDL tinyVectors(final Path dir, final PaddingStrategy padding)
      throws Exception {
    return tinyVectors(dir, padding, SentenceVectorsDL.DEFAULT_MAX_LENGTH);
  }

  private static SentenceVectorsDL tinyPooled(final Path dir, final PaddingStrategy padding,
      final int maxLength, final File vocabulary) throws Exception {
    return new SentenceVectorsDL(model(dir, "tiny-pooled.onnx"), vocabulary, true, Pooling.MEAN,
        false, maxLength, padding);
  }

  private static String repeat(final String word, final int times) {
    final StringBuilder text = new StringBuilder(word.length() * times + times);
    for (int i = 0; i < times; i++) {
      if (i > 0) {
        text.append(' ');
      }
      text.append(word);
    }
    return text.toString();
  }

  /**
   * Inputs whose tokenized lengths differ widely, the shortest and the longest included, so a
   * single batch of all of them pads almost every row.
   */
  private static List<String> variedLengths() {
    final List<String> texts = new ArrayList<>(List.of("", " ", "\t\n ", "x", "hello", "world",
        "hello world", "hello world hello", "éèê", "中文",
        "𐐒́", "é", "é"));
    // No 1: repeat("hello world", 1) is already in the list above, and indexOf must be unique.
    for (final int words : new int[] {2, 3, 5, 8, 13, 21, 34, 55, 89, 144}) {
      texts.add(repeat("hello world", words));
    }
    return texts;
  }

  /** Every input of {@link #variedLengths()} crossed with every strategy. */
  private static Stream<Arguments> everyInputAndStrategy() {
    final List<Arguments> arguments = new ArrayList<>();
    for (final PaddingStrategy padding : PaddingStrategy.values()) {
      for (final String text : variedLengths()) {
        arguments.add(Arguments.of(padding, text));
      }
    }
    return arguments.stream();
  }

  /** Unicode inputs crossed with every strategy. */
  private static Stream<Arguments> unicodeAndStrategy() {
    final List<String> texts = List.of(
        "𐐒",                                       // U+10412, supplementary
        "𐐒́",                                 // supplementary plus combining acute
        "𐐒 𐐓 𐐔",
        "é",                                             // precomposed e acute
        "é",                                            // decomposed e acute
        "été",
        "été",
        "中文句子",                           // Han
        "العربية",         // Arabic
        "हिन्दी",               // Devanagari
        "Алексей");        // Cyrillic
    final List<Arguments> arguments = new ArrayList<>();
    for (final PaddingStrategy padding : PaddingStrategy.values()) {
      for (final String text : texts) {
        arguments.add(Arguments.of(padding, text));
      }
    }
    return arguments.stream();
  }

  /**
   * Under every strategy, each input of one mixed-length batch reproduces the vector it gets on
   * its own, within {@link #DELTA}. This is the invariant that padding changes no output vector.
   */
  @ParameterizedTest(name = "{0}: [{1}]")
  @MethodSource("everyInputAndStrategy")
  void testBatchedRowEqualsSingleEmbed(final PaddingStrategy padding, final String text,
      @TempDir final Path dir) throws Exception {
    final List<String> texts = variedLengths();
    try (SentenceVectorsDL vectors = tinyVectors(dir, padding)) {
      final float[][] batch = vectors.embedAll(texts);
      final int index = texts.indexOf(text);
      assertArrayEquals(vectors.embed(text), batch[index], DELTA,
          padding + ": batched row " + index + " differs from the single embed of [" + text + "]");
    }
  }

  /**
   * Under every strategy a vector is independent of how many inputs travel with it: alone, in a
   * batch of two, and in a large mixed batch all give the same vector.
   */
  @ParameterizedTest(name = "{0}: [{1}]")
  @MethodSource("everyInputAndStrategy")
  void testVectorIsIndependentOfBatchSize(final PaddingStrategy padding, final String text,
      @TempDir final Path dir) throws Exception {
    final List<String> mixed = variedLengths();
    try (SentenceVectorsDL vectors = tinyVectors(dir, padding)) {
      final float[] alone = vectors.embed(text);
      final String companion = repeat("hello world", 89);
      assertArrayEquals(alone, vectors.embedAll(List.of(text))[0], DELTA,
          padding + ": a batch of one");
      assertArrayEquals(alone, vectors.embedAll(List.of(text, companion))[0], DELTA,
          padding + ": a batch of two");
      assertArrayEquals(alone, vectors.embedAll(List.of(companion, text))[1], DELTA,
          padding + ": a batch of two, reversed");
      assertArrayEquals(alone, vectors.embedAll(mixed)[mixed.indexOf(text)], DELTA,
          padding + ": a batch of " + mixed.size());
    }
  }

  /**
   * The extremes together: the shortest and the longest input of the set in a batch of two, where
   * the short row is padded by the whole difference in length.
   */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testShortestAndLongestInTheSameBatch(final PaddingStrategy padding) throws Exception {
    final Path dir = Files.createTempDirectory("extremes");
    try (SentenceVectorsDL vectors = tinyVectors(dir, padding)) {
      final String shortest = "";
      final String longest = repeat("hello world", 144);
      final float[][] batch = vectors.embedAll(List.of(shortest, longest));
      assertArrayEquals(vectors.embed(shortest), batch[0], DELTA, "the shortest input");
      assertArrayEquals(vectors.embed(longest), batch[1], DELTA, "the longest input");
      // The two are not the same vector, so the assertions above are not trivially satisfied.
      assertFalse(Arrays.equals(batch[0], batch[1]));
    }
  }

  /**
   * All three strategies produce the same vectors for the same inputs. {@link
   * PaddingStrategy#EXACT_LENGTH} pads nothing, so its rows are the reference the padded rows of
   * the other two must match.
   */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testEveryStrategyAgreesWithTheUnpaddedOne(final PaddingStrategy padding,
      @TempDir final Path dir) throws Exception {
    final List<String> texts = variedLengths();
    try (SentenceVectorsDL reference = tinyVectors(dir, PaddingStrategy.EXACT_LENGTH);
         SentenceVectorsDL other = tinyVectors(dir, padding)) {
      final float[][] unpadded = reference.embedAll(texts);
      final float[][] actual = other.embedAll(texts);
      assertEquals(unpadded.length, actual.length);
      for (int i = 0; i < texts.size(); i++) {
        assertArrayEquals(unpadded[i], actual[i], DELTA,
            padding + ": row " + i + ", [" + texts.get(i) + "]");
      }
    }
  }

  /**
   * The strategies differ in the only place they may: the tensor shapes and the number of
   * inferences one call runs. A test that reads the vectors alone cannot tell them apart, so this
   * one reads the batch plan {@code embedAll} executes.
   *
   * <p>The tokenized lengths of {@link #DISTINCT} are 3, 4, 5, 3, 4, 3 in call order.</p>
   */
  @Test
  void testStrategiesShapeTheTensorsDifferently(@TempDir final Path dir) throws Exception {
    try (SentenceVectorsDL exact = tinyVectors(dir, PaddingStrategy.EXACT_LENGTH);
         SentenceVectorsDL longest = tinyVectors(dir, PaddingStrategy.LONGEST);
         SentenceVectorsDL fixed = tinyVectors(dir, PaddingStrategy.MAX_LENGTH, 8)) {

      // One inference per distinct length, in first-seen order, nothing padded.
      assertArrayEquals(new int[][] {{3, 3}, {2, 4}, {1, 5}}, exact.batchShapes(DISTINCT),
          "EXACT_LENGTH must run one inference per distinct tokenized length");

      // One inference for the whole call, at the longest row of it.
      assertArrayEquals(new int[][] {{6, 5}}, longest.batchShapes(DISTINCT),
          "LONGEST must run the whole call as one inference at the longest row");

      // One inference for the whole call, at the configured maximum whatever the rows hold.
      assertArrayEquals(new int[][] {{6, 8}}, fixed.batchShapes(DISTINCT),
          "MAX_LENGTH must run at the configured maximum");
    }
  }

  /**
   * The tensor width each strategy runs at, read out of the model rather than out of the plan.
   * {@code tiny-pooled.onnx} sums every position of a row whatever the mask says, so with the pad
   * token at id {@code 1} the in-graph sum of a row grows by exactly the number of positions it
   * was padded by.
   */
  @Test
  void testStrategiesRunAtDifferentTensorWidths() throws Exception {
    final Path dir = Files.createTempDirectory("widths");
    final File vocabulary = padVocab(dir);
    // "hello" = [CLS]=7 hello=4 [SEP]=3, sum 14 at width 3; "hello world" adds world=5, sum 19.
    final List<String> texts = List.of("hello", "hello world");
    try (SentenceVectorsDL exact = tinyPooled(dir, PaddingStrategy.EXACT_LENGTH, 8, vocabulary);
         SentenceVectorsDL longest = tinyPooled(dir, PaddingStrategy.LONGEST, 8, vocabulary);
         SentenceVectorsDL fixed = tinyPooled(dir, PaddingStrategy.MAX_LENGTH, 8, vocabulary)) {

      final float[][] unpadded = exact.embedAll(texts);
      assertArrayEquals(scale(14), unpadded[0], DELTA, "EXACT_LENGTH pads nothing");
      assertArrayEquals(scale(19), unpadded[1], DELTA, "EXACT_LENGTH pads nothing");

      final float[][] toLongest = longest.embedAll(texts);
      assertArrayEquals(scale(14 + 1), toLongest[0], DELTA,
          "LONGEST pads the 3-token row by one position at pad id 1");
      assertArrayEquals(scale(19), toLongest[1], DELTA, "the longest row is not padded");

      final float[][] toMax = fixed.embedAll(texts);
      assertArrayEquals(scale(14 + 5), toMax[0], DELTA,
          "MAX_LENGTH pads the 3-token row out to 8 positions");
      assertArrayEquals(scale(19 + 4), toMax[1], DELTA,
          "MAX_LENGTH pads the 4-token row out to 8 positions too");
    }
  }

  /**
   * {@link PaddingStrategy#MAX_LENGTH} truncates before it pads, so an input longer than the
   * configured maximum fills the fixed width exactly and does not widen it. The vector it yields
   * is the vector of its truncated form under any strategy.
   */
  @Test
  void testFixedWidthMeetsTruncation() throws Exception {
    final Path dir = Files.createTempDirectory("fixed-truncation");
    final File vocabulary = padVocab(dir);
    final String longText = repeat("hello world", 100);
    try (SentenceVectorsDL fixed = tinyPooled(dir, PaddingStrategy.MAX_LENGTH, 8, vocabulary);
         SentenceVectorsDL exact = tinyPooled(dir, PaddingStrategy.EXACT_LENGTH, 8, vocabulary)) {

      // Truncated to 8 with the final [SEP] kept: 7 + 4 + 5 + 4 + 5 + 4 + 5 + 3 = 37.
      assertArrayEquals(scale(37), fixed.embed(longText), DELTA,
          "a truncated row exactly fills the fixed width, so nothing is padded");
      assertArrayEquals(scale(37), exact.embed(longText), DELTA,
          "the truncation contract does not depend on the strategy");

      final List<String> texts = List.of("hello", longText, "world");
      final float[][] batch = fixed.embedAll(texts);
      assertArrayEquals(new int[][] {{3, 8}}, fixed.batchShapes(texts),
          "every row of the call is 8 wide, the truncated one included");
      assertArrayEquals(scale(14 + 5), batch[0], DELTA, "a short row beside a truncated one");
      assertArrayEquals(scale(37), batch[1], DELTA, "the truncated row, unpadded");
      assertArrayEquals(scale(15 + 5), batch[2], DELTA, "world = 7 + 5 + 3 = 15, plus 5 pads");
    }
  }

  /**
   * An input longer than the configured maximum is truncated to it, keeping the final
   * {@code [SEP]}, exactly as a single-input call truncates it, under every strategy.
   */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testInputLongerThanTheMaximumLength(final PaddingStrategy padding) throws Exception {
    final Path dir = Files.createTempDirectory("truncation");
    try (SentenceVectorsDL vectors = tinyVectors(dir, padding, 8)) {
      final String longText = repeat("hello world", 100);
      assertArrayEquals(scale(37 / 8f), vectors.embed(longText), DELTA);
      final List<String> texts = List.of("hello", longText, repeat("hello", 500), "world");
      final float[][] batch = vectors.embedAll(texts);
      assertArrayEquals(scale(14 / 3f), batch[0], DELTA, "a short row beside a truncated one");
      assertArrayEquals(scale(37 / 8f), batch[1], DELTA, "the truncated row");
      // [CLS] then 6 hello then [SEP]: 7 + 6 * 4 + 3 = 34.
      assertArrayEquals(scale(34 / 8f), batch[2], DELTA, "a second truncated row");
      assertArrayEquals(scale(5f), batch[3], DELTA, "a short row after two truncated ones");
    }
  }

  /**
   * Row {@code i} of the result is the vector of input {@code i}, under every strategy. The inputs
   * are ordered so that their expected vectors are all distinct and their tokenized lengths are
   * interleaved rather than ascending, so a result left in a batch's internal order would not
   * match.
   */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testOrderIsPreserved(final PaddingStrategy padding, @TempDir final Path dir)
      throws Exception {
    try (SentenceVectorsDL vectors = tinyVectors(dir, padding)) {
      final float[][] batch = vectors.embedAll(DISTINCT);
      assertEquals(DISTINCT.size(), batch.length);
      for (int i = 0; i < DISTINCT.size(); i++) {
        assertArrayEquals(scale(DISTINCT_MEANS[i]), batch[i], DELTA,
            padding + ": row " + i + " should be the vector of [" + DISTINCT.get(i) + "]");
      }
      // The same inputs reversed give the reversed rows, not the same rows.
      final List<String> reversed = new ArrayList<>(DISTINCT);
      Collections.reverse(reversed);
      final float[][] back = vectors.embedAll(reversed);
      for (int i = 0; i < reversed.size(); i++) {
        assertArrayEquals(scale(DISTINCT_MEANS[DISTINCT_MEANS.length - 1 - i]), back[i], DELTA,
            padding + ": reversed row " + i + " should be the vector of [" + reversed.get(i) + "]");
      }
    }
  }

  /** Duplicate inputs each get their own row, and all of them get the same vector. */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testDuplicateInputs(final PaddingStrategy padding, @TempDir final Path dir)
      throws Exception {
    try (SentenceVectorsDL vectors = tinyVectors(dir, padding)) {
      final List<String> texts = List.of("hello", "hello world", "hello", "hello world", "hello");
      final float[][] batch = vectors.embedAll(texts);
      assertEquals(5, batch.length);
      for (int i = 0; i < texts.size(); i++) {
        assertArrayEquals(scale("hello".equals(texts.get(i)) ? 14 / 3f : 4.75f), batch[i], DELTA,
            padding + ": row " + i + ", [" + texts.get(i) + "]");
      }
      assertNotSame(batch[0], batch[2], "duplicate inputs must not share one array");
      batch[0][0] = Float.NaN;
      assertFalse(Float.isNaN(batch[2][0]));
    }
  }

  /**
   * Boundary inputs a batch has to carry alongside ordinary text: no inputs at all, one input, an
   * empty string, whitespace only, and one character. Whitespace-only and empty input both
   * tokenize to the wrapped {@code [CLS] [SEP]}, so both pool to the mean of {@code 7} and
   * {@code 3}.
   */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testBoundaryInputs(final PaddingStrategy padding, @TempDir final Path dir)
      throws Exception {
    try (SentenceVectorsDL vectors = tinyVectors(dir, padding)) {
      assertEquals(0, vectors.embedAll(List.of()).length);
      assertEquals(0, vectors.batchShapes(List.of()).length,
          "an empty call must run no inference at all");
      assertArrayEquals(scale(14 / 3f), vectors.embedAll(List.of("hello"))[0], DELTA);
      final List<String> edges = List.of("", " ", "\t\r\n", " ", "x", "hello world");
      final float[][] batch = vectors.embedAll(edges);
      assertEquals(edges.size(), batch.length);
      assertArrayEquals(scale(5f), batch[0], DELTA, "the empty string");
      assertArrayEquals(scale(5f), batch[1], DELTA, "an ASCII space");
      assertArrayEquals(scale(5f), batch[2], DELTA, "ASCII whitespace only");
      assertArrayEquals(scale(5f), batch[3], DELTA, "a no-break space only");
      assertArrayEquals(scale(4f), batch[4], DELTA, "one character");
      assertArrayEquals(scale(4.75f), batch[5], DELTA, "ordinary text");
    }
  }

  /**
   * Unicode inputs of differing UTF-16 length share a batch with ASCII ones and each still
   * reproduces its single-input vector. Covered here: supplementary code points, one of them with
   * a combining mark, the precomposed and decomposed forms of an accented letter, and four
   * non-Latin scripts. The tiny vocabulary maps all of them to {@code [UNK]}, so this proves batch
   * behavior, not tokenization coverage; {@code SentenceVectorsDLPaddedBatchEval} covers the same
   * inputs against the real vocabulary.
   */
  @ParameterizedTest(name = "{0}: [{1}]")
  @MethodSource("unicodeAndStrategy")
  void testUnicodeInputsInAMixedBatch(final PaddingStrategy padding, final String text,
      @TempDir final Path dir) throws Exception {
    try (SentenceVectorsDL vectors = tinyVectors(dir, padding)) {
      final List<String> texts = List.of("hello", text, repeat("hello world", 21), "world");
      final float[][] batch = vectors.embedAll(texts);
      assertArrayEquals(vectors.embed(text), batch[1], DELTA,
          padding + ": [" + text + "] in a mixed batch");
      assertArrayEquals(scale(14 / 3f), batch[0], DELTA, "the row before it");
      assertArrayEquals(scale(5f), batch[3], DELTA, "the row after the long one");
    }
  }

  /**
   * The pad id is read from the vocabulary rather than assumed to be {@code 0}. With the pad token
   * at id {@code 1}, a row padded by one position sums to one more in
   * {@code tiny-pooled.onnx} than it does on its own; a hardcoded {@code 0} would leave the sum
   * unchanged. A RoBERTa-style vocabulary is read for {@code <pad>}, not {@code [PAD]}.
   */
  @Test
  void testPadIdComesFromTheVocabulary() throws Exception {
    final Path dir = Files.createTempDirectory("pad-id");
    try (SentenceVectorsDL vectors = tinyPooled(dir, PaddingStrategy.LONGEST,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH, padVocab(dir))) {
      assertArrayEquals(scale(14), vectors.embed("hello"), DELTA, "alone, unpadded");
      assertArrayEquals(scale(15), vectors.embedAll(List.of("hello", "hello world"))[0], DELTA,
          "one pad position at id 1 must add 1 to the in-graph sum");
      assertArrayEquals(scale(14 + 3),
          vectors.embedAll(List.of("hello", "hello world hello world"))[0], DELTA,
          "three pad positions add 3");
    }
    final Path roberta = dir.resolve("roberta.txt");
    Files.write(roberta, List.of("unused0", "<pad>", "<unk>", "</s>", "hello", "world",
        "unused6", "<s>"));
    try (SentenceVectorsDL vectors = tinyPooled(dir, PaddingStrategy.LONGEST,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH, roberta.toFile())) {
      assertArrayEquals(scale(15), vectors.embedAll(List.of("hello", "hello world"))[0], DELTA,
          "<s>=7 hello=4 </s>=3 padded with <pad>=1");
    }
  }

  /**
   * {@link SentenceVectorsDL#MAX_BATCH_TOKEN_POSITIONS} bounds the token positions of one
   * inference, so a call that would exceed it is cut into sub-batches. The cut is observed rather
   * than assumed: {@code tiny-pooled.onnx} sums the pad ids too, so a short row reveals how wide
   * the sub-batch it ran in was.
   *
   * <p>Under {@link PaddingStrategy#LONGEST} the rows run in ascending length order, so {@code n}
   * rows of 3 tokens plus one of 4 share one inference while {@code (n + 1) * 4} is within the
   * bound, that is for {@code n} up to {@code 4095}. At {@code n = 4095} the batch is exactly the
   * bound and every short row is padded to 4. At {@code n = 4096} it is one row over, so the short
   * rows run in a sub-batch of their own at width 3 and are not padded at all.</p>
   */
  @Test
  void testSubBatchesAreCutAtTheTokenPositionCap() throws Exception {
    final Path dir = Files.createTempDirectory("cap");
    final int atCap = SentenceVectorsDL.MAX_BATCH_TOKEN_POSITIONS / 4 - 1;
    try (SentenceVectorsDL vectors = tinyPooled(dir, PaddingStrategy.LONGEST,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH, padVocab(dir))) {

      final List<String> exactly = new ArrayList<>(Collections.nCopies(atCap, "hello"));
      exactly.add("hello world");
      assertArrayEquals(new int[][] {{atCap + 1, 4}}, vectors.batchShapes(exactly),
          "exactly at the bound the whole call is one inference");
      final float[][] single = vectors.embedAll(exactly);
      assertEquals(atCap + 1, single.length);
      assertArrayEquals(scale(15), single[0], DELTA,
          "at the bound every short row shares one inference with the 4-token row, padded to 4");
      assertArrayEquals(scale(15), single[atCap - 1], DELTA, "the last short row");
      assertArrayEquals(scale(19), single[atCap], DELTA, "the long row");

      final List<String> over = new ArrayList<>(Collections.nCopies(atCap + 1, "hello"));
      over.add("hello world");
      assertArrayEquals(new int[][] {{atCap + 1, 3}, {1, 4}}, vectors.batchShapes(over),
          "one row over the bound the call is cut into two inferences");
      final float[][] chunked = vectors.embedAll(over);
      assertEquals(atCap + 2, chunked.length);
      assertArrayEquals(scale(14), chunked[0], DELTA,
          "one row over the bound the short rows run in a sub-batch of their own, unpadded");
      assertArrayEquals(scale(14), chunked[atCap], DELTA, "the last short row");
      assertArrayEquals(scale(19), chunked[atCap + 1], DELTA, "the long row, alone");
    }
  }

  /**
   * The bound applies to the other two strategies as well, so no strategy can build an unbounded
   * tensor. Read from the plan, since the row counts involved are large.
   */
  @Test
  void testTheCapBoundsEveryStrategy(@TempDir final Path dir) throws Exception {
    final int rows = 6000;
    final List<String> many = Collections.nCopies(rows, "hello");
    try (SentenceVectorsDL exact = tinyVectors(dir, PaddingStrategy.EXACT_LENGTH);
         SentenceVectorsDL fixed = tinyVectors(dir, PaddingStrategy.MAX_LENGTH, 512)) {

      // One tokenized length, 3 tokens, so 16384 / 3 = 5461 rows fit one inference.
      final int perExactBatch = SentenceVectorsDL.MAX_BATCH_TOKEN_POSITIONS / 3;
      assertArrayEquals(new int[][] {{perExactBatch, 3}, {rows - perExactBatch, 3}},
          exact.batchShapes(many), "EXACT_LENGTH must split a group larger than the bound");

      // Every row is 512 wide, so 16384 / 512 = 32 rows fit one inference.
      final int perFixedBatch = SentenceVectorsDL.MAX_BATCH_TOKEN_POSITIONS / 512;
      final int[][] fixedShapes = fixed.batchShapes(many);
      assertEquals((rows + perFixedBatch - 1) / perFixedBatch, fixedShapes.length);
      int counted = 0;
      for (int b = 0; b < fixedShapes.length; b++) {
        assertEquals(512, fixedShapes[b][1], "MAX_LENGTH runs every inference at 512");
        // Full batches but the last, which holds the remainder.
        assertEquals(b == fixedShapes.length - 1 ? rows % perFixedBatch : perFixedBatch,
            fixedShapes[b][0], "batch " + b);
        counted += fixedShapes[b][0];
      }
      assertEquals(rows, counted, "every input must appear in exactly one batch");
    }
  }

  /**
   * A single input whose own tokenized length exceeds the bound still runs, in a sub-batch of one,
   * and the short inputs beside it are not padded out to its width.
   */
  @Test
  void testOneInputWiderThanTheCap() throws Exception {
    final Path dir = Files.createTempDirectory("wide");
    final int wide = SentenceVectorsDL.MAX_BATCH_TOKEN_POSITIONS + 8;
    try (SentenceVectorsDL vectors = tinyPooled(dir, PaddingStrategy.LONGEST, wide,
        padVocab(dir))) {
      final String huge = repeat("hello", wide);
      final List<String> texts = List.of("hello", huge, "world");
      assertArrayEquals(new int[][] {{2, 3}, {1, wide}}, vectors.batchShapes(texts),
          "the oversized row runs alone and does not widen the short ones");
      final float[][] batch = vectors.embedAll(texts);
      assertEquals(3, batch.length);
      assertArrayEquals(scale(14), batch[0], DELTA, "the first short row, unpadded");
      assertArrayEquals(scale(15), batch[2], DELTA, "the second short row, unpadded");
      // [CLS] + (wide - 2) hello + [SEP] = 7 + (wide - 2) * 4 + 3.
      assertArrayEquals(scale(7 + (wide - 2) * 4 + 3), batch[1], 1e-2f, "the oversized row");
    }
  }

  /** {@code null} is rejected at the public boundary, with the messages the class already uses. */
  @Test
  void testNullHandling(@TempDir final Path dir) throws Exception {
    try (SentenceVectorsDL vectors = tinyVectors(dir, PaddingStrategy.LONGEST)) {
      assertEquals("texts must not be null", assertThrows(IllegalArgumentException.class,
          () -> vectors.embedAll(null)).getMessage());
      assertEquals("texts[1] must not be null", assertThrows(IllegalArgumentException.class,
          () -> vectors.embedAll(Arrays.asList("hello", null, "world"))).getMessage());
      assertEquals("texts[0] must not be null", assertThrows(IllegalArgumentException.class,
          () -> vectors.embedAll(Collections.singletonList(null))).getMessage());
      assertEquals("text must not be null", assertThrows(IllegalArgumentException.class,
          () -> vectors.embed(null)).getMessage());
      assertEquals("sentence must not be null", assertThrows(IllegalArgumentException.class,
          () -> vectors.getVectors(null)).getMessage());
    }
    assertEquals("padding must not be null", assertThrows(IllegalArgumentException.class,
        () -> new SentenceVectorsDL(model(dir, "tiny-vectors.onnx"), vocab(dir), true,
            Pooling.MEAN, false, SentenceVectorsDL.DEFAULT_MAX_LENGTH, null)).getMessage());
  }

  /**
   * Nothing specified means {@link PaddingStrategy#EXACT_LENGTH}: the constructors that predate
   * the option keep the behavior they had, one inference per distinct tokenized length and no
   * padding.
   */
  @Test
  void testTheDefaultIsExactLengthGrouping(@TempDir final Path dir) throws Exception {
    assertEquals(PaddingStrategy.EXACT_LENGTH, SentenceVectorsDL.DEFAULT_PADDING);
    final File graph = model(dir, "tiny-vectors.onnx");
    final File vocabulary = vocab(dir);
    try (SentenceVectorsDL twoArgs = new SentenceVectorsDL(graph, vocabulary);
         SentenceVectorsDL threeArgs = new SentenceVectorsDL(graph, vocabulary, true);
         SentenceVectorsDL sixArgs = new SentenceVectorsDL(graph, vocabulary, true, Pooling.MEAN,
             false, SentenceVectorsDL.DEFAULT_MAX_LENGTH)) {
      final int[][] expected = {{3, 3}, {2, 4}, {1, 5}};
      assertArrayEquals(expected, twoArgs.batchShapes(DISTINCT));
      assertArrayEquals(expected, threeArgs.batchShapes(DISTINCT));
      assertArrayEquals(expected, sixArgs.batchShapes(DISTINCT));
    }
  }

  /**
   * A vocabulary without a padding token cannot pad, and says so at construction rather than on
   * the first call that happens to mix lengths. {@link PaddingStrategy#EXACT_LENGTH} needs no
   * padding token and still loads.
   */
  @Test
  void testVocabularyWithoutAPadToken(@TempDir final Path dir) throws Exception {
    final File graph = model(dir, "tiny-vectors.onnx");
    final Path bert = dir.resolve("no-pad.txt");
    Files.write(bert, List.of("unused0", "unused1", "[UNK]", "[SEP]", "hello", "world",
        "unused6", "[CLS]"));
    for (final PaddingStrategy padding : List.of(PaddingStrategy.LONGEST,
        PaddingStrategy.MAX_LENGTH)) {
      assertEquals("The vocabulary has no padding token '[PAD]', so rows of different lengths "
              + "cannot be padded into one batch.",
          assertThrows(IllegalArgumentException.class,
              () -> new SentenceVectorsDL(graph, bert.toFile(), true, Pooling.MEAN, false,
                  SentenceVectorsDL.DEFAULT_MAX_LENGTH, padding)).getMessage(),
          padding.toString());
    }
    try (SentenceVectorsDL vectors = new SentenceVectorsDL(graph, bert.toFile(), true,
        Pooling.MEAN, false, SentenceVectorsDL.DEFAULT_MAX_LENGTH,
        PaddingStrategy.EXACT_LENGTH)) {
      assertArrayEquals(scale(14 / 3f), vectors.embedAll(List.of("hello"))[0], DELTA);
    }

    final Path roberta = dir.resolve("roberta-no-pad.txt");
    Files.write(roberta, List.of("unused0", "unused1", "<unk>", "</s>", "hello", "world",
        "unused6", "<s>"));
    assertEquals("The vocabulary has no padding token '<pad>', so rows of different lengths "
            + "cannot be padded into one batch.",
        assertThrows(IllegalArgumentException.class,
            () -> new SentenceVectorsDL(graph, roberta.toFile(), true, Pooling.MEAN, false,
                SentenceVectorsDL.DEFAULT_MAX_LENGTH, PaddingStrategy.LONGEST)).getMessage());
  }

  /** No strategy changes the reported dimension, which is model metadata. */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testDimensionIsUnchanged(final PaddingStrategy padding, @TempDir final Path dir)
      throws Exception {
    try (SentenceVectorsDL vectors = tinyVectors(dir, padding);
         SentenceVectorsDL pooled = tinyPooled(dir, padding,
             SentenceVectorsDL.DEFAULT_MAX_LENGTH, vocab(dir))) {
      assertEquals(3, vectors.dimension());
      assertEquals(3, pooled.dimension());
      // A batch does not change it, and every row has it.
      for (final float[] vector : vectors.embedAll(variedLengths())) {
        assertEquals(3, vector.length);
      }
      assertEquals(3, vectors.dimension());
      assertEquals(3, vectors.embed("hello").length);
    }
  }

  /**
   * An in-graph {@code sentence_embedding} output is still selected by name and still used as it
   * is when the batch pads. With the pad token at id {@code 0} the sum
   * {@code tiny-pooled.onnx} computes is unaffected by padding, which is the behavior a
   * mask-honoring encoder has for any pad id.
   */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testPooledOutputUnderPadding(final PaddingStrategy padding, @TempDir final Path dir)
      throws Exception {
    try (SentenceVectorsDL vectors = tinyPooled(dir, padding,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH, vocab(dir))) {
      final List<String> texts = List.of("hello", "hello world", "x", "hello world hello",
          "world");
      final float[][] batch = vectors.embedAll(texts);
      for (int i = 0; i < texts.size(); i++) {
        assertArrayEquals(vectors.embed(texts.get(i)), batch[i], DELTA,
            padding + ": row " + i + ", [" + texts.get(i) + "]");
      }
      assertArrayEquals(scale(14), batch[0], DELTA);
      assertArrayEquals(scale(19), batch[1], DELTA);
    }
  }

  /** Vectors are still scaled to unit length when the row they came from was padded. */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testNormalizedVectorsUnderPadding(final PaddingStrategy padding, @TempDir final Path dir)
      throws Exception {
    try (SentenceVectorsDL vectors = new SentenceVectorsDL(model(dir, "tiny-vectors.onnx"),
        vocab(dir), true, Pooling.MEAN, true, SentenceVectorsDL.DEFAULT_MAX_LENGTH, padding)) {
      for (final float[] vector : vectors.embedAll(variedLengths())) {
        double squares = 0;
        for (final float value : vector) {
          squares += (double) value * value;
        }
        assertEquals(1.0, Math.sqrt(squares), 1e-5);
      }
    }
  }

  /** {@link Pooling#CLS} reads position {@code 0}, which padding never touches. */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testClsPoolingUnderPadding(final PaddingStrategy padding, @TempDir final Path dir)
      throws Exception {
    try (SentenceVectorsDL vectors = new SentenceVectorsDL(model(dir, "tiny-vectors.onnx"),
        vocab(dir), true, Pooling.CLS, false, SentenceVectorsDL.DEFAULT_MAX_LENGTH, padding)) {
      final List<String> texts = variedLengths();
      final float[][] batch = vectors.embedAll(texts);
      for (int i = 0; i < texts.size(); i++) {
        assertArrayEquals(scale(7), batch[i], DELTA,
            padding + ": row " + i + ", [" + texts.get(i) + "]");
      }
    }
  }
}
