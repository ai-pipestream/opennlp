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
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import opennlp.dl.InferenceOptions;
import opennlp.tools.eval.AbstractEvalTest;

/**
 * The CUDA execution provider of {@link SentenceVectorsDL} against the real sentence-transformers
 * MiniLM model, which is the only way to show that moving inference to the GPU does not move the
 * vectors.
 *
 * <p>Before the {@link InferenceOptions} constructor existed, {@link SentenceVectorsDL} built its
 * session with hardcoded default options, so {@code opennlp-dl-gpu} gave embeddings no
 * acceleration at all and said nothing about it. The unit tests in {@code opennlp-dl} pin down the
 * plumbing on a three-dimensional toy graph; only the real 384-dimensional encoder, with its
 * attention and layer norms, can show that the CUDA kernels agree with the CPU kernels to
 * {@link #TOLERANCE}.</p>
 *
 * <p>Every case is skipped unless a CUDA device can actually be used, which needs the
 * {@code onnxruntime_gpu} runtime in place of the CPU-only {@code onnxruntime} this module
 * normally resolves. One way to arrange that without changing the build is to put the GPU runtime
 * ahead of the CPU one and point ONNX Runtime at natives unpacked from the same artifact:</p>
 *
 * <pre>{@code
 * unzip -o -q -j onnxruntime_gpu-<version>.jar 'ai/onnxruntime/native/linux-x64/*' -d /tmp/ortgpu
 * JAVA_TOOL_OPTIONS="-Xbootclasspath/a:/path/onnxruntime_gpu-<version>.jar \
 *     -Donnxruntime.native.path=/tmp/ortgpu" \
 *   ./mvnw -Peval-tests -pl opennlp-eval-tests test \
 *     -Dtest=SentenceVectorsDLExecutionProviderEval -DOPENNLP_DATA_DIR=...
 * }</pre>
 *
 * <p>Mixing the two artifacts is what has to be avoided: the CUDA provider library of one ONNX
 * Runtime build will not link against the {@code libonnxruntime.so} of another, and the failure
 * reads as a missing CUDA installation.</p>
 */
public class SentenceVectorsDLExecutionProviderEval extends AbstractEvalTest {

  /**
   * The stated bound on how far one component of a vector computed on the GPU may sit from the same
   * component computed on the CPU. The two providers run different kernels over the same
   * {@code float} weights, accumulate in a different order and block differently, so they are not
   * expected to agree bit for bit. Over these inputs, with onnxruntime 1.27.0 on an RTX 4080 SUPER,
   * the measured worst component difference is {@code 1.9e-4} on unit-length vectors, and it
   * is the same order of magnitude whether or not any padding is involved, so it is the providers
   * and not the batching. The bound is set an order of magnitude above what was measured.
   */
  private static final float TOLERANCE = 2e-3f;

  /**
   * The stated bound on the angle between a vector computed on the GPU and the same vector computed
   * on the CPU. This is the assertion that carries the meaning: component differences of {@code
   * 2e-4} could still be a different embedding, whereas a cosine similarity this close to
   * {@code 1} says the two vectors are the same direction to far better precision than any
   * retrieval or clustering step downstream can distinguish. The measured worst case over these
   * inputs is {@code 5.1e-7}.
   */
  private static final double COSINE_TOLERANCE = 1e-5;

  private static final String MODEL = "onnx/sentence-transformers/model.onnx";
  private static final String VOCABULARY = "onnx/sentence-transformers/vocab.txt";
  private static final String CORPUS = "leipzig/eng_news_2010_300K-sentences.txt";

  private static final int DIMENSION = 384;

  /** How many natural sentences the comparison runs over. */
  private static final int CORPUS_SENTENCES = 48;

  private static File model() throws IOException {
    return new File(getOpennlpDataDir(), MODEL);
  }

  private static File vocabulary() throws IOException {
    return new File(getOpennlpDataDir(), VOCABULARY);
  }

  private static InferenceOptions cpu() {
    return new InferenceOptions();
  }

  private static InferenceOptions cuda() {
    final InferenceOptions options = new InferenceOptions();
    options.setGpu(true);
    return options;
  }

  private static SentenceVectorsDL vectors(final PaddingStrategy padding,
      final InferenceOptions inferenceOptions) throws Exception {
    return new SentenceVectorsDL(model(), vocabulary(), true, Pooling.MEAN, true,
        SentenceVectorsDL.DEFAULT_MAX_LENGTH, padding, inferenceOptions);
  }

  /**
   * {@return whether a CUDA session can actually be created on device {@code 0} here}
   *
   * <p>The environment is fetched first because ONNX Runtime cannot load a shared execution
   * provider library before its default logger exists, and it remembers the failure.</p>
   */
  private static boolean cudaUsable() {
    if (!OrtEnvironment.getAvailableProviders().contains(OrtProvider.CUDA)) {
      return false;
    }
    OrtEnvironment.getEnvironment();
    try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
      options.addCUDA(0);
      return true;
    } catch (final OrtException | RuntimeException e) {
      return false;
    }
  }

  /**
   * Natural sentences of widely varying length from the Leipzig English news corpus, plus short
   * and degenerate inputs, so that the shortest and the longest input share a batch.
   */
  private static List<String> inputs() throws IOException {
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
    texts.add("");
    texts.add("a");
    texts.add(String.join(" ", Collections.nCopies(400, "the quick brown fox jumped")));
    return texts;
  }

  /**
   * {@return the largest absolute difference between two sets of vectors}
   *
   * @param expected The vectors of the reference provider.
   * @param actual The vectors of the provider under test.
   */
  private static float deviation(final float[][] expected, final float[][] actual) {
    Assertions.assertEquals(expected.length, actual.length, "vector counts differ");
    float worst = 0;
    for (int i = 0; i < expected.length; i++) {
      Assertions.assertEquals(DIMENSION, actual[i].length, "vector " + i + " has the wrong width");
      for (int d = 0; d < DIMENSION; d++) {
        worst = Math.max(worst, Math.abs(expected[i][d] - actual[i][d]));
      }
    }
    return worst;
  }

  /**
   * {@return the largest angular separation between corresponding vectors, as {@code 1} minus the
   * smallest cosine similarity}
   *
   * @param expected The vectors of the reference provider.
   * @param actual The vectors of the provider under test.
   */
  private static double angularDeviation(final float[][] expected, final float[][] actual) {
    double worst = 0;
    for (int i = 0; i < expected.length; i++) {
      double dot = 0;
      double leftSquares = 0;
      double rightSquares = 0;
      for (int d = 0; d < DIMENSION; d++) {
        dot += (double) expected[i][d] * actual[i][d];
        leftSquares += (double) expected[i][d] * expected[i][d];
        rightSquares += (double) actual[i][d] * actual[i][d];
      }
      final double norms = Math.sqrt(leftSquares) * Math.sqrt(rightSquares);
      // A zero vector cannot occur here: every input is wrapped in [CLS] and [SEP] and scaled to
      // unit length, so the norms are around 1.
      Assertions.assertTrue(norms > 0, "vector " + i + " has zero length");
      worst = Math.max(worst, 1 - dot / norms);
    }
    return worst;
  }

  /**
   * Asserts that two sets of vectors agree, by component and by direction, and reports both
   * measurements so a run records what the providers actually did rather than only that it passed.
   *
   * @param what The case being compared, for the failure message.
   * @param expected The vectors of the reference provider.
   * @param actual The vectors of the provider under test.
   */
  private static void assertSameVectors(final String what, final float[][] expected,
      final float[][] actual) {
    final float worst = deviation(expected, actual);
    final double worstAngle = angularDeviation(expected, actual);
    System.out.println("# " + what + " worstComponentDifference=" + worst
        + " worstOneMinusCosine=" + worstAngle);
    Assertions.assertTrue(worst <= TOLERANCE, what + " deviates from the CPU by " + worst
        + ", above the stated tolerance " + TOLERANCE);
    Assertions.assertTrue(worstAngle <= COSINE_TOLERANCE, what + " turns the vector by "
        + worstAngle + " in one minus cosine, above the stated tolerance " + COSINE_TOLERANCE);
  }

  /**
   * The vectors do not depend on the execution provider. This is the assertion that matters most:
   * a GPU that ran but produced different numbers would be worse than one that never ran.
   */
  @Test
  void testGpuAndCpuAgreeOnEveryVector() throws Exception {
    Assumptions.assumeTrue(cudaUsable(), "a usable CUDA device is needed");
    final List<String> texts = inputs();
    final float[][] onCpu;
    final float[][] onGpu;
    try (SentenceVectorsDL cpu = vectors(PaddingStrategy.EXACT_LENGTH, cpu())) {
      onCpu = cpu.embedAll(texts);
    }
    try (SentenceVectorsDL gpu = vectors(PaddingStrategy.EXACT_LENGTH, cuda())) {
      onGpu = gpu.embedAll(texts);
      Assertions.assertEquals(DIMENSION, gpu.dimension());
    }
    assertSameVectors("the GPU against the CPU", onCpu, onGpu);
  }

  /**
   * Every {@link PaddingStrategy} composes with the GPU. The reference is a CPU instance under
   * {@link PaddingStrategy#EXACT_LENGTH}, which pads nothing, so a padding bug cannot cancel out
   * against an identically padded reference.
   *
   * @param padding The strategy under test.
   */
  @ParameterizedTest
  @EnumSource(PaddingStrategy.class)
  void testEveryPaddingStrategyRunsOnTheGpu(final PaddingStrategy padding) throws Exception {
    Assumptions.assumeTrue(cudaUsable(), "a usable CUDA device is needed");
    final List<String> texts = inputs();
    final float[][] reference;
    try (SentenceVectorsDL cpu = vectors(PaddingStrategy.EXACT_LENGTH, cpu())) {
      reference = cpu.embedAll(texts);
    }
    try (SentenceVectorsDL gpu = vectors(padding, cuda())) {
      assertSameVectors(padding + " on the GPU against the unpadded CPU reference", reference,
          gpu.embedAll(texts));
    }
  }

  /**
   * A device id no card answers to fails rather than falling back to another card or to the CPU.
   * Silent fallback is the failure mode this whole change exists to remove, so it is asserted
   * against the real model too.
   */
  @Test
  void testUnusableDeviceIdFailsRatherThanFallingBack() throws Exception {
    Assumptions.assumeTrue(cudaUsable(), "a usable CUDA device is needed");
    final InferenceOptions options = cuda();
    options.setGpuDeviceId(99);
    final OrtException failure = Assertions.assertThrows(OrtException.class,
        () -> vectors(PaddingStrategy.EXACT_LENGTH, options));
    Assertions.assertTrue(failure.getMessage().contains("device"),
        "the failure should name the device: " + failure.getMessage());
  }

}
