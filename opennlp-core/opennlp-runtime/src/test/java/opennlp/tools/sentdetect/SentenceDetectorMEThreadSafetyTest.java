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

package opennlp.tools.sentdetect;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.util.Span;

/**
 * Thread-safety tests for {@link SentenceDetectorME}.
 * <p>
 * Verifies that a single {@link SentenceDetectorME} instance can be safely shared
 * across multiple threads without data corruption or exceptions.
 */
public class SentenceDetectorMEThreadSafetyTest extends AbstractSentenceDetectorTest {

  private static SentenceModel sentModel;

  @BeforeAll
  static void setup() throws IOException {
    Dictionary abbreviationDict = loadAbbDictionary(Locale.ENGLISH);
    SentenceDetectorFactory factory = new SentenceDetectorFactory(
        "eng", true, abbreviationDict, null);
    sentModel = train(factory, Locale.ENGLISH);
    Assertions.assertNotNull(sentModel);
  }

  /**
   * Share ONE SentenceDetectorME instance across multiple threads.
   * Each thread detects sentences in different text and verifies
   * that results correspond to its own input.
   */
  @Test
  void testConcurrentSentenceDetection() throws Exception {
    final int threadCount = 10;
    final int iterationsPerThread = 200;
    final SentenceDetectorME detector = new SentenceDetectorME(sentModel);

    String[] inputs = {
        "This is a test. There are many tests, this is the second.",
        "Hello world. How are you? I am fine.",
        "One sentence only.",
        "First sentence. Second sentence. Third sentence.",
        "Is this a question? Yes it is. Really?",
        "The quick brown fox. The lazy dog. Both are fine.",
        "Mr. Smith went to Washington. He arrived on Monday.",
        "Testing one two three. Four five six. Seven eight nine ten.",
        "A short one. And another. Plus one more.",
        "End with exclamation! And question? Then period."
    };

    ExecutorService executor = Executors.newFixedThreadPool(threadCount);
    CountDownLatch startLatch = new CountDownLatch(1);
    AtomicInteger errorCount = new AtomicInteger(0);
    List<Future<?>> futures = new ArrayList<>();

    for (int t = 0; t < threadCount; t++) {
      final String input = inputs[t % inputs.length];
      futures.add(executor.submit(() -> {
        try {
          startLatch.await(); // all threads start at the same time
          for (int i = 0; i < iterationsPerThread; i++) {
            // Test sentDetect (String[])
            String[] sentences = detector.sentDetect(input);
            Assertions.assertNotNull(sentences, "sentences should not be null");
            Assertions.assertTrue(sentences.length > 0,
                "Should detect at least one sentence for: " + input);

            // Verify each sentence is a substring of the input
            for (String sentence : sentences) {
              Assertions.assertNotNull(sentence, "Individual sentence should not be null");
              Assertions.assertTrue(input.contains(sentence),
                  "Sentence '" + sentence + "' should be a substring of input '" + input + "'");
            }

            // Test sentPosDetect (Span[])
            Span[] spans = detector.sentPosDetect(input);
            Assertions.assertNotNull(spans, "spans should not be null");

            // Verify spans are within bounds
            for (Span span : spans) {
              if (span != null) {
                Assertions.assertTrue(span.getStart() >= 0,
                    "Span start should be >= 0");
                Assertions.assertTrue(span.getEnd() <= input.length(),
                    "Span end should be <= input length");
              }
            }
          }
        } catch (Exception e) {
          errorCount.incrementAndGet();
          throw new RuntimeException("Thread failed: " + e.getMessage(), e);
        }
      }));
    }

    // Release all threads at once to maximize contention
    startLatch.countDown();

    // Wait for all threads to complete
    for (Future<?> future : futures) {
      future.get(60, TimeUnit.SECONDS);
    }

    executor.shutdown();
    Assertions.assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS),
        "Executor should terminate cleanly");
    Assertions.assertEquals(0, errorCount.get(),
        "No thread should have encountered errors");
  }

  /**
   * Verifies that concurrent sentPosDetect() calls do not cause
   * NullPointerExceptions, IndexOutOfBoundsExceptions, or data corruption.
   */
  @Test
  void testConcurrentSentPosDetectNoExceptions() throws Exception {
    final int threadCount = 8;
    final int iterationsPerThread = 300;
    final SentenceDetectorME detector = new SentenceDetectorME(sentModel);

    ExecutorService executor = Executors.newFixedThreadPool(threadCount);
    CountDownLatch startLatch = new CountDownLatch(1);
    List<Future<?>> futures = new ArrayList<>();

    String[] texts = {
        "Simple test.",
        "Two sentences here. And here.",
        "Question? Answer. More!",
        "",
        "   ",
        "One. Two. Three. Four. Five.",
        "A single sentence with no period",
        "Dr. Smith and Mr. Jones met. They talked."
    };

    for (int t = 0; t < threadCount; t++) {
      final String text = texts[t % texts.length];
      futures.add(executor.submit(() -> {
        try {
          startLatch.await();
          for (int i = 0; i < iterationsPerThread; i++) {
            String[] sentences = detector.sentDetect(text);
            Assertions.assertNotNull(sentences);

            Span[] spans = detector.sentPosDetect(text);
            Assertions.assertNotNull(spans);

            // For non-empty, non-whitespace input, we should get results
            if (!text.isBlank()) {
              Assertions.assertTrue(sentences.length > 0,
                  "Should detect sentences for non-blank input: '" + text + "'");
            }
          }
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
        }
      }));
    }

    startLatch.countDown();

    for (Future<?> future : futures) {
      future.get(60, TimeUnit.SECONDS);
    }

    executor.shutdown();
    Assertions.assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
  }
}
