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

package opennlp.tools.tokenize;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import opennlp.tools.util.Span;

/**
 * Thread-safety tests for {@link TokenizerME}.
 * <p>
 * Verifies that a single {@link TokenizerME} instance can be safely shared
 * across multiple threads without data corruption or exceptions.
 */
public class TokenizerMEThreadSafetyTest {

  private static TokenizerModel model;

  @BeforeAll
  static void setup() throws IOException {
    model = TokenizerTestUtil.createMaxentTokenModel();
    Assertions.assertNotNull(model);
  }

  /**
   * Share ONE TokenizerME instance across multiple threads.
   * Each thread tokenizes different text and verifies that the returned
   * tokens correspond to its own input (no cross-thread data bleeding).
   */
  @Test
  void testConcurrentTokenization() throws Exception {
    final int threadCount = 10;
    final int iterationsPerThread = 200;
    final TokenizerME tokenizer = new TokenizerME(model);

    // Each thread gets a distinct sentence to tokenize
    String[] inputs = {
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, this is a test.",
        "Sounds like it's not properly thought through!",
        "One two three four five.",
        "Testing, testing, one two three.",
        "Apache OpenNLP is a machine learning toolkit.",
        "The year is 2024, and things are changing.",
        "It was the best of times, it was the worst of times.",
        "To be or not to be, that is the question.",
        "All good things must come to an end."
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
            // Test tokenize (String[])
            String[] tokens = tokenizer.tokenize(input);
            Assertions.assertNotNull(tokens, "tokens should not be null");
            Assertions.assertTrue(tokens.length > 0,
                "Should produce at least one token for input: " + input);

            // Verify each token is actually a substring of the input
            for (String token : tokens) {
              Assertions.assertNotNull(token, "Individual token should not be null");
              Assertions.assertTrue(input.contains(token),
                  "Token '" + token + "' should be a substring of input '" + input + "'");
            }

            // Test tokenizePos (Span[])
            Span[] spans = tokenizer.tokenizePos(input);
            Assertions.assertNotNull(spans, "spans should not be null");
            Assertions.assertEquals(tokens.length, spans.length,
                "tokenize and tokenizePos should return same number of tokens");

            // Verify each span is within bounds and covers the correct text
            for (Span span : spans) {
              Assertions.assertNotNull(span, "Individual span should not be null");
              Assertions.assertTrue(span.getStart() >= 0, "Span start should be >= 0");
              Assertions.assertTrue(span.getEnd() <= input.length(),
                  "Span end should be <= input length");
              Assertions.assertTrue(span.getStart() < span.getEnd(),
                  "Span start should be < span end");
              String covered = input.substring(span.getStart(), span.getEnd());
              Assertions.assertTrue(input.contains(covered),
                  "Span-covered text should be within the input");
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
      future.get(30, TimeUnit.SECONDS);
    }

    executor.shutdown();
    Assertions.assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS),
        "Executor should terminate cleanly");
    Assertions.assertEquals(0, errorCount.get(),
        "No thread should have encountered errors");
  }

  /**
   * Verifies that concurrent tokenizePos() calls do not cause
   * NullPointerExceptions or ArrayIndexOutOfBoundsExceptions,
   * which would indicate shared mutable state corruption.
   */
  @Test
  void testConcurrentTokenizePosNoExceptions() throws Exception {
    final int threadCount = 8;
    final int iterationsPerThread = 500;
    final TokenizerME tokenizer = new TokenizerME(model);

    ExecutorService executor = Executors.newFixedThreadPool(threadCount);
    CountDownLatch startLatch = new CountDownLatch(1);
    List<Future<?>> futures = new ArrayList<>();

    String[] texts = {
        "year,",
        "it,",
        "yes,",
        "test, test, test",
        "Hello, world!",
        "A B C D E F G H",
        "One. Two. Three.",
        "x,"
    };

    for (int t = 0; t < threadCount; t++) {
      final String text = texts[t % texts.length];
      futures.add(executor.submit(() -> {
        try {
          startLatch.await();
          for (int i = 0; i < iterationsPerThread; i++) {
            Span[] spans = tokenizer.tokenizePos(text);
            Assertions.assertNotNull(spans);
            for (Span span : spans) {
              Assertions.assertNotNull(span,
                  "No span should be null for input: " + text);
            }
            // Reconstruct text from spans and verify
            String[] tokensFromSpans = Span.spansToStrings(spans, text);
            for (String tok : tokensFromSpans) {
              Assertions.assertTrue(text.contains(tok),
                  "Reconstructed token '" + tok + "' must be in original text '" + text + "'");
            }
          }
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
        }
      }));
    }

    startLatch.countDown();

    for (Future<?> future : futures) {
      future.get(30, TimeUnit.SECONDS);
    }

    executor.shutdown();
    Assertions.assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
  }
}
