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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.ml.model.Event;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Span;

/**
 * Tests for the {@link TokSpanEventStream} class.
 */
public class TokSpanEventStreamTest {

  @Test
  void testOptimizationRejectsContradictoryTrainingAnnotation() throws IOException {
    TokenSample sample = new TokenSample("abcd", new Span[] {new Span(0, 2), new Span(2, 4)});
    try (ObjectStream<Event> events = new TokSpanEventStream(
        ObjectStreamUtils.createObjectStream(sample), true)) {
      IllegalArgumentException error = Assertions.assertThrows(IllegalArgumentException.class, events::read);
      Assertions.assertTrue(error.getMessage().contains("policy"));
    }
  }

  @Test
  void testTrainingDoesNotCreateBoundaryInsideSurrogatePair() throws IOException {
    TokenSample sample = new TokenSample("a\uD83D\uDE00b", new Span[] {new Span(0, 4)});
    TokenContextGenerator context = (token, index) -> {
      Assertions.assertNotEquals(2, index, "A surrogate pair has no token boundary inside it");
      return new String[] {"offset=" + index};
    };
    try (ObjectStream<Event> events = new TokSpanEventStream(
        ObjectStreamUtils.createObjectStream(sample), false, context)) {
      Assertions.assertEquals("offset=1", events.read().getContext()[0]);
      Assertions.assertEquals("offset=3", events.read().getContext()[0]);
      Assertions.assertNull(events.read());
    }
  }

  @Test
  void testTrainingRejectsAnnotationBoundaryInsideSurrogatePair() throws IOException {
    TokenSample sample = new TokenSample("a\uD83D\uDE00b",
        new Span[] {new Span(0, 2), new Span(2, 4)});
    try (ObjectStream<Event> events = new TokSpanEventStream(
        ObjectStreamUtils.createObjectStream(sample), false)) {
      IllegalArgumentException error = Assertions.assertThrows(
          IllegalArgumentException.class, events::read);
      Assertions.assertTrue(error.getMessage().contains("surrogate pair"));
    }
  }

  @Test
  void testLongOptimizedSampleProducesNoEvents() throws IOException {
    StringBuilder text = new StringBuilder();
    List<Span> spans = new ArrayList<>();
    for (int i = 0; i < 10_000; i++) {
      if (!text.isEmpty()) {
        text.append(' ');
      }
      int start = text.length();
      text.append("word");
      spans.add(new Span(start, text.length()));
    }
    TokenSample sample = new TokenSample(text.toString(), spans.toArray(Span[]::new));

    try (ObjectStream<Event> events = new TokSpanEventStream(
        ObjectStreamUtils.createObjectStream(sample), true)) {
      Assertions.assertNull(events.read());
    }
  }

  @Test
  void testRejectsEmptyTrainingSpan() throws IOException {
    TokenSample sample = new TokenSample("ab", new Span[] {new Span(0, 2), new Span(2, 2)});
    try (ObjectStream<Event> events = new TokSpanEventStream(
        ObjectStreamUtils.createObjectStream(sample), false)) {
      IllegalArgumentException error = Assertions.assertThrows(
          IllegalArgumentException.class, events::read);
      Assertions.assertTrue(error.getMessage().contains("empty"));
    }
  }

  @Test
  void testRejectsOverlappingTrainingSpans() throws IOException {
    TokenSample sample = new TokenSample("abc", new Span[] {new Span(0, 2), new Span(1, 3)});
    try (ObjectStream<Event> events = new TokSpanEventStream(
        ObjectStreamUtils.createObjectStream(sample), false)) {
      IllegalArgumentException error = Assertions.assertThrows(
          IllegalArgumentException.class, events::read);
      Assertions.assertTrue(error.getMessage().contains("ordered"));
    }
  }

  @Test
  void testRejectsNullTrainingStream() {
    IllegalArgumentException error = Assertions.assertThrows(IllegalArgumentException.class,
        () -> new TokSpanEventStream(null, false));
    Assertions.assertTrue(error.getMessage().contains("tokenSamples"));
  }

  /**
   * Tests the event stream for correctly generated outcomes.
   */
  @Test
  void testEventOutcomes() throws IOException {

    ObjectStream<String> sentenceStream =
        ObjectStreamUtils.createObjectStream("\"<SPLIT>out<SPLIT>.<SPLIT>\"");

    ObjectStream<TokenSample> tokenSampleStream = new TokenSampleStream(sentenceStream);

    try (ObjectStream<Event> eventStream = new TokSpanEventStream(tokenSampleStream, false)) {

      Assertions.assertEquals(TokenizerME.SPLIT, eventStream.read().getOutcome());
      Assertions.assertEquals(TokenizerME.NO_SPLIT, eventStream.read().getOutcome());
      Assertions.assertEquals(TokenizerME.NO_SPLIT, eventStream.read().getOutcome());
      Assertions.assertEquals(TokenizerME.SPLIT, eventStream.read().getOutcome());
      Assertions.assertEquals(TokenizerME.SPLIT, eventStream.read().getOutcome());

      Assertions.assertNull(eventStream.read());
      Assertions.assertNull(eventStream.read());
    }
  }
}
