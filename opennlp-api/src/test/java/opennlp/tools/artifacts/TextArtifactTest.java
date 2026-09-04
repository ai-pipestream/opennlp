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

package opennlp.tools.artifacts;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.tools.util.Span;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Pins the construction contract of the {@link TextArtifact} record: the compact
 * constructor rejects a {@code null} span and a {@code null} or blank type with an
 * {@link IllegalArgumentException} carrying the exact stated message, and a valid
 * instance hands back its components unchanged.
 */
public class TextArtifactTest {

  /**
   * Verifies that a {@code null} span is rejected with an
   * {@link IllegalArgumentException}, not a {@link NullPointerException}, and the exact
   * message the record states.
   */
  @Test
  void testNullSpanIsRejected() {
    final IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
        () -> new TextArtifact(null, TextArtifact.TYPE_REPLACEMENT));
    assertEquals("span must not be null", e.getMessage());
  }

  /**
   * Verifies that a {@code null} type is rejected with an
   * {@link IllegalArgumentException}, not a {@link NullPointerException}, and the exact
   * message the record states.
   */
  @Test
  void testNullTypeIsRejected() {
    final IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
        () -> new TextArtifact(new Span(0, 1), null));
    assertEquals("type must not be null or blank", e.getMessage());
  }

  /**
   * Verifies that blank types are rejected under the toolkit whitespace definition,
   * which includes the no-break space, with the same message as a {@code null} type.
   */
  @ParameterizedTest
  @ValueSource(strings = {"", " ", "\t", "\n", "\u00A0"})
  void testBlankTypeIsRejected(String type) {
    final IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
        () -> new TextArtifact(new Span(2, 5), type));
    assertEquals("type must not be null or blank", e.getMessage());
  }

  /**
   * Verifies that a valid instance returns exactly the span and type it was built
   * from, and that an open type beyond the declared constants is accepted as the
   * javadoc promises.
   */
  @Test
  void testAccessorsReturnTheComponents() {
    final Span span = new Span(3, 5);
    final TextArtifact declared = new TextArtifact(span, TextArtifact.TYPE_MOJIBAKE);
    assertSame(span, declared.span());
    assertEquals(TextArtifact.TYPE_MOJIBAKE, declared.type());

    final TextArtifact open = new TextArtifact(new Span(0, 1), "custom-detector-type");
    assertEquals("custom-detector-type", open.type());
  }
}
