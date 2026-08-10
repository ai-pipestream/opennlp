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

package opennlp.tools.assets;

import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import opennlp.tools.util.Span;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Pins the record contract of {@link EmbeddedAsset} under direct construction: every
 * compact-constructor rejection with its exact message, the {@code decode} bounds
 * check, and the accessor round-trip of a valid instance.
 */
public class EmbeddedAssetTest {

  /**
   * Builds a valid asset whose payload sits inside its span, for the accessor and
   * decode tests.
   *
   * @return An asset with span 4..12, payload 8..12, and matching metadata.
   */
  private static EmbeddedAsset valid() {
    return new EmbeddedAsset(new Span(4, 12), new Span(8, 12), "png", "image/png",
        3, 5, 7);
  }

  /**
   * The invalid constructions the compact constructor documents, each paired with the
   * exact rejection message it must produce.
   *
   * @return One case per violated constraint: payload starting before the span,
   *         payload ending behind the span, blank format, blank media type, and a
   *         negative decoded length.
   */
  private static Stream<Arguments> invalidConstructions() {
    return Stream.of(
        Arguments.of("payload must lie inside the span",
            (Executable) () -> new EmbeddedAsset(new Span(4, 12), new Span(3, 12),
                "png", "image/png", 3, -1, -1)),
        Arguments.of("payload must lie inside the span",
            (Executable) () -> new EmbeddedAsset(new Span(4, 12), new Span(4, 13),
                "png", "image/png", 3, -1, -1)),
        Arguments.of("format must not be null or blank",
            (Executable) () -> new EmbeddedAsset(new Span(4, 12), new Span(8, 12),
                " ", "image/png", 3, -1, -1)),
        Arguments.of("mediaType must not be null or blank",
            (Executable) () -> new EmbeddedAsset(new Span(4, 12), new Span(8, 12),
                "png", " ", 3, -1, -1)),
        Arguments.of("decodedLength must not be negative",
            (Executable) () -> new EmbeddedAsset(new Span(4, 12), new Span(8, 12),
                "png", "image/png", -1, -1, -1)));
  }

  @ParameterizedTest
  @MethodSource("invalidConstructions")
  void testConstructorRejectsWithTheDocumentedMessage(String message,
      Executable construction) {
    final IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, construction);
    assertEquals(message, e.getMessage());
  }

  @Test
  void testAccessorsRoundTrip() {
    final EmbeddedAsset asset = valid();
    assertEquals(new Span(4, 12), asset.span());
    assertEquals(new Span(8, 12), asset.payload());
    assertEquals("png", asset.format());
    assertEquals("image/png", asset.mediaType());
    assertEquals(3, asset.decodedLength());
    assertEquals(5, asset.width());
    assertEquals(7, asset.height());
  }

  @Test
  void testDecodeReadsThePayloadSpanFromTheText() {
    // Payload characters at 8..12 are the base64 image of "ABC".
    assertArrayEquals(new byte[] {'A', 'B', 'C'}, valid().decode("padding QUJD"));
  }

  @Test
  void testDecodeRejectsTextShorterThanThePayloadSpan() {
    final IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
        () -> valid().decode("short"));
    assertEquals("text is shorter than the payload span", e.getMessage());
  }
}
