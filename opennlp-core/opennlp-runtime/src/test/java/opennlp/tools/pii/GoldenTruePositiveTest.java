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

package opennlp.tools.pii;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/** Verifies exact spans, types, and normalized forms from the positive golden corpus. */
public class GoldenTruePositiveTest {

  private static final String RESOURCE =
      "/opennlp/tools/pii/golden-true-positives.tsv";

  private static final PiiExtractor EXTRACTOR = PiiPacks.allStructured();

  /**
   * Verifies one exact positive-corpus row.
   *
   * @param type The expected mention type.
   * @param covered The expected covered source text.
   * @param normalized The expected normalized form.
   * @param text The source text.
   */
  @ParameterizedTest
  @MethodSource("examples")
  void testGoldenExample(String type, String covered, String normalized, String text) {
    final List<PiiMention> mentions = EXTRACTOR.extract(text);

    Assertions.assertEquals(1, mentions.size(), text);
    final PiiMention mention = mentions.get(0);
    Assertions.assertEquals(type, mention.type(), text);
    Assertions.assertEquals(covered,
        text.substring(mention.span().getStart(), mention.span().getEnd()), text);
    Assertions.assertEquals(normalized, mention.normalized(), text);
  }

  /**
   * Loads the tab-separated corpus.
   *
   * @return The examples.
   * @throws IOException Thrown if the resource cannot be read.
   */
  private static Stream<Arguments> examples() throws IOException {
    final InputStream stream = GoldenTruePositiveTest.class.getResourceAsStream(RESOURCE);
    if (stream == null) {
      throw new IOException("missing resource: " + RESOURCE);
    }
    try (BufferedReader reader = new BufferedReader(
        new InputStreamReader(stream, StandardCharsets.UTF_8))) {
      return reader.lines()
          .filter(line -> !line.isBlank() && line.charAt(0) != '#')
          .map(GoldenTruePositiveTest::parse)
          .toList()
          .stream();
    }
  }

  /**
   * Parses one four-field corpus row without a regular expression.
   *
   * @param line The row.
   * @return The test arguments.
   */
  private static Arguments parse(String line) {
    final String[] fields = new String[4];
    int start = 0;
    for (int field = 0; field < fields.length - 1; field++) {
      final int end = line.indexOf('\t', start);
      if (end < 0) {
        throw new IllegalArgumentException("golden row must contain four fields: " + line);
      }
      fields[field] = line.substring(start, end);
      start = end + 1;
    }
    fields[fields.length - 1] = line.substring(start);
    return Arguments.of((Object[]) fields);
  }
}
