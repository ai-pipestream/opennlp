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

package opennlp.examples;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.util.StringUtil;

/**
 * Keeps the samples printed in the manual and the samples exercised here identical.
 * <p>
 * A {@code <programlisting>} in the manual opts in by carrying an {@code xml:id} of the form
 * {@code example.<chapter>.<section>}. The matching region in the example sources starts at a
 * {@code docs:begin} marker followed by that id and ends at the next {@code docs:end} marker. A
 * region may sit in a method or at class level. A listing without an id is not checked, so
 * chapters can be converted one at a time, but an id without a region and a region without a
 * listing are both failures.
 */
class ManualExamplesTest {

  private static final String LISTING_OPEN = "<programlisting";

  private static final String CDATA_OPEN = "<![CDATA[";

  private static final String CDATA_CLOSE = "]]>";

  private static final String ID_ATTRIBUTE = "xml:id=\"";

  private static final String ID_PREFIX = "example.";

  private final Path docbkxDir = Paths.get(property("opennlp.docbkx.dir",
      "../opennlp-docs/src/docbkx"));

  private final Path examplesDir = Paths.get(property("opennlp.examples.dir", "src/test/java"));

  @Test
  void everyManualListingMatchesItsExampleRegion() throws IOException {
    final Map<String, Snippet> listings = readListings();
    final Map<String, Snippet> regions = readRegions();
    final List<String> problems = new ArrayList<>();

    for (Snippet region : regions.values()) {
      final Snippet listing = listings.get(region.id());
      if (listing == null) {
        problems.add(region.where() + " marks region " + region.id()
            + ", but the manual has no listing with that xml:id");
      } else if (!listing.code().equals(region.code())) {
        problems.add(listing.where() + " does not match " + region.where() + System.lineSeparator()
            + indent("manual", listing.code()) + System.lineSeparator()
            + indent("example", region.code()));
      }
    }
    for (Snippet listing : listings.values()) {
      if (!regions.containsKey(listing.id())) {
        problems.add(listing.where() + " carries xml:id=\"" + listing.id() + "\", but no "
            + "example marks a region " + DocExampleSupport.BEGIN_MARKER + " " + listing.id());
      }
    }
    Assertions.assertTrue(problems.isEmpty(),
        () -> problems.size() + " manual example problem(s):" + System.lineSeparator()
            + String.join(System.lineSeparator() + System.lineSeparator(), problems));
  }

  private Map<String, Snippet> readListings() throws IOException {
    final Map<String, Snippet> listings = new LinkedHashMap<>();
    for (Path chapter : files(docbkxDir, ".xml")) {
      final String text = Files.readString(chapter, StandardCharsets.UTF_8);
      for (int open = text.indexOf(LISTING_OPEN); open >= 0;
           open = text.indexOf(LISTING_OPEN, open + LISTING_OPEN.length())) {
        final String tag = openingTag(text, open);
        final String id = attribute(tag);
        if (id == null || !id.startsWith(ID_PREFIX)) {
          continue;
        }
        final int line = lineOf(text, open);
        final int body = skipWhitespace(text, open + tag.length());
        Assertions.assertTrue(text.startsWith(CDATA_OPEN, body),
            () -> chapter.getFileName() + ":" + line + " listing " + id + " has no CDATA body");
        final int end = text.indexOf(CDATA_CLOSE, body);
        Assertions.assertTrue(end >= 0,
            () -> chapter.getFileName() + ":" + line + " listing " + id + " has an open CDATA");
        add(listings, new Snippet(id, chapter, line,
            normalize(text.substring(body + CDATA_OPEN.length(), end))));
      }
    }
    return listings;
  }

  private Map<String, Snippet> readRegions() throws IOException {
    final Map<String, Snippet> regions = new LinkedHashMap<>();
    for (Path source : files(examplesDir, ".java")) {
      final List<String> lines = Files.readAllLines(source, StandardCharsets.UTF_8);
      String id = null;
      int start = -1;
      for (int i = 0; i < lines.size(); i++) {
        final String trimmed = StringUtil.trimUnicodeWhitespace(lines.get(i));
        final String where = source.getFileName() + ":" + (i + 1);
        if (trimmed.startsWith(DocExampleSupport.BEGIN_MARKER)) {
          final String outer = id;
          Assertions.assertNull(outer, () -> where + " opens a region inside region " + outer);
          id = StringUtil.trimUnicodeWhitespace(
              trimmed.substring(DocExampleSupport.BEGIN_MARKER.length()));
          Assertions.assertTrue(id.startsWith(ID_PREFIX),
              () -> where + " names no listing id after " + DocExampleSupport.BEGIN_MARKER);
          start = i + 1;
        } else if (trimmed.equals(DocExampleSupport.END_MARKER)) {
          Assertions.assertNotNull(id,
              () -> where + " closes a region that was never opened");
          add(regions, new Snippet(id, source, start + 1, dedent(lines.subList(start, i))));
          id = null;
        }
      }
      final String open = id;
      Assertions.assertNull(open, () -> source.getFileName() + " never closes region " + open);
    }
    return regions;
  }

  private static void add(Map<String, Snippet> snippets, Snippet snippet) {
    final Snippet clash = snippets.put(snippet.id(), snippet);
    Assertions.assertNull(clash, () -> snippet.where() + " and " + clash.where()
        + " both use " + snippet.id());
  }

  private static List<Path> files(Path dir, String extension) throws IOException {
    try (Stream<Path> files = Files.walk(dir)) {
      return files.filter(p -> p.getFileName().toString().endsWith(extension))
          .sorted(Comparator.comparing(Path::toString))
          .toList();
    }
  }

  private static String property(String key, String fallback) {
    final String value = System.getProperty(key);
    return StringUtil.isUnicodeBlank(value) ? fallback : value;
  }

  private static String openingTag(String text, int open) {
    final int close = text.indexOf('>', open);
    return close < 0 ? text.substring(open) : text.substring(open, close + 1);
  }

  private static String attribute(String tag) {
    final int start = tag.indexOf(ID_ATTRIBUTE);
    if (start < 0) {
      return null;
    }
    final int from = start + ID_ATTRIBUTE.length();
    final int end = tag.indexOf('"', from);
    return end < 0 ? null : tag.substring(from, end);
  }

  private static int skipWhitespace(String text, int from) {
    int i = from;
    while (i < text.length()) {
      final int cp = text.codePointAt(i);
      if (!StringUtil.isUnicodeWhitespace(cp)) {
        break;
      }
      i += Character.charCount(cp);
    }
    return i;
  }

  private static int lineOf(String text, int offset) {
    int line = 1;
    for (int i = 0; i < offset; i++) {
      if (text.charAt(i) == '\n') {
        line++;
      }
    }
    return line;
  }

  private static String normalize(String code) {
    final List<String> lines = new ArrayList<>(List.of(StringUtil.split(code, '\n', -1)));
    lines.replaceAll(ManualExamplesTest::stripTrailing);
    while (!lines.isEmpty() && lines.get(0).isEmpty()) {
      lines.remove(0);
    }
    while (!lines.isEmpty() && lines.get(lines.size() - 1).isEmpty()) {
      lines.remove(lines.size() - 1);
    }
    return String.join("\n", lines);
  }

  private static String dedent(List<String> lines) {
    final int margin = lines.stream().filter(l -> !StringUtil.isUnicodeBlank(l))
        .mapToInt(l -> skipWhitespace(l, 0)).min().orElse(0);
    final List<String> out = new ArrayList<>(lines.size());
    for (String line : lines) {
      out.add(line.length() < margin ? "" : line.substring(margin));
    }
    return normalize(String.join("\n", out));
  }

  private static String stripTrailing(String line) {
    int end = line.length();
    while (end > 0) {
      final int cp = line.codePointBefore(end);
      if (!StringUtil.isUnicodeWhitespace(cp)) {
        break;
      }
      end -= Character.charCount(cp);
    }
    return line.substring(0, end);
  }

  private static String indent(String label, String code) {
    final StringBuilder out = new StringBuilder(code.length() + 32);
    out.append("  ").append(label).append(':');
    for (String line : StringUtil.split(code, '\n', -1)) {
      out.append(System.lineSeparator()).append("    ").append(line);
    }
    return out.toString();
  }

  /** A code sample, either printed in the manual or marked in an example source. */
  private record Snippet(String id, Path file, int line, String code) {
    String where() {
      return file.getFileName() + ":" + line;
    }
  }
}
