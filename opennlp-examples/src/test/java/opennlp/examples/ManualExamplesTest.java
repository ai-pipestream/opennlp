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
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Keeps the samples printed in the manual and the samples exercised here identical.
 * <p>
 * A {@code <programlisting>} in the manual opts in by carrying an {@code xml:id} of the form
 * {@code example.<chapter>.<section>}. The matching example is the method whose name is that id
 * without the {@code example.} prefix and with dots written as underscores, and the text compared
 * is the region between the {@code docs:begin} and {@code docs:end} markers in that method. A
 * listing without an id is not checked, so chapters can be converted one at a time, but an id
 * without a method and a method without an id are both failures.
 */
class ManualExamplesTest {

  private static final Pattern LISTING = Pattern.compile(
      "<programlisting[^>]*xml:id=\"(example\\.[^\"]+)\"[^>]*>\\s*<!\\[CDATA\\[(.*?)]]>",
      Pattern.DOTALL);

  private static final Pattern JAVA_LISTING = Pattern.compile("<programlisting[^>]*language=\"java\"");

  private static final Pattern METHOD = Pattern.compile(
      "^\\s+(?:public\\s+|protected\\s+|private\\s+)?(?:static\\s+)?void\\s+(\\w+)\\s*\\(");

  private static final String ID_PREFIX = "example.";

  private static final String CLASS_SUFFIX = "ChapterTest";

  private final Path docbkxDir = Paths.get(property("opennlp.docbkx.dir",
      "../opennlp-docs/src/docbkx"));

  private final Path examplesDir = Paths.get(property("opennlp.examples.dir", "src/test/java"));

  @Test
  void everyManualListingHasAMatchingExample() throws IOException {
    final Map<String, Listing> listings = readListings();
    final Map<String, Region> regions = readRegions();
    final List<String> problems = new ArrayList<>();

    for (Region region : regions.values()) {
      final Listing listing = listings.get(region.id());
      if (listing == null) {
        problems.add(region.where() + " marks an example region, but the manual has no listing "
            + "with xml:id=\"" + region.id() + "\"");
        continue;
      }
      if (!listing.code().equals(region.code())) {
        problems.add(listing.where() + " does not match " + region.where() + System.lineSeparator()
            + indent("manual", listing.code()) + System.lineSeparator()
            + indent("example", region.code()));
      }
    }
    for (Listing listing : listings.values()) {
      if (!regions.containsKey(listing.id())) {
        problems.add(listing.where() + " carries xml:id=\"" + listing.id() + "\", but no example "
            + "method " + methodNameFor(listing.id()) + " exists");
      }
    }
    Assertions.assertTrue(problems.isEmpty(),
        () -> problems.size() + " manual example problem(s):" + System.lineSeparator()
            + String.join(System.lineSeparator() + System.lineSeparator(), problems));
  }

  @Test
  void everyExampleSitsInTheTestClassForItsChapter() throws IOException {
    final List<String> problems = new ArrayList<>();
    for (Region region : readRegions().values()) {
      final String expected = region.chapter() + ".";
      final String tail = region.id().substring(ID_PREFIX.length());
      if (!tail.startsWith(expected) && !tail.equals(region.chapter())) {
        problems.add(region.where() + " is in the " + region.chapter() + " chapter class but "
            + "claims listing " + region.id());
      }
    }
    Assertions.assertTrue(problems.isEmpty(),
        () -> String.join(System.lineSeparator(), problems));
  }

  @Test
  void writeCoverageReport() throws IOException {
    final Map<String, Listing> listings = readListings();
    final StringBuilder report = new StringBuilder(512);
    report.append("# Manual example coverage").append(System.lineSeparator())
        .append(System.lineSeparator())
        .append("| chapter | java listings | verified |").append(System.lineSeparator())
        .append("| --- | --- | --- |").append(System.lineSeparator());

    int totalJava = 0;
    int totalVerified = 0;
    for (Path chapter : chapters()) {
      final String text = read(chapter);
      final String name = chapter.getFileName().toString().replace(".xml", "");
      int java = 0;
      final Matcher matcher = JAVA_LISTING.matcher(text);
      while (matcher.find()) {
        java++;
      }
      final String prefix = ID_PREFIX + name.replace("-", "") + ".";
      final long verified = listings.keySet().stream().filter(id -> id.startsWith(prefix)).count();
      totalJava += java;
      totalVerified += verified;
      report.append("| ").append(name).append(" | ").append(java).append(" | ")
          .append(verified).append(" |").append(System.lineSeparator());
    }
    report.append("| **total** | **").append(totalJava).append("** | **")
        .append(totalVerified).append("** |").append(System.lineSeparator());

    final Path out = Paths.get("manual-example-coverage.md");
    Files.write(out, report.toString().getBytes(StandardCharsets.UTF_8));
  }

  private Map<String, Listing> readListings() throws IOException {
    final Map<String, Listing> listings = new LinkedHashMap<>();
    for (Path chapter : chapters()) {
      final String text = read(chapter);
      final Matcher matcher = LISTING.matcher(text);
      while (matcher.find()) {
        final Listing listing = new Listing(matcher.group(1), chapter,
            lineOf(text, matcher.start()), normalize(matcher.group(2)));
        final Listing clash = listings.put(listing.id(), listing);
        Assertions.assertNull(clash, () -> "Two listings share xml:id " + listing.id() + ": "
            + listing.where() + " and " + clash.where());
      }
    }
    return listings;
  }

  private Map<String, Region> readRegions() throws IOException {
    final Map<String, Region> regions = new LinkedHashMap<>();
    for (Path source : sources()) {
      final List<String> lines = Files.readAllLines(source, StandardCharsets.UTF_8);
      String method = null;
      int start = -1;
      final List<String> collected = new ArrayList<>();
      for (int i = 0; i < lines.size(); i++) {
        final String line = lines.get(i);
        final Matcher matcher = METHOD.matcher(line);
        if (matcher.find()) {
          method = matcher.group(1);
        } else if (line.trim().equals(DocExampleSupport.BEGIN_MARKER)) {
          start = i + 1;
          collected.clear();
        } else if (line.trim().equals(DocExampleSupport.END_MARKER)) {
          Assertions.assertNotNull(method, () -> "A docs region outside a method in " + source);
          Assertions.assertTrue(start >= 0, () -> "docs:end without docs:begin in " + source);
          final Region region = new Region(method, source, start + 1, dedent(collected));
          final Region clash = regions.put(region.id(), region);
          Assertions.assertNull(clash, () -> "Two examples claim " + region.id());
          start = -1;
        } else if (start >= 0) {
          collected.add(line);
        }
      }
    }
    return regions;
  }

  private List<Path> chapters() throws IOException {
    try (Stream<Path> files = Files.list(docbkxDir)) {
      return files.filter(p -> p.getFileName().toString().endsWith(".xml"))
          .filter(p -> !p.getFileName().toString().equals("opennlp.xml"))
          .sorted(Comparator.comparing(Path::toString))
          .toList();
    }
  }

  private List<Path> sources() throws IOException {
    try (Stream<Path> files = Files.walk(examplesDir)) {
      return files.filter(p -> p.getFileName().toString().endsWith(".java"))
          .sorted(Comparator.comparing(Path::toString))
          .toList();
    }
  }

  private static String read(Path file) {
    try {
      return Files.readString(file, StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private static String property(String key, String fallback) {
    final String value = System.getProperty(key);
    return value == null || value.isBlank() ? fallback : value;
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

  private static String methodNameFor(String id) {
    return id.substring(ID_PREFIX.length()).replace('.', '_');
  }

  private static String normalize(String code) {
    final List<String> lines = new ArrayList<>(List.of(code.split("\n", -1)));
    for (int i = 0; i < lines.size(); i++) {
      lines.set(i, stripTrailing(lines.get(i)));
    }
    while (!lines.isEmpty() && lines.get(0).isEmpty()) {
      lines.remove(0);
    }
    while (!lines.isEmpty() && lines.get(lines.size() - 1).isEmpty()) {
      lines.remove(lines.size() - 1);
    }
    return String.join("\n", lines);
  }

  private static String dedent(List<String> lines) {
    final int margin = lines.stream().filter(l -> !l.isBlank())
        .mapToInt(l -> l.length() - l.stripLeading().length()).min().orElse(0);
    final List<String> out = new ArrayList<>(lines.size());
    for (String line : lines) {
      out.add(line.length() < margin ? "" : stripTrailing(line.substring(margin)));
    }
    return normalize(String.join("\n", out));
  }

  private static String stripTrailing(String line) {
    return line.stripTrailing();
  }

  private static String indent(String label, String code) {
    final StringBuilder out = new StringBuilder(code.length() + 32);
    out.append("  ").append(label).append(':');
    for (String line : code.split("\n", -1)) {
      out.append(System.lineSeparator()).append("    ").append(line);
    }
    return out.toString();
  }

  /** A code sample printed in the manual. */
  private record Listing(String id, Path file, int line, String code) {
    String where() {
      return file.getFileName() + ":" + line;
    }
  }

  /** The region of an example method that the manual prints. */
  private record Region(String method, Path file, int line, String code) {
    String id() {
      return ID_PREFIX + method.replace('_', '.');
    }

    String chapter() {
      final String name = file.getFileName().toString().replace(".java", "");
      return name.endsWith(CLASS_SUFFIX)
          ? name.substring(0, name.length() - CLASS_SUFFIX.length()).toLowerCase(Locale.ROOT)
          : name.toLowerCase(Locale.ROOT);
    }

    String where() {
      return file.getFileName().toString().replace(".java", "") + "#" + method
          + " (" + file.getFileName() + ":" + line + ")";
    }
  }
}
