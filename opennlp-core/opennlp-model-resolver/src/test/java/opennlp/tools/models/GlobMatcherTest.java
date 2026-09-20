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
package opennlp.tools.models;

import java.net.URI;
import java.net.URL;
import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class GlobMatcherTest {

  private record TestArguments(Object... values) {
    static TestArguments of(Object... values) {
      return new TestArguments(values);
    }

    String string(int index) {
      return (String) values[index];
    }

    boolean bool(int index) {
      return (boolean) values[index];
    }
  }

  private static final String MODEL_URL =
      "jar:file:/repo/opennlp-models-pos-en-1.2.0.jar!/opennlp/models/en-pos.bin";

  private static final String SMILEY = "\uD83D\uDE00";

  private static Stream<TestArguments> accepted() {
    return Stream.of(
        TestArguments.of("*", ""),
        TestArguments.of("*", "anything at all"),
        TestArguments.of("**", "ab"),
        TestArguments.of("", ""),
        TestArguments.of("a", "a"),
        TestArguments.of("a*", "a"),
        TestArguments.of("a*", "abc"),
        TestArguments.of("*a", "a"),
        TestArguments.of("*a", "bca"),
        TestArguments.of("*a*b*", "xaybz"),
        TestArguments.of("a?c", "abc"),
        TestArguments.of("a?c", "a.c"),
        TestArguments.of("?", "a"),
        TestArguments.of("?", SMILEY),
        TestArguments.of("*.bin", "en-pos.bin"),
        TestArguments.of("*.bin", ".bin"),
        TestArguments.of("*model.properties", "/x/opennlp-models-pos-en-1.2.0.jar!/model.properties"),
        TestArguments.of("*opennlp-models-*", "/repo/opennlp-models-pos-en-1.2.0.jar"),
        TestArguments.of("*opennlp-models-*.jar", "/repo/opennlp-models-pos-en-1.2.0.jar"),
        TestArguments.of("*-en-*.jar", "/repo/opennlp-models-pos-en-1.2.0.jar"),
        TestArguments.of("(a)", "(a)"),
        TestArguments.of("[ab]", "[ab]"),
        TestArguments.of("a$", "a$"),
        TestArguments.of("a+", "a+"),
        TestArguments.of("a\\b", "a\\b"),
        TestArguments.of("*" + SMILEY + "*", "a" + SMILEY + "b"),
        TestArguments.of(SMILEY + "?", SMILEY + SMILEY),
        // an unpaired surrogate is one character
        TestArguments.of("?", "\uD83D"),
        TestArguments.of("*", "\uD83D"),
        TestArguments.of("a?c", "a\uDE00c"),
        TestArguments.of("a*b*c", "abc"),
        TestArguments.of("*?", "a"),
        TestArguments.of("?*", "a"),
        TestArguments.of("*\n*", "a\nb"),
        TestArguments.of("a\nb", "a\nb"),
        TestArguments.of("*\n*\n*", "a\nb\nc"),
        TestArguments.of("*\r\n*", "a\r\nb"),
        // a wildcard covers a line terminator like any other character
        TestArguments.of("*", "a\nb"),
        TestArguments.of("*", "\n"),
        TestArguments.of("*.bin", "a\n.bin"),
        TestArguments.of("*a*b", "a\nab"),
        TestArguments.of("?", "\n"),
        TestArguments.of("a?b", "a\nb"),
        TestArguments.of("*", "a\rb"),
        TestArguments.of("*", "a\u0085b"),
        TestArguments.of("*", "a\u2028b"),
        TestArguments.of("*", "a\u2029b"),
        TestArguments.of("**", ""),
        TestArguments.of("***", "abc"),
        TestArguments.of("a**b", "ab"),
        TestArguments.of("a**b", "axyzb"),
        TestArguments.of("*?*", "a"),
        TestArguments.of("abc*", "abc"),
        TestArguments.of("*.bin*", ".bin"),
        TestArguments.of("[", "["),
        TestArguments.of("[a-z]", "[a-z]"),
        TestArguments.of("\\", "\\"),
        TestArguments.of("\\*", "\\lib\\a.jar"),
        TestArguments.of("\\Q*\\E", "\\Qx\\E"),
        // path separators are plain characters
        TestArguments.of("*/models/*.bin", "/x/models/en.bin"),
        TestArguments.of("/*.bin", "/en.bin"),
        TestArguments.of("*/*", "a/"),
        TestArguments.of("*/*", "/"),
        TestArguments.of("*\\*", "C:\\lib\\a.jar"),
        // a wildcard covers the jar separator and the entry path
        TestArguments.of("*.jar!/*.bin", "/repo/a.jar!/opennlp/en.bin"),
        TestArguments.of("*.jar!/opennlp/*", "/repo/a.jar!/opennlp/"),
        TestArguments.of("*!/*", "/repo/a.jar!/"),
        // a drive letter in the file part of a file URL
        TestArguments.of("/C:/*.jar", "/C:/lib/a.jar"),
        TestArguments.of("*:/lib/*", "/C:/lib/a.jar"),
        // percent-encoded file parts match as written
        TestArguments.of("*%20*", "/my%20models/en.bin"),
        TestArguments.of("/my%20models/*", "/my%20models/en.bin"),
        TestArguments.of("?%20?", "a%20b"),
        TestArguments.of("*" + SMILEY + "?", SMILEY + SMILEY));
  }

  @Test
  void testMatchesAccepts() {
    Assertions.assertAll(accepted().map(arguments -> () -> {
      String glob = arguments.string(0);
      String input = arguments.string(1);
      Assertions.assertTrue(GlobMatcher.matches(glob, input),
          "glob '" + glob + "' should accept '" + input + "'");
    }));
  }

  private static Stream<TestArguments> rejected() {
    return Stream.of(
        TestArguments.of("a", ""),
        TestArguments.of("", "a"),
        TestArguments.of("a", "b"),
        TestArguments.of("a", "ab"),
        TestArguments.of("ab", "a"),
        TestArguments.of("a*", "ba"),
        TestArguments.of("*a", "ab"),
        TestArguments.of("a?c", "ac"),
        TestArguments.of("a?c", "abbc"),
        TestArguments.of("?", ""),
        TestArguments.of("?", "ab"),
        TestArguments.of("*.bin", "en-pos.bini"),
        TestArguments.of("*.bin", "en-posxbin"),
        TestArguments.of("*.bin", "en-pos.BIN"),
        TestArguments.of("*model.properties", "/x/model.properties.bak"),
        TestArguments.of("*opennlp-models-*", "/repo/opennlp-model-pos-en-1.2.0.jar"),
        TestArguments.of("(a)", "a"),
        TestArguments.of("[ab]", "a"),
        TestArguments.of("a+", "aa"),
        TestArguments.of("a\\b", "a"),
        TestArguments.of(SMILEY, "\uD83D"),
        TestArguments.of("??", SMILEY),
        TestArguments.of("*.BIN", "en-pos.bin"),
        TestArguments.of("*?", ""),
        TestArguments.of("?*", ""),
        TestArguments.of("a\nb", "a b"),
        TestArguments.of("a\nb", "a\rb"),
        TestArguments.of("abcd", "abc"),
        TestArguments.of("a?cd", "abc"),
        TestArguments.of("*abcd", "abc"),
        TestArguments.of("abc?", "abc"),
        TestArguments.of("a**b", "a"),
        TestArguments.of("a**b", "ba"),
        TestArguments.of("*?*", ""),
        TestArguments.of("[a-z]", "b"),
        TestArguments.of("[", ""),
        TestArguments.of("\\", "\\\\"),
        TestArguments.of("\\Q*\\E", "x"),
        TestArguments.of("a/b", "a\\b"),
        TestArguments.of("a\\b", "a/b"),
        TestArguments.of("/*.bin", "en.bin"),
        TestArguments.of("*/models/*.bin", "/x/model/en.bin"),
        TestArguments.of("*.jar!/en.bin", "/repo/a.jar!/models/en.bin"),
        TestArguments.of("*.jar!/*", "/repo/a.jar/en.bin"),
        TestArguments.of("/C:/*.jar", "/D:/lib/a.jar"),
        TestArguments.of("C:/*.jar", "/C:/lib/a.jar"),
        TestArguments.of("*my models*", "/my%20models/en.bin"),
        TestArguments.of("*%20*", "/my models/en.bin"),
        TestArguments.of("?%20?", "a b"),
        TestArguments.of("?", "\r\n"),
        TestArguments.of("*" + SMILEY + "?", SMILEY),
        TestArguments.of(SMILEY + "*", "\uD83D"));
  }

  @Test
  void testMatchesRejects() {
    Assertions.assertAll(rejected().map(arguments -> () -> {
      String glob = arguments.string(0);
      String input = arguments.string(1);
      Assertions.assertFalse(GlobMatcher.matches(glob, input),
          "glob '" + glob + "' should reject '" + input + "'");
    }));
  }

  /**
   * Creates a finder probe with no context and no matches.
   *
   * @return A minimal {@link AbstractClassPathModelFinder} for matcher tests.
   */
  AbstractClassPathModelFinder newProbeFinder() {
    return new AbstractClassPathModelFinder() {
      @Override
      protected Object getContext() {
        return null;
      }

      @Override
      protected List<URI> getMatchingURIs(String wildcardPattern, Object context) {
        return List.of();
      }
    };
  }

  private static Stream<TestArguments> urlsAndWildcards() {
    return Stream.of(
        TestArguments.of(MODEL_URL, "*.bin", true),
        TestArguments.of(MODEL_URL, "*.jar!/opennlp/models/*.bin", true),
        TestArguments.of(MODEL_URL, "*.jar!/*", true),
        TestArguments.of(MODEL_URL, "*.jar!/en-pos.bin", false),
        TestArguments.of(MODEL_URL, "*.jar", false),
        TestArguments.of(MODEL_URL, "*opennlp-models-???-en-*", true),
        TestArguments.of(MODEL_URL, "*opennlp-models-??-en-*", false),
        // the file part of a file URL starts with a slash, the drive letter follows
        TestArguments.of("file:/C:/lib/opennlp-models-pos-en-1.2.0.jar", "/C:/*.jar", true),
        TestArguments.of("file:/C:/lib/opennlp-models-pos-en-1.2.0.jar", "C:/*.jar", false),
        TestArguments.of("file:/C:/lib/opennlp-models-pos-en-1.2.0.jar", "*opennlp-models-*", true),
        // the file part of a jar URL is the inner URL, scheme included
        TestArguments.of("jar:file:/C:/lib/a.jar!/opennlp/en-pos.bin", "file:/C:/*.jar!/*.bin", true),
        TestArguments.of("jar:file:/C:/lib/a.jar!/opennlp/en-pos.bin", "/C:/*.jar!/*.bin", false),
        TestArguments.of(MODEL_URL, "file:/repo/*.jar!/opennlp/models/en-pos.bin", true),
        TestArguments.of(MODEL_URL, "/repo/*.jar!/opennlp/models/en-pos.bin", false),
        TestArguments.of("jar:file:/C:/lib/a.jar!/opennlp/en-pos.bin", "*/en-pos.bin", true),
        TestArguments.of("jar:file:/C:/lib/a.jar!/opennlp/en-pos.bin", "*\\en-pos.bin", false),
        // URI escapes are decoded once, while literal plus signs are preserved
        TestArguments.of("file:/my%20models/en-pos.bin", "*%20*", false),
        TestArguments.of("file:/my%20models/en-pos.bin", "*my models*", true),
        TestArguments.of("file:/my%20models/en-pos.bin", "/my models/*.bin", true),
        TestArguments.of("file:/models/model-%F0%9F%98%80.jar", "*model-?.jar", true),
        TestArguments.of("file:/models/model-%F0%9F%98%80.jar", "*model-??.jar", false),
        TestArguments.of("file:/models/model-%2520.jar", "*model-%20.jar", true),
        TestArguments.of("file:/models/model-%2520.jar", "*model- .jar", false),
        TestArguments.of("file:/models/model+1.jar", "*model+1.jar", true),
        TestArguments.of("file:/models/model+1.jar", "*model 1.jar", false),
        TestArguments.of("jar:file:/my%20models/a.jar!/caf%C3%A9/%F0%9F%98%80.bin",
            "file:/my models/a.jar!/café/?.bin", true),
        TestArguments.of("jar:file:/models/a.jar!/model%2520.bin", "*model%20.bin", true),
        TestArguments.of("jar:file:/models/a.jar!/model%2520.bin", "*model .bin", false),
        TestArguments.of("jar:file:/models/a.jar!/model%23x.bin", "*model#x.bin", true),
        TestArguments.of("jar:file:/models/a.jar!/model%3Fx.bin", "*model?x.bin", true),
        // a query is part of the file part, a fragment is not
        TestArguments.of("http://host/models/en-pos.bin?x=1", "*.bin", false),
        TestArguments.of("http://host/models/en-pos.bin?x=1", "*.bin?x=1", true),
        TestArguments.of("http://host/models/en-pos.bin#top", "*.bin", true),
        TestArguments.of("file:/models/en-pos.bin", "*", true),
        TestArguments.of("file:/models/en-pos.bin", "", false),
        TestArguments.of("file:/models/en-pos.bin", "?", false));
  }

  /**
   * Checks decoded file part matching against jar, file, and http URLs.
   */
  @Test
  void testMatchesWildcardOnUrlFilePart() {
    Assertions.assertAll(urlsAndWildcards().map(arguments -> () -> {
      String url = arguments.string(0);
      String wildcard = arguments.string(1);
      boolean expected = arguments.bool(2);
      final AbstractClassPathModelFinder finder = newProbeFinder();
      final URL parsed = new URI(url).toURL();
      Assertions.assertEquals(expected, finder.matchesWildcard(parsed, wildcard),
          "wildcard '" + wildcard + "' on '" + parsed.getFile() + "'");
    }));
  }

  private static Stream<TestArguments> literalGlobs() {
    return Stream.of(
        TestArguments.of("*.bin", "en-pos.bin", "en-posxbin"),
        TestArguments.of("*.bin", ".bin", "en-pos.bin.bak"),
        TestArguments.of("a?c", "abc", "ac"),
        TestArguments.of("a?c", "a\nc", "abbc"),
        TestArguments.of("*a*b*", "xa\nyb\nz", "ba"),
        TestArguments.of("(a)", "(a)", "a"),
        TestArguments.of("[ab]", "[ab]", "a"),
        TestArguments.of("a+", "a+", "aa"),
        TestArguments.of("a$", "a$", "a"),
        TestArguments.of("a\\b", "a\\b", "ab"),
        TestArguments.of("\\Q*\\E", "\\Qx\\E", "x"),
        TestArguments.of("[", "[", ""),
        TestArguments.of("[a-z]", "[a-z]", "b"),
        TestArguments.of("\\", "\\", "\\\\"),
        TestArguments.of("a**", "a", "ba"),
        TestArguments.of("*a", "\na", "\n"),
        TestArguments.of("a*", "a\r\n", "\na"),
        TestArguments.of("?", "\n", "\r\n"),
        TestArguments.of("?", "\uD83D", SMILEY + SMILEY),
        TestArguments.of("*.jar!/*.bin", "/repo/a.jar!/en.bin", "/repo/a.jar/en.bin"),
        TestArguments.of("/C:/*", "/C:/lib/a.jar", "C:/lib/a.jar"),
        TestArguments.of("*%20*", "/my%20models", "/my models"),
        TestArguments.of(SMILEY + "?", SMILEY + SMILEY, SMILEY),
        TestArguments.of("", "", "a"));
  }

  /** Checks literal characters and both wildcards against explicit accept/reject examples. */
  @Test
  void testLiteralGlobSemantics() {
    Assertions.assertAll(literalGlobs().map(arguments -> () -> {
      String glob = arguments.string(0);
      String accepted = arguments.string(1);
      String rejected = arguments.string(2);
      Assertions.assertTrue(GlobMatcher.matches(glob, accepted));
      Assertions.assertFalse(GlobMatcher.matches(glob, rejected));
    }));
  }

  private static Stream<TestArguments> nullInputs() {
    return Stream.of(
        TestArguments.of(null, "a"),
        TestArguments.of("a", null),
        TestArguments.of(null, null));
  }

  /**
   * Checks that null glob or input fails fast instead of matching.
   */
  @Test
  void testMatchesRejectsNull() {
    Assertions.assertAll(nullInputs().map(arguments -> () ->
        Assertions.assertThrows(IllegalArgumentException.class,
            () -> GlobMatcher.matches(arguments.string(0), arguments.string(1)))));
  }

  /**
   * Checks that null arguments to the finder matchers fail fast.
   */
  @Test
  void testFinderMatchersRejectNull() throws Exception {
    final AbstractClassPathModelFinder finder = newProbeFinder();
    final URL url = new URI(MODEL_URL).toURL();
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> finder.matchesWildcard(null, "*.bin"));
    Assertions.assertThrows(IllegalArgumentException.class, () -> finder.matchesWildcard(url, null));
  }

  @Test
  void testMatchesWildcardUsesFilePart() throws Exception {
    final AbstractClassPathModelFinder finder = newProbeFinder();
    final URL url = new URI(MODEL_URL).toURL();
    Assertions.assertTrue(finder.matchesWildcard(url, "*.bin"));
    Assertions.assertTrue(finder.matchesWildcard(url, "*opennlp-models-*"));
    Assertions.assertTrue(finder.matchesWildcard(url, "*en-pos.bin"));
    Assertions.assertFalse(finder.matchesWildcard(url, "en-pos.bin"));
    Assertions.assertFalse(finder.matchesWildcard(url, "*.properties"));
    Assertions.assertFalse(finder.matchesWildcard(url, "jar:*"));
  }
}
