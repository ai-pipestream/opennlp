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

package opennlp.tools.glossary;

import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class AhoCorasickGlossaryMatcherTest {

  @Test
  void testFindsSingleAndMultiwordTerms() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(List.of(
        new GlossaryEntry("Q60", "New York City"),
        new GlossaryEntry("Q11299", "Manhattan")), false);

    final String text = "She moved from Manhattan to New York City last year.";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals("Q11299", matches.get(0).id());
    Assertions.assertEquals("Manhattan", text.substring(
        matches.get(0).span().getStart(), matches.get(0).span().getEnd()));
    Assertions.assertEquals("Q60", matches.get(1).id());
    Assertions.assertEquals("New York City", text.substring(
        matches.get(1).span().getStart(), matches.get(1).span().getEnd()));
  }

  @Test
  void testPrefersLongestMatch() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(List.of(
        new GlossaryEntry("STATE", "New York"),
        new GlossaryEntry("CITY", "New York City")), false);

    final List<GlossaryMatch> matches = matcher.match("Flights to New York City are full.");

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("CITY", matches.get(0).id());
  }

  @Test
  void testLeftmostWinsOverLaterOverlap() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(List.of(
        new GlossaryEntry("A", "New York"),
        new GlossaryEntry("B", "York City")), false);

    final List<GlossaryMatch> matches = matcher.match("in New York City today");

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("A", matches.get(0).id());
  }

  @Test
  void testIgnoreCaseKeepsOriginalSpan() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ML", "machine learning")), true);

    final String text = "Machine Learning is popular.";
    final List<GlossaryMatch> matches = matcher.match(text);

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals(0, matches.get(0).span().getStart());
    Assertions.assertEquals("Machine Learning".length(), matches.get(0).span().getEnd());
    Assertions.assertEquals("machine learning", matches.get(0).term());
  }

  @Test
  void testCaseSensitiveByDefaultDoesNotCrossCase() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("ML", "machine learning")), false);

    Assertions.assertTrue(matcher.match("Machine Learning is popular.").isEmpty());
  }

  @Test
  void testRespectsWordBoundaries() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("CAT", "cat")), false);

    Assertions.assertTrue(matcher.match("concatenate the files").isEmpty());
    Assertions.assertEquals(1, matcher.match("the cat sleeps").size());
    Assertions.assertEquals(1, matcher.match("cat").size());
    Assertions.assertEquals(1, matcher.match("a cat.").size());
  }

  @Test
  void testAliasesShareOneId() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(List.of(
        new GlossaryEntry("ACME-1", "widget press"),
        new GlossaryEntry("ACME-1", "press for widgets")), false);

    final List<GlossaryMatch> matches =
        matcher.match("The widget press replaced the old press for widgets.");

    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals("ACME-1", matches.get(0).id());
    Assertions.assertEquals("ACME-1", matches.get(1).id());
    Assertions.assertEquals("widget press", matches.get(0).term());
    Assertions.assertEquals("press for widgets", matches.get(1).term());
  }

  @Test
  void testDuplicateTermFirstRegistrationWins() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(List.of(
        new GlossaryEntry("FIRST", "python"),
        new GlossaryEntry("SECOND", "python")), false);

    final List<GlossaryMatch> matches = matcher.match("a python slithered by");

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("FIRST", matches.get(0).id());
  }

  @Test
  void testRepeatedHitsAllReported() {
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("T", "gene therapy")), true);

    Assertions.assertEquals(2,
        matcher.match("Gene therapy trials expand; gene therapy works.").size());
  }

  @Test
  void testInvalidArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new AhoCorasickGlossaryMatcher(null, false));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new AhoCorasickGlossaryMatcher(List.of(), false));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new GlossaryEntry("id", " "));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new GlossaryEntry(null, "term"));
    final AhoCorasickGlossaryMatcher matcher = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("T", "term")), false);
    Assertions.assertThrows(IllegalArgumentException.class, () -> matcher.match(null));
  }
}
