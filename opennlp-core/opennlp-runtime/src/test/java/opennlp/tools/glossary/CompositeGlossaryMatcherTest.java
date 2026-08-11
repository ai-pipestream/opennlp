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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.stemmer.snowball.SnowballStemmer;
import opennlp.tools.stemmer.snowball.SnowballStemmerFactory;
import opennlp.tools.util.Span;
import opennlp.tools.util.normalizer.TermAnalyzer;

/**
 * Pins composite priority merging: earlier matchers win on span overlap, touching
 * spans both survive, and accepted hits are reported in text order.
 */
public class CompositeGlossaryMatcherTest {

  /**
   * Builds the analyzer shared by the English inflection cases.
   *
   * @return A case-folding English stemming analyzer.
   */
  private TermAnalyzer englishStemmingAnalyzer() {
    return TermAnalyzer.builder()
        .caseFold()
        .stem(new SnowballStemmerFactory(SnowballStemmer.ALGORITHM.ENGLISH))
        .build();
  }

  /**
   * Exact matcher first: when both paths hit the same stretch, the exact hit
   * survives and the inflected hit is dropped.
   */
  @Test
  void testExactWinsOverInflectedOnSameStretch() {
    final AhoCorasickGlossaryMatcher exact = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("EXACT", "hot dog")), true);
    final TermAnalyzingGlossaryMatcher inflected = new TermAnalyzingGlossaryMatcher(
        List.of(new GlossaryEntry("INFLECTED", "hot dog")),
        englishStemmingAnalyzer());
    final CompositeGlossaryMatcher composite = new CompositeGlossaryMatcher(
        List.of(exact, inflected));

    final List<GlossaryMatch> matches = composite.match("hot dog");

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("EXACT", matches.get(0).id());
    Assertions.assertEquals("hot dog", matches.get(0).term());
    Assertions.assertEquals(new Span(0, 7), matches.get(0).span());
  }

  /**
   * The inflected path fills gaps the exact matcher misses, while exact hits
   * elsewhere still appear in the merged result.
   */
  @Test
  void testInflectedFillsGapsExactMisses() {
    final List<GlossaryEntry> glossary = List.of(
        new GlossaryEntry("FOOD", "hot dog"),
        new GlossaryEntry("CITY", "New York City"));
    final AhoCorasickGlossaryMatcher exact =
        new AhoCorasickGlossaryMatcher(glossary, true);
    final TermAnalyzingGlossaryMatcher inflected =
        new TermAnalyzingGlossaryMatcher(glossary, englishStemmingAnalyzer());
    final CompositeGlossaryMatcher composite = new CompositeGlossaryMatcher(
        List.of(exact, inflected));

    final String text = "hot dogs in New York City";
    final List<GlossaryMatch> matches = composite.match(text);

    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals("FOOD", matches.get(0).id());
    Assertions.assertEquals("hot dog", matches.get(0).term());
    Assertions.assertEquals(new Span(0, 8), matches.get(0).span());
    Assertions.assertEquals("CITY", matches.get(1).id());
    Assertions.assertEquals("New York City", matches.get(1).term());
    Assertions.assertEquals("New York City", text.substring(
        matches.get(1).span().getStart(), matches.get(1).span().getEnd()));
  }

  /**
   * Inflected matcher first: on the same stretch the inflected hit wins and the
   * exact hit is dropped.
   */
  @Test
  void testInflectedWinsWhenOrderedFirst() {
    final AhoCorasickGlossaryMatcher exact = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("EXACT", "hot dog")), true);
    final TermAnalyzingGlossaryMatcher inflected = new TermAnalyzingGlossaryMatcher(
        List.of(new GlossaryEntry("INFLECTED", "hot dog")),
        englishStemmingAnalyzer());
    final CompositeGlossaryMatcher composite = new CompositeGlossaryMatcher(
        List.of(inflected, exact));

    final List<GlossaryMatch> matches = composite.match("hot dog");

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("INFLECTED", matches.get(0).id());
    Assertions.assertEquals("hot dog", matches.get(0).term());
    Assertions.assertEquals(new Span(0, 7), matches.get(0).span());
  }

  /**
   * Touching spans (one end equals the next start) do not intersect, so both
   * hits survive.
   */
  @Test
  void testTouchingSpansBothSurvive() {
    final GlossaryMatcher first = text -> List.of(
        new GlossaryMatch(new Span(0, 3), "A", "foo"));
    final GlossaryMatcher second = text -> List.of(
        new GlossaryMatch(new Span(3, 6), "B", "bar"));
    final CompositeGlossaryMatcher composite = new CompositeGlossaryMatcher(
        List.of(first, second));

    final List<GlossaryMatch> matches = composite.match("foobar");

    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals("A", matches.get(0).id());
    Assertions.assertEquals(new Span(0, 3), matches.get(0).span());
    Assertions.assertEquals("B", matches.get(1).id());
    Assertions.assertEquals(new Span(3, 6), matches.get(1).span());
  }

  /**
   * A lower-priority hit that partially overlaps an accepted span, and one that
   * is contained by it, are both dropped.
   */
  @Test
  void testPartialOverlapAndContainmentDropped() {
    final GlossaryMatcher high = text -> List.of(
        new GlossaryMatch(new Span(5, 15), "HIGH", "kept"));
    final GlossaryMatcher low = text -> List.of(
        new GlossaryMatch(new Span(10, 20), "PARTIAL", "overlap"),
        new GlossaryMatch(new Span(7, 12), "INNER", "contained"));
    final CompositeGlossaryMatcher composite = new CompositeGlossaryMatcher(
        List.of(high, low));

    final List<GlossaryMatch> matches = composite.match("01234567890123456789");

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("HIGH", matches.get(0).id());
    Assertions.assertEquals(new Span(5, 15), matches.get(0).span());
  }

  /**
   * A hit accepted from a later matcher that occurs earlier in the text is
   * reported first: the merged list is sorted by start offset ascending.
   */
  @Test
  void testResultOrderedByAscendingStartOffset() {
    final GlossaryMatcher high = text -> List.of(
        new GlossaryMatch(new Span(10, 15), "LATE", "later"));
    final GlossaryMatcher low = text -> List.of(
        new GlossaryMatch(new Span(0, 5), "EARLY", "earlier"));
    final CompositeGlossaryMatcher composite = new CompositeGlossaryMatcher(
        List.of(high, low));

    final List<GlossaryMatch> matches = composite.match("0123456789012345");

    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals("EARLY", matches.get(0).id());
    Assertions.assertEquals(new Span(0, 5), matches.get(0).span());
    Assertions.assertEquals("LATE", matches.get(1).id());
    Assertions.assertEquals(new Span(10, 15), matches.get(1).span());
    Assertions.assertTrue(
        matches.get(0).span().getStart() < matches.get(1).span().getStart());
  }

  /**
   * A single-delegate composite returns the same hits as that delegate alone.
   */
  @Test
  void testSingleDelegatePassthrough() {
    final AhoCorasickGlossaryMatcher exact = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("CITY", "New York City")), true);
    final CompositeGlossaryMatcher composite = new CompositeGlossaryMatcher(
        List.of(exact));

    final String text = "Flights to New York City are full.";
    final List<GlossaryMatch> direct = exact.match(text);
    final List<GlossaryMatch> merged = composite.match(text);

    Assertions.assertEquals(direct.size(), merged.size());
    Assertions.assertEquals(direct.get(0).id(), merged.get(0).id());
    Assertions.assertEquals(direct.get(0).term(), merged.get(0).term());
    Assertions.assertEquals(direct.get(0).span(), merged.get(0).span());
  }

  /**
   * A composite used as a delegate inside another composite still merges under
   * the outer priority order.
   */
  @Test
  void testNestedCompositeAsDelegate() {
    final GlossaryMatcher innerHigh = text -> List.of(
        new GlossaryMatch(new Span(0, 3), "INNER", "foo"));
    final GlossaryMatcher innerLow = text -> List.of(
        new GlossaryMatch(new Span(0, 3), "DROPPED", "foo"));
    final CompositeGlossaryMatcher inner = new CompositeGlossaryMatcher(
        List.of(innerHigh, innerLow));
    final GlossaryMatcher outerLow = text -> List.of(
        new GlossaryMatch(new Span(3, 6), "OUTER", "bar"));
    final CompositeGlossaryMatcher outer = new CompositeGlossaryMatcher(
        List.of(inner, outerLow));

    final List<GlossaryMatch> matches = outer.match("foobar");

    Assertions.assertEquals(2, matches.size());
    Assertions.assertEquals("INNER", matches.get(0).id());
    Assertions.assertEquals(new Span(0, 3), matches.get(0).span());
    Assertions.assertEquals("OUTER", matches.get(1).id());
    Assertions.assertEquals(new Span(3, 6), matches.get(1).span());
  }

  /**
   * Registering the same delegate twice adds nothing: the second pass produces
   * identical spans that intersect the already-accepted hits.
   */
  @Test
  void testSameDelegateTwiceAddsNothing() {
    final GlossaryMatcher delegate = text -> List.of(
        new GlossaryMatch(new Span(0, 3), "A", "foo"));
    final CompositeGlossaryMatcher composite = new CompositeGlossaryMatcher(
        List.of(delegate, delegate));

    final List<GlossaryMatch> matches = composite.match("foobar");

    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("A", matches.get(0).id());
    Assertions.assertEquals(new Span(0, 3), matches.get(0).span());
  }

  /**
   * Construction rejects a null list, an empty list, and a list containing null.
   */
  @Test
  void testInvalidConstructorArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new CompositeGlossaryMatcher(null));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new CompositeGlossaryMatcher(List.of()));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new CompositeGlossaryMatcher(Collections.singletonList(null)));
  }

  /**
   * Null text fails loud with {@link IllegalArgumentException}.
   */
  @Test
  void testNullTextThrows() {
    final CompositeGlossaryMatcher composite = new CompositeGlossaryMatcher(
        List.of(text -> List.of()));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> composite.match(null));
  }

  /**
   * No registered hits and empty text both yield an empty, non-null list.
   */
  @Test
  void testNoHitsAndEmptyTextYieldEmptyList() {
    final CompositeGlossaryMatcher composite = new CompositeGlossaryMatcher(
        List.of(text -> List.of()));
    final List<GlossaryMatch> miss = composite.match("nothing here");
    final List<GlossaryMatch> empty = composite.match("");
    Assertions.assertNotNull(miss);
    Assertions.assertNotNull(empty);
    Assertions.assertTrue(miss.isEmpty());
    Assertions.assertTrue(empty.isEmpty());
  }

  /**
   * The constructor copies the delegate list, so clearing the caller's list
   * after construction does not break matching.
   */
  @Test
  void testConstructorDefensivelyCopiesDelegateList() {
    final AhoCorasickGlossaryMatcher exact = new AhoCorasickGlossaryMatcher(
        List.of(new GlossaryEntry("CITY", "New York City")), true);
    final List<GlossaryMatcher> delegates = new ArrayList<>();
    delegates.add(exact);
    final CompositeGlossaryMatcher composite =
        new CompositeGlossaryMatcher(delegates);
    delegates.clear();

    final List<GlossaryMatch> matches =
        composite.match("Flights to New York City are full.");
    Assertions.assertEquals(1, matches.size());
    Assertions.assertEquals("CITY", matches.get(0).id());
  }
}
