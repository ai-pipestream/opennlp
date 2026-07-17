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

package opennlp.geo;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Exercises the boundary behavior of place-profile similarity: identical and opposite
 * profiles, orthogonal profiles, tables without any variance, malformed rows, extreme
 * metric magnitudes, duplicate identifiers, tie ordering, and line-ending tolerance.
 * All tables are project-authored miniatures embedded in this file.
 *
 * <p>Expected scores that are exact by construction, such as {@code 0.0} for
 * orthogonal or variance-free profiles and {@code -1.0} for exactly mirrored ones, are
 * asserted without a delta. Scores that pass through rounded arithmetic were produced
 * by running the same table through {@link PlaceProfiles} and are asserted with a
 * tight delta, {@code 1e-12} or finer, that tolerates only last-ulp variation in
 * summation order.</p>
 */
public class PlaceProfilesEdgeCaseTest {

  /**
   * Loads a profile table from an in-memory string through the stream entry point,
   * exactly as production code would read the same bytes from any stream.
   *
   * @param table The tab-separated table content. Must not be {@code null}.
   * @return The loaded profiles. Never {@code null}.
   * @throws IOException Thrown if the table is malformed.
   */
  private static PlaceProfiles load(String table) throws IOException {
    if (table == null) {
      throw new IllegalArgumentException("table must not be null");
    }
    return PlaceProfiles.load(
        new ByteArrayInputStream(table.getBytes(StandardCharsets.UTF_8)));
  }

  /**
   * Asserts that two places with identical raw rows score fractionally above
   * {@code 1.0}: their standardized vectors are bitwise equal, and dividing the dot
   * product by the rounded product of the two square roots lands one ulp above unity.
   * Self-similarity of such a place yields the same value, and the third, mirrored
   * place scores exactly {@code -1.0}.
   */
  @Test
  void testIdenticalProfilesScoreOneUlpAboveUnity() throws IOException {
    final PlaceProfiles profiles = load(String.join("\n",
        "id\ta\tb",
        "twin-1\t10\t5",
        "twin-2\t10\t5",
        "other\t2\t9",
        ""));
    Assertions.assertEquals(1.0000000000000002,
        profiles.similarity("twin-1", "twin-2"), 1e-15);
    Assertions.assertEquals(profiles.similarity("twin-1", "twin-2"),
        profiles.similarity("twin-1", "twin-1"));
    Assertions.assertEquals(-1.0, profiles.similarity("twin-1", "other"));
  }

  /**
   * Asserts exact scores on a symmetric compass table whose columns standardize
   * without rounding: profiles on perpendicular axes score exactly {@code 0.0},
   * mirrored profiles score exactly {@code -1.0}, a profile against itself scores
   * exactly {@code 1.0}, and a neighbor query with a generous count returns every
   * other place.
   */
  @Test
  void testOrthogonalAndOppositeProfilesScoreExactly() throws IOException {
    final PlaceProfiles profiles = load(String.join("\n",
        "id\tx\ty",
        "north\t1\t0",
        "south\t-1\t0",
        "east\t0\t1",
        "west\t0\t-1",
        ""));
    Assertions.assertEquals(0.0, profiles.similarity("north", "east"));
    Assertions.assertEquals(0.0, profiles.similarity("north", "west"));
    Assertions.assertEquals(-1.0, profiles.similarity("north", "south"));
    Assertions.assertEquals(1.0, profiles.similarity("north", "north"));
    Assertions.assertEquals(3, profiles.mostSimilar("north", 10).size());
  }

  /**
   * Asserts that a table whose columns are all constant produces zero vectors for
   * every place, so every similarity is exactly {@code 0.0}, including a place
   * compared against itself. This is the documented no-variance behavior rather than
   * a division by zero.
   */
  @Test
  void testZeroVarianceProfilesScoreZeroEvenAgainstThemselves() throws IOException {
    final PlaceProfiles profiles = load(String.join("\n",
        "id\ta\tb",
        "flat-1\t5\t7",
        "flat-2\t5\t7",
        ""));
    Assertions.assertEquals(0.0, profiles.similarity("flat-1", "flat-1"));
    Assertions.assertEquals(0.0, profiles.similarity("flat-1", "flat-2"));
  }

  /**
   * Asserts that a row listing fewer values than the header declares is rejected at
   * load with a message naming the row and both field counts; a place can therefore
   * never silently miss an attribute another place has.
   */
  @Test
  void testRowMissingAValueFailsLoud() {
    final IOException e = Assertions.assertThrows(IOException.class,
        () -> load("id\ta\tb\nlonely\t1\n"));
    Assertions.assertEquals("row 2 has 2 fields, expected 3", e.getMessage());
  }

  /**
   * Asserts that a cell reading {@code NaN}, a routine export artifact for a missing
   * metric, is rejected at load with a message naming the row and the offending value.
   * A single such cell would otherwise standardize the whole column to {@code NaN} and,
   * through the column mean, every profile in the table, so the rejection has to happen
   * before standardization rather than at query time. The same table with the cell
   * repaired loads and scores exactly as it would have without the bad cell.
   */
  @Test
  void testNotANumberValueFailsLoud() throws IOException {
    final IOException e = Assertions.assertThrows(IOException.class,
        () -> load(String.join("\n",
            "id\tx\ty",
            "north\t1\tNaN",
            "south\t-1\t0",
            "east\t0\t1",
            "west\t0\t-1",
            "")));
    Assertions.assertEquals("non-finite value in row 2: NaN", e.getMessage());
    final PlaceProfiles repaired = load(String.join("\n",
        "id\tx\ty",
        "north\t1\t0",
        "south\t-1\t0",
        "east\t0\t1",
        "west\t0\t-1",
        ""));
    Assertions.assertEquals(-1.0, repaired.similarity("north", "south"));
    Assertions.assertEquals(0.0, repaired.similarity("north", "east"));
    Assertions.assertEquals(3, repaired.mostSimilar("north", 5).size());
  }

  /**
   * Asserts that both infinite values are rejected at load with a message naming the
   * row and the offending value, for the same reason {@code NaN} is: they survive
   * {@link Double#parseDouble} and then propagate through the column statistics.
   */
  @Test
  void testInfiniteValuesFailLoud() {
    final IOException positive = Assertions.assertThrows(IOException.class,
        () -> load("id\ta\nparis\tInfinity\nlyon\t2.0\n"));
    Assertions.assertEquals("non-finite value in row 2: Infinity", positive.getMessage());
    final IOException negative = Assertions.assertThrows(IOException.class,
        () -> load("id\ta\nparis\t-Infinity\nlyon\t2.0\n"));
    Assertions.assertEquals("non-finite value in row 2: -Infinity",
        negative.getMessage());
  }

  /**
   * Asserts that the Java float and double literal suffixes are rejected even though
   * they parse to ordinary finite values, because a metric table holds data rather than
   * Java source and a suffixed cell means the table was generated by something that
   * leaked its own literal syntax into the output.
   */
  @Test
  void testJavaLiteralSuffixValuesFailLoud() {
    final IOException floatSuffix = Assertions.assertThrows(IOException.class,
        () -> load("id\ta\nparis\t1.0f\nlyon\t2.0\n"));
    Assertions.assertEquals("malformed value in row 2: 1.0f", floatSuffix.getMessage());
    final IOException doubleSuffix = Assertions.assertThrows(IOException.class,
        () -> load("id\ta\nparis\t1.0d\nlyon\t2.0\n"));
    Assertions.assertEquals("malformed value in row 2: 1.0d", doubleSuffix.getMessage());
  }

  /**
   * Asserts that a cell that is not numeric at all is rejected with the same message
   * shape, naming the row and quoting the value the user has to go and fix.
   */
  @Test
  void testMalformedValueNamesTheOffendingCell() {
    final IOException e = Assertions.assertThrows(IOException.class,
        () -> load("id\ta\nparis\tnot-a-number\nlyon\t2.0\n"));
    Assertions.assertEquals("malformed value in row 2: not-a-number", e.getMessage());
  }

  /**
   * Asserts that a header consisting of only the identifier column is rejected,
   * because a profile without at least one metric cannot be compared.
   */
  @Test
  void testHeaderWithoutMetricsFailsLoud() {
    final IOException e = Assertions.assertThrows(IOException.class,
        () -> load("id\nlonely\n"));
    Assertions.assertEquals("the header must be: id, then at least one metric",
        e.getMessage());
  }

  /**
   * Asserts that {@code null} arguments to the stream loader and to the query methods
   * fail loud with {@link IllegalArgumentException} before any computation happens.
   */
  @Test
  void testNullArgumentsFailLoud() throws IOException {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> PlaceProfiles.load((InputStream) null));
    final PlaceProfiles profiles = load("id\ta\np\t1\nq\t2\n");
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> profiles.similarity(null, "p"));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> profiles.similarity("p", null));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> profiles.mostSimilar(null, 1));
  }

  /**
   * Asserts that metrics on wildly different scales contribute equally after
   * standardization: a column counted in billions and a column confined to the unit
   * interval carry the same weight, so two places on opposite sides of both metrics
   * mirror each other, and the place sitting exactly at the mean of every column
   * standardizes to the zero vector and scores {@code 0.0} against everything.
   */
  @Test
  void testMixedScaleMetricsContributeEqually() throws IOException {
    final PlaceProfiles profiles = load(String.join("\n",
        "id\tgiga\ttiny",
        "alpha\t9000000000\t0.9",
        "beta\t1000000000\t0.1",
        "gamma\t5000000000\t0.5",
        ""));
    Assertions.assertEquals(-1.0000000000000002,
        profiles.similarity("alpha", "beta"), 1e-15);
    Assertions.assertEquals(0.0, profiles.similarity("alpha", "gamma"));
    Assertions.assertEquals(1.0000000000000002,
        profiles.similarity("alpha", "alpha"), 1e-15);
  }

  /**
   * Asserts that values around {@code 1e150} still standardize to exact results: the
   * squared deviations stay finite, so opposite extremes score exactly {@code -1.0},
   * the place at the column mean scores exactly {@code 0.0}, and self-similarity is
   * exactly {@code 1.0}.
   */
  @Test
  void testHugeFiniteMagnitudesStandardizeSafely() throws IOException {
    final PlaceProfiles profiles = load(String.join("\n",
        "id\tv",
        "big\t3e150",
        "small\t1e150",
        "mid\t2e150",
        ""));
    Assertions.assertEquals(-1.0, profiles.similarity("big", "small"));
    Assertions.assertEquals(0.0, profiles.similarity("big", "mid"));
    Assertions.assertEquals(1.0, profiles.similarity("big", "big"));
  }

  /**
   * Asserts the behavior at magnitudes whose squared deviations overflow a double:
   * around {@code 1e200} the summed squares become infinite, the standard deviation
   * becomes infinite, every standardized value collapses to zero, and the column is
   * effectively treated as having no variance, so all similarities are exactly
   * {@code 0.0}, self-similarity included.
   */
  @Test
  void testMagnitudesBeyondSquaredDoubleRangeCollapseToNoVariance() throws IOException {
    final PlaceProfiles profiles = load(String.join("\n",
        "id\tv",
        "a\t3e200",
        "b\t1e200",
        ""));
    Assertions.assertEquals(0.0, profiles.similarity("a", "b"));
    Assertions.assertEquals(0.0, profiles.similarity("a", "a"));
  }

  /**
   * Asserts how exact ties rank: two places with identical raw rows score the same
   * bitwise-identical similarity against the query, they occupy the two adjacent top
   * positions in the ranking in ascending identifier order, and the clearly dissimilar
   * place ranks last. The tied pair is asserted as an exact sequence rather than as a
   * set, because the documented tie-break makes the sequence reproducible across runs.
   */
  @Test
  void testTiedScoresRankByAscendingIdentifier() throws IOException {
    final PlaceProfiles profiles = load(String.join("\n",
        "id\tm1\tm2",
        "query\t4\t1",
        "twin-b\t1\t3",
        "twin-a\t1\t3",
        "far\t9\t9",
        ""));
    final List<PlaceProfiles.Neighbor> neighbors = profiles.mostSimilar("query", 3);
    Assertions.assertEquals(3, neighbors.size());
    Assertions.assertEquals("twin-a", neighbors.get(0).id());
    Assertions.assertEquals("twin-b", neighbors.get(1).id());
    Assertions.assertEquals(neighbors.get(0).similarity(), neighbors.get(1).similarity());
    Assertions.assertEquals(0.29643507578021855, neighbors.get(0).similarity(), 1e-12);
    Assertions.assertEquals("far", neighbors.get(2).id());
    Assertions.assertEquals(-0.6651076860027197, neighbors.get(2).similarity(), 1e-12);
  }

  /**
   * Asserts that the top-{@code count} cut through a tie keeps a reproducible set of
   * places, not merely a reproducible order among whatever survives. Three places tie
   * at exactly the same score and only two are asked for, so the two lowest identifiers
   * are returned; without a tie-break the surviving pair would follow the iteration
   * order of the underlying immutable map, which the JDK randomizes per run, and a
   * place would silently enter and leave the result between runs of the same query.
   */
  @Test
  void testTiedTopCountCutKeepsTheLowestIdentifiers() throws IOException {
    final PlaceProfiles profiles = load(String.join("\n",
        "id\tm1\tm2",
        "query\t4\t1",
        "tie-c\t1\t3",
        "tie-a\t1\t3",
        "tie-b\t1\t3",
        ""));
    final List<PlaceProfiles.Neighbor> neighbors = profiles.mostSimilar("query", 2);
    Assertions.assertEquals(List.of("tie-a", "tie-b"),
        List.of(neighbors.get(0).id(), neighbors.get(1).id()));
    Assertions.assertEquals(neighbors.get(0).similarity(), neighbors.get(1).similarity());
  }

  /**
   * Asserts that an attribution comment placed ahead of the header, the natural place
   * to record the licenses of the sources a table was derived from, is ignored rather
   * than mistaken for the header. Blank lines ahead of the header are ignored the same
   * way, and the header is then read from the first line that carries content.
   */
  @Test
  void testCommentsAndBlankLinesBeforeTheHeaderAreIgnored() throws IOException {
    final PlaceProfiles profiles = load(String.join("\n",
        "# compass fixture, derived from a project-authored source",
        "",
        "id\tx\ty",
        "north\t1\t0",
        "south\t-1\t0",
        ""));
    Assertions.assertEquals(List.of("x", "y"), profiles.metrics());
    Assertions.assertTrue(profiles.contains("north"));
    Assertions.assertEquals(-1.0, profiles.similarity("north", "south"));
  }

  /**
   * Asserts that a table holding nothing but comments and blank space is rejected as
   * having no header, rather than treating a comment as the header.
   */
  @Test
  void testTableOfOnlyCommentsHasNoHeader() {
    final IOException e = Assertions.assertThrows(IOException.class,
        () -> load("# just a note\n\n# and another\n"));
    Assertions.assertEquals("the profile table has no header", e.getMessage());
  }

  /**
   * Asserts that the parser strips identifiers and values with the same whitespace
   * definition the rest of the toolkit uses, which counts the no-break space that
   * {@link String#trim()} leaves in place. A table copied out of a PDF or a rendered web
   * table routinely carries no-break spaces around its cells, and leaving one attached
   * would store the place under an unreachable identifier.
   */
  @Test
  void testNoBreakSpaceAroundCellsIsStripped() throws IOException {
    final PlaceProfiles profiles = load(String.join("\n",
        "id\tx\ty",
        "\u00A0north\u00A0\t\u00A01\t0",
        "south\t-1\t0",
        ""));
    Assertions.assertTrue(profiles.contains("north"));
    Assertions.assertEquals(-1.0, profiles.similarity("north", "south"));
  }

  /**
   * Asserts that when an identifier appears on more than one row, the last row wins:
   * the duplicated place ends up indistinguishable from the place sharing its final
   * values and mirrored against the remaining place, proving the earlier row was
   * discarded before standardization.
   */
  @Test
  void testDuplicateIdentifierKeepsLastRow() throws IOException {
    final PlaceProfiles profiles = load(String.join("\n",
        "id\ta\tb",
        "dup\t100\t100",
        "dup\t1\t2",
        "twin\t1\t2",
        "far\t9\t5",
        ""));
    Assertions.assertEquals(1.0, profiles.similarity("dup", "twin"), 1e-12);
    Assertions.assertEquals(-1.0, profiles.similarity("dup", "far"), 1e-12);
  }

  /**
   * Asserts that carriage-return line endings, comment lines starting with {@code #},
   * and blank lines are all tolerated: the table loads, the comment line does not
   * become a place, the four real places are present, and a mirrored pair still
   * scores exactly {@code -1.0}.
   */
  @Test
  void testCarriageReturnsCommentsAndBlankLinesAreIgnored() throws IOException {
    final String table = "id\tx\ty\r\n"
        + "# compass fixture with carriage returns and a comment\r\n"
        + "north\t1\t0\r\n"
        + "\r\n"
        + "south\t-1\t0\r\n"
        + "east\t0\t1\r\n"
        + "west\t0\t-1\r\n";
    final PlaceProfiles profiles = load(table);
    Assertions.assertEquals(List.of("x", "y"), profiles.metrics());
    Assertions.assertFalse(
        profiles.contains("# compass fixture with carriage returns and a comment"));
    Assertions.assertTrue(profiles.contains("north"));
    Assertions.assertEquals(3, profiles.mostSimilar("north", 10).size());
    Assertions.assertEquals(-1.0, profiles.similarity("north", "south"));
  }
}
