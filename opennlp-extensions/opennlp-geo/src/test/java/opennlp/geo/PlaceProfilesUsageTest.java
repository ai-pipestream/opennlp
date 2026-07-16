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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Demonstrates the full place-profile workflow end to end: a caller assembles a small
 * tab-separated metric table, writes it to a file, loads it, and then queries pairwise
 * similarities and nearest-neighbor rankings. The five places and their metric values
 * are project-authored; no external metadata source is involved.
 *
 * <p>Every expected score in this class was produced by running exactly this table
 * through {@link PlaceProfiles}. The scores are floating-point results of standardizing
 * the columns and taking cosines, so they are asserted with a delta of {@code 1e-12},
 * which is tight enough to pin the value while tolerating only last-ulp variation in
 * summation order.</p>
 */
public class PlaceProfilesUsageTest {

  /**
   * A miniature profile table with three metrics per place: residential density in
   * people per square kilometer, median household income in dollars, and a transit
   * access score from 1 to 10. The two dense urban places resemble each other, the
   * suburb and the rural town sit progressively farther away, and the mid-sized mill
   * city lands in between.
   */
  private static final String TABLE = String.join("\n",
      "id\tdensity\tincome\ttransit",
      "metroville\t12000\t72000\t8",
      "harborview\t11000\t68000\t9",
      "suburb-glen\t1800\t85000\t3",
      "farmdale\t120\t46000\t1",
      "mill-city\t9500\t52000\t7",
      "");

  /** The profiles under test, loaded once from a real file for the whole class. */
  private static PlaceProfiles profiles;

  /**
   * Writes the table to a file inside the supplied temporary directory and loads it
   * through the {@link Path} entry point, mirroring how a caller supplies a table on
   * disk.
   *
   * @param tempDir A directory managed by the test framework. Never {@code null}.
   * @throws IOException Thrown if the fixture cannot be written or loaded.
   */
  @BeforeAll
  static void loadProfilesFromFile(@TempDir Path tempDir) throws IOException {
    final Path table = tempDir.resolve("profiles.tsv");
    Files.writeString(table, TABLE, StandardCharsets.UTF_8);
    profiles = PlaceProfiles.load(table);
  }

  /**
   * Asserts that the loaded table exposes its metric names in column order and answers
   * membership questions for listed and unlisted places.
   */
  @Test
  void testLoadedTableExposesMetricsAndMembership() {
    Assertions.assertEquals(List.of("density", "income", "transit"), profiles.metrics());
    Assertions.assertTrue(profiles.contains("metroville"));
    Assertions.assertTrue(profiles.contains("mill-city"));
    Assertions.assertFalse(profiles.contains("shangri-la"));
  }

  /**
   * Asserts the exact pairwise scores against the query place: the sibling urban place
   * scores highest, the mill city is mildly positive, the suburb is mildly negative,
   * and the rural town is strongly negative. Also asserts that self-similarity lands
   * one ulp below {@code 1.0}, because the square root of the squared norm is rounded
   * before the division.
   */
  @Test
  void testPairwiseSimilaritiesMatchComputedScores() {
    Assertions.assertEquals(0.94217361531876,
        profiles.similarity("metroville", "harborview"), 1e-12);
    Assertions.assertEquals(0.27243341210254074,
        profiles.similarity("metroville", "mill-city"), 1e-12);
    Assertions.assertEquals(-0.34971429214039584,
        profiles.similarity("metroville", "suburb-glen"), 1e-12);
    Assertions.assertEquals(-0.9684681668897906,
        profiles.similarity("metroville", "farmdale"), 1e-12);
    Assertions.assertEquals(0.9999999999999999,
        profiles.similarity("metroville", "metroville"), 1e-15);
  }

  /**
   * Asserts that a full-width neighbor query returns every other place, never the
   * query place itself, in strictly descending score order, and that each score equals
   * the corresponding pairwise similarity.
   */
  @Test
  void testMostSimilarReturnsFullDescendingRanking() {
    final List<PlaceProfiles.Neighbor> neighbors = profiles.mostSimilar("metroville", 4);
    Assertions.assertEquals(4, neighbors.size());
    Assertions.assertEquals("harborview", neighbors.get(0).id());
    Assertions.assertEquals("mill-city", neighbors.get(1).id());
    Assertions.assertEquals("suburb-glen", neighbors.get(2).id());
    Assertions.assertEquals("farmdale", neighbors.get(3).id());
    Assertions.assertEquals(0.94217361531876, neighbors.get(0).similarity(), 1e-12);
    Assertions.assertEquals(0.27243341210254074, neighbors.get(1).similarity(), 1e-12);
    Assertions.assertEquals(-0.34971429214039584, neighbors.get(2).similarity(), 1e-12);
    Assertions.assertEquals(-0.9684681668897906, neighbors.get(3).similarity(), 1e-12);
    for (final PlaceProfiles.Neighbor neighbor : neighbors) {
      Assertions.assertNotEquals("metroville", neighbor.id());
    }
  }

  /**
   * Asserts a truncated ranking from a different query place: the rural town's least
   * dissimilar neighbor is the suburb, followed by the mill city, and the requested
   * count caps the result at two entries.
   */
  @Test
  void testTruncatedRankingFromAnotherQueryPlace() {
    final List<PlaceProfiles.Neighbor> neighbors = profiles.mostSimilar("farmdale", 2);
    Assertions.assertEquals(2, neighbors.size());
    Assertions.assertEquals("suburb-glen", neighbors.get(0).id());
    Assertions.assertEquals("mill-city", neighbors.get(1).id());
    Assertions.assertEquals(0.15895265022433747, neighbors.get(0).similarity(), 1e-12);
    Assertions.assertEquals(-0.08091385909242901, neighbors.get(1).similarity(), 1e-12);
  }
}
