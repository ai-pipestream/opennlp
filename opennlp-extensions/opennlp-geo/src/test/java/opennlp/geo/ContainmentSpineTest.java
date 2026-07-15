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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import opennlp.tools.geo.PlaceAncestor;

/**
 * Tests the containment hierarchy against project-authored miniature tables; no
 * external hierarchy data is involved.
 */
public class ContainmentSpineTest {

  private static ContainmentSpine parkSlopeSpine() {
    return ContainmentSpine.builder()
        .add("85865587", "421205765", "Park Slope", "neighbourhood")
        .add("421205765", "85977539", "Brooklyn", "borough")
        .add("85977539", "85688543", "New York", "locality")
        .add("85688543", "85633793", "New York", "region")
        .add("85633793", null, "United States", "country")
        .build();
  }

  @Test
  void testAncestorsWalkOutward() {
    final List<PlaceAncestor> chain = parkSlopeSpine().ancestors("85865587");

    Assertions.assertEquals(4, chain.size());
    Assertions.assertEquals("Brooklyn", chain.get(0).name());
    Assertions.assertEquals("borough", chain.get(0).type());
    Assertions.assertEquals("locality", chain.get(1).type());
    Assertions.assertEquals("region", chain.get(2).type());
    Assertions.assertEquals("United States", chain.get(3).name());
    Assertions.assertEquals("country", chain.get(3).type());
  }

  @Test
  void testUnknownAndRootPlacesHaveNoAncestors() {
    Assertions.assertTrue(parkSlopeSpine().ancestors("999").isEmpty());
    Assertions.assertTrue(parkSlopeSpine().ancestors("85633793").isEmpty());
  }

  @Test
  void testCyclesTerminate() {
    final ContainmentSpine cyclic = ContainmentSpine.builder()
        .add("a", "b", "A", "t")
        .add("b", "a", "B", "t")
        .build();
    Assertions.assertEquals(1, cyclic.ancestors("a").size());
  }

  @Test
  void testNeutralTableLoads(@TempDir Path dir) throws IOException {
    final Path table = dir.resolve("containment.tsv");
    Files.write(table, String.join("\n",
        "# id\tparent\tname\ttype",
        "Q123\tQ60\tSoHo\tneighbourhood",
        "Q60\tQ1384\tNew York City\tcity",
        "Q1384\t\tNew York\tstate",
        "").getBytes(StandardCharsets.UTF_8));

    final ContainmentSpine spine = ContainmentSpine.builder().addTable(table).build();
    final List<PlaceAncestor> chain = spine.ancestors("Q123");
    Assertions.assertEquals(2, chain.size());
    Assertions.assertEquals("New York City", chain.get(0).name());
    Assertions.assertEquals("New York", chain.get(1).name());
  }

  @Test
  void testWofMetaCsvLoadsWithQuotedNames(@TempDir Path dir) throws IOException {
    final Path meta = dir.resolve("wof-locality-latest.csv");
    Files.write(meta, String.join("\n",
        "bbox,id,name,parent_id,placetype,source",
        "\"1,2,3,4\",85977539,Brooklyn,85688543,borough,mz",
        "\"5,6,7,8\",85688543,\"New York, the Big Apple\",-1,locality,mz",
        "").getBytes(StandardCharsets.UTF_8));

    final ContainmentSpine spine = ContainmentSpine.builder().addWofMeta(meta).build();
    final List<PlaceAncestor> chain = spine.ancestors("85977539");
    Assertions.assertEquals(1, chain.size());
    Assertions.assertEquals("New York, the Big Apple", chain.get(0).name());
    Assertions.assertEquals("locality", chain.get(0).type());
  }

  @Test
  void testMalformedTablesFailLoud(@TempDir Path dir) throws IOException {
    final Path bad = dir.resolve("bad.tsv");
    Files.write(bad, "onlyone\n".getBytes(StandardCharsets.UTF_8));
    Assertions.assertThrows(IOException.class,
        () -> ContainmentSpine.builder().addTable(bad));

    final Path noColumns = dir.resolve("meta.csv");
    Files.write(noColumns, "a,b,c\n1,2,3\n".getBytes(StandardCharsets.UTF_8));
    Assertions.assertThrows(IOException.class,
        () -> ContainmentSpine.builder().addWofMeta(noColumns));
  }

  @Test
  void testInvalidArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> ContainmentSpine.builder().build());
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> ContainmentSpine.builder().add(null, null, "n", "t"));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> parkSlopeSpine().ancestors(null));
  }
}
