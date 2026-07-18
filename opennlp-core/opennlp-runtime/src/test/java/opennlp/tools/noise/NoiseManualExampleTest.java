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

package opennlp.tools.noise;

import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Runs the manual's noise examples (docbkx {@code noise.xml}) verbatim: every value the
 * chapter states is asserted here, so a change breaking this test breaks the manual.
 */
public class NoiseManualExampleTest {

  private static final String TEXT = "rnodern tirnes and zxkcvbnmsdfg";

  /** The scoring example prints exactly the two stated lines. */
  @Test
  void testScoringExamplePrintsTheStatedLines() {
    final Set<String> words = Set.of("modern", "times");
    final NoiseScorer scorer = new StructuralNoiseScorer(words::contains);
    final List<NoiseSpan> noise = scorer.score(TEXT, List.of());
    assertEquals(2, noise.size());
    assertEquals("misspelled [0..14)",
        noise.get(0).severity() + " " + noise.get(0).span());
    assertEquals("gibberish [19..31)",
        noise.get(1).severity() + " " + noise.get(1).span());
  }

  /** The chapter's contrast: without a dictionary only the gibberish is reported. */
  @Test
  void testWithoutADictionaryOnlyTheGibberishRemains() {
    final List<NoiseSpan> noise = new StructuralNoiseScorer().score(TEXT, List.of());
    assertEquals(1, noise.size());
    assertEquals(NoiseSpan.SEVERITY_GIBBERISH, noise.get(0).severity());
  }
}
