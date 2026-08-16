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
package opennlp.embeddings.eval;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import opennlp.embeddings.StaticEmbeddingModel;
import opennlp.embeddings.corpus.CasePassage;
import opennlp.embeddings.corpus.DictionaryEntry;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The evaluator on a tiny deterministic fixture: metric ranges and structure, the self-labeling
 * ground truth, determinism across runs, and both report renderings.
 */
class SearchEvaluatorTest {

  private static final List<CasePassage> PASSAGES = List.of(
      new CasePassage("p-king", "King v. Queen", "1 U.S. 1", "1900-01-01", "1",
          "king queen king queen"),
      new CasePassage("p-man", "Man v. Woman", "1 U.S. 2", "1900-01-02", "1",
          "man woman man woman"),
      new CasePassage("p-apple", "In re Apple", "1 U.S. 3", "1900-01-03", "1",
          "apple apple apple apple"));

  private static final List<DictionaryEntry> DICTIONARY = List.of(
      new DictionaryEntry("KING", "royal man on the throne, a king"),
      new DictionaryEntry("APPLE", "an apple, a fruit"));

  @TempDir
  static Path modelDir;
  private static StaticEmbeddingModel model;

  @BeforeAll
  static void loadModel() throws IOException {
    opennlp.embeddings.EmbeddingTestFixtures.writeAnalogyDirectory(modelDir);
    model = StaticEmbeddingModel.load(modelDir);
  }

  private static SearchEvaluator.Report run() {
    return SearchEvaluator.run(model, PASSAGES, DICTIONARY, 4, 42, 2);
  }

  @Test
  void testMetricsAreCompleteAndInRange() {
    final SearchEvaluator.Report report = run();

    assertEquals(3, report.passageCount());
    assertEquals(2, report.headwordCount());
    assertEquals(model.dimension(), report.dimension());
    assertEquals(4, report.bits());
    assertEquals(2, report.topK());
    assertEquals(3, report.flat().rows());
    assertEquals(3, report.quantized().rows());
    assertTrue(report.flat().bytesPerVector() > 0);
    assertTrue(report.quantized().bytesPerVector() > 0);
    assertTrue(report.flat().queriesPerSecond() > 0);
    assertTrue(report.quantized().queriesPerSecond() > 0);
    for (final double ratio : new double[] {report.fidelityRecallAtK(),
        report.fidelityAgreement()}) {
      assertTrue(ratio >= 0 && ratio <= 1, "ratio " + ratio);
    }
    assertEquals(2, report.definitionToHeadword().size());
    assertEquals(2, report.halfPassage().size());
  }

  @Test
  void testTheSelfLabelingGroundTruthIsRecovered() {
    final SearchEvaluator.Report report = run();

    // Three well-separated passages at k = 2: the quantized top-2 matches the exact top-2.
    assertEquals(1.0, report.fidelityRecallAtK(), 1e-12);
    assertEquals(1.0, report.fidelityAgreement(), 1e-12);
    // The definitions name their headwords' words, so both indexes solve the retrieval.
    for (final SearchEvaluator.RetrievalMetrics m : report.definitionToHeadword()) {
      assertEquals(2, m.queries());
      assertEquals(1.0, m.recallAt1(), 1e-12, m.name());
      assertEquals(1.0, m.mrr(), 1e-12, m.name());
    }
    // A half passage repeats the same words, so it embeds onto its own passage exactly.
    for (final SearchEvaluator.RetrievalMetrics m : report.halfPassage()) {
      assertEquals(3, m.queries());
      assertEquals(1.0, m.recallAt1(), 1e-12, m.name());
    }
  }

  @Test
  void testTheEvaluationIsDeterministicApartFromTimings() {
    final SearchEvaluator.Report first = run();
    final SearchEvaluator.Report second = run();

    assertEquals(first.fidelityRecallAtK(), second.fidelityRecallAtK());
    assertEquals(first.fidelityAgreement(), second.fidelityAgreement());
    assertEquals(first.definitionToHeadword(), second.definitionToHeadword());
    assertEquals(first.halfPassage(), second.halfPassage());
  }

  @Test
  void testTheMarkdownAndTsvRenderTheSameNumbers() {
    final SearchEvaluator.Report report = run();
    final String markdown = report.toMarkdown();
    final String tsv = report.toTsv();

    assertTrue(markdown.contains("| exact |"), markdown);
    assertTrue(markdown.contains("| turboquant |"), markdown);
    assertTrue(markdown.contains("recall@2 vs exact | 1.000"), markdown);
    assertTrue(tsv.contains("fidelity.recallAtK\t1.000"), tsv);
    assertTrue(tsv.contains("passages\t3"), tsv);
    assertTrue(tsv.contains("definitionToHeadword.exact.mrr\t1.000"), tsv);
  }

  @Test
  void testAnEmptyDictionarySkipsThatEvaluation() {
    final SearchEvaluator.Report report =
        SearchEvaluator.run(model, PASSAGES, List.of(), 4, 42, 2);
    assertEquals(0, report.headwordCount());
    assertTrue(report.definitionToHeadword().isEmpty());
    // The markdown still renders, with an empty section table.
    assertTrue(report.toMarkdown().contains("Definition to headword"));
  }

  @Test
  void testInvalidArgumentsAreRejected() {
    assertThrows(IllegalArgumentException.class,
        () -> SearchEvaluator.run(null, PASSAGES, DICTIONARY, 4, 42, 2));
    assertThrows(IllegalArgumentException.class,
        () -> SearchEvaluator.run(model, null, DICTIONARY, 4, 42, 2));
    assertThrows(IllegalArgumentException.class,
        () -> SearchEvaluator.run(model, List.of(), DICTIONARY, 4, 42, 2));
    assertThrows(IllegalArgumentException.class,
        () -> SearchEvaluator.run(model, PASSAGES, null, 4, 42, 2));
    assertThrows(IllegalArgumentException.class,
        () -> SearchEvaluator.run(model, PASSAGES, DICTIONARY, 4, 42, 0));
    assertThrows(IllegalArgumentException.class,
        () -> SearchEvaluator.run(model, PASSAGES, DICTIONARY, 9, 42, 2));
  }
}
