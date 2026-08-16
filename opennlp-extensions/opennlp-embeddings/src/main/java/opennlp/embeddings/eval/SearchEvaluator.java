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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import opennlp.embeddings.StaticEmbeddingModel;
import opennlp.embeddings.corpus.CasePassage;
import opennlp.embeddings.corpus.DictionaryEntry;
import opennlp.embeddings.index.FlatFloatIndex;
import opennlp.embeddings.index.TurboQuantIndex;
import opennlp.embeddings.index.VectorIndex;
import opennlp.tools.util.java.Experimental;

/**
 * Runs the zero-label evaluation loop over a static embedding model, a passage corpus, and a
 * dictionary: embed everything, build the exact and the quantized index, and measure what the
 * quantization costs and what the search stack retrieves, without any hand-labeled relevance
 * judgments.
 *
 * <p>Three evaluations, each self-labeling:</p>
 * <ol>
 *   <li><b>Index fidelity</b>: every passage vector queries both indexes; the quantized index's
 *   overlap with the exact top-k and its rank-1 agreement measure pure quantization loss.</li>
 *   <li><b>Definition to headword</b>: each dictionary definition queries an index of headword
 *   embeddings; the definition's own headword is the relevant answer.</li>
 *   <li><b>Half passage</b>: the first half of each passage queries the passage index; the
 *   passage itself is the relevant answer.</li>
 * </ol>
 *
 * <p>The run is deterministic apart from the wall-clock timings: the quantization seed is
 * fixed by the caller and the queries are the inputs in order.</p>
 *
 * <p>Warning: Experimental new feature; the API might change in a later release.</p>
 */
@Experimental
public final class SearchEvaluator {

  /** Not instantiable. */
  private SearchEvaluator() {
  }

  /**
   * The build and throughput measurements of one index.
   *
   * @param name             The index's display name.
   * @param rows             The number of indexed vectors.
   * @param bytesPerVector   The in-memory storage cost of one vector.
   * @param buildMillis      The freeze time in milliseconds.
   * @param queriesPerSecond Single-thread queries per second, measured after a warm-up pass.
   */
  public record IndexMetrics(String name, int rows, double bytesPerVector, long buildMillis,
                             double queriesPerSecond) {
  }

  /**
   * The outcome of one retrieval evaluation on one index.
   *
   * @param name      The index's display name.
   * @param queries   The number of queries with a usable (non-zero) embedding.
   * @param mrr       The mean reciprocal rank at the evaluated depth; a target beyond the depth
   *                  contributes zero.
   * @param recallAt1 The share of queries whose target ranked first.
   * @param recallAtK The share of queries whose target ranked within the evaluated depth.
   */
  public record RetrievalMetrics(String name, int queries, double mrr, double recallAt1,
                                 double recallAtK) {
  }

  /**
   * One complete evaluation run.
   *
   * @param passageCount        The number of indexed passages.
   * @param headwordCount       The number of indexed dictionary headwords.
   * @param dimension           The embedding dimension.
   * @param vocabularySize      The model's subword vocabulary size.
   * @param termCount           The model's term-row count.
   * @param bits                The quantization bit width.
   * @param topK                The evaluated depth.
   * @param embedMillis         The time to embed every passage, in milliseconds.
   * @param flat                The exact passage index's build and throughput metrics.
   * @param quantized           The quantized passage index's build and throughput metrics.
   * @param fidelityRecallAtK   The quantized index's mean overlap with the exact top-k.
   * @param fidelityAgreement   The share of queries where both indexes rank the same id first.
   * @param definitionToHeadword The definition-to-headword metrics, exact then quantized.
   * @param halfPassage         The half-passage metrics, exact then quantized.
   */
  public record Report(int passageCount, int headwordCount, int dimension, int vocabularySize,
                       int termCount, int bits, int topK, long embedMillis,
                       IndexMetrics flat, IndexMetrics quantized,
                       double fidelityRecallAtK, double fidelityAgreement,
                       List<RetrievalMetrics> definitionToHeadword,
                       List<RetrievalMetrics> halfPassage) {

    /** {@return the human-readable report as GitHub-flavored markdown} */
    public String toMarkdown() {
      final StringBuilder md = new StringBuilder(4096);
      md.append("# Vector search evaluation\n\n");
      md.append("Model: ").append(vocabularySize).append(" subword rows, ")
          .append(termCount).append(" term rows, dimension ").append(dimension)
          .append(". Corpus: ").append(passageCount).append(" passages, ")
          .append(headwordCount).append(" dictionary headwords. Quantization: ")
          .append(bits).append(" bits per dimension. Evaluation depth: top ")
          .append(topK).append(".\n\n");
      md.append("Embedding the passages took ").append(embedMillis).append(" ms.\n\n");

      md.append("## Passage index build and throughput\n\n");
      md.append("| index | rows | bytes/vector | build (ms) | QPS (1 thread) |\n");
      md.append("|---|---|---|---|---|\n");
      for (final IndexMetrics index : List.of(flat, quantized)) {
        md.append("| ").append(index.name())
            .append(" | ").append(index.rows())
            .append(" | ").append(format(index.bytesPerVector()))
            .append(" | ").append(index.buildMillis())
            .append(" | ").append(String.format(Locale.ROOT, "%.0f", index.queriesPerSecond()))
            .append(" |\n");
      }
      md.append('\n');

      md.append("## Index fidelity (quantized vs exact, passages as queries)\n\n");
      md.append("| metric | value |\n|---|---|\n");
      md.append("| recall@").append(topK).append(" vs exact | ")
          .append(format(fidelityRecallAtK)).append(" |\n");
      md.append("| rank-1 agreement | ").append(format(fidelityAgreement)).append(" |\n\n");

      appendRetrieval(md, "Definition to headword retrieval", definitionToHeadword);
      appendRetrieval(md, "Half-passage retrieval", halfPassage);
      return md.toString();
    }

    /**
     * Appends one retrieval section.
     *
     * @param md      The markdown under construction.
     * @param title   The section title.
     * @param metrics The per-index metrics.
     */
    private void appendRetrieval(StringBuilder md, String title,
                                 List<RetrievalMetrics> metrics) {
      md.append("## ").append(title).append("\n\n");
      md.append("| index | queries | MRR@").append(topK)
          .append(" | recall@1 | recall@").append(topK).append(" |\n");
      md.append("|---|---|---|---|---|\n");
      for (final RetrievalMetrics m : metrics) {
        md.append("| ").append(m.name())
            .append(" | ").append(m.queries())
            .append(" | ").append(format(m.mrr()))
            .append(" | ").append(format(m.recallAt1()))
            .append(" | ").append(format(m.recallAtK()))
            .append(" |\n");
      }
      md.append('\n');
    }

    /** {@return the machine-readable report: one {@code key<TAB>value} line per metric} */
    public String toTsv() {
      final StringBuilder tsv = new StringBuilder(1024);
      line(tsv, "passages", passageCount);
      line(tsv, "headwords", headwordCount);
      line(tsv, "dimension", dimension);
      line(tsv, "vocabulary.subwords", vocabularySize);
      line(tsv, "vocabulary.terms", termCount);
      line(tsv, "bits", bits);
      line(tsv, "topK", topK);
      line(tsv, "embed.millis", embedMillis);
      for (final IndexMetrics index : List.of(flat, quantized)) {
        line(tsv, index.name() + ".bytesPerVector", format(index.bytesPerVector()));
        line(tsv, index.name() + ".build.millis", index.buildMillis());
        line(tsv, index.name() + ".qps", String.format(Locale.ROOT, "%.0f",
            index.queriesPerSecond()));
      }
      line(tsv, "fidelity.recallAtK", format(fidelityRecallAtK));
      line(tsv, "fidelity.rank1Agreement", format(fidelityAgreement));
      for (final RetrievalMetrics m : definitionToHeadword) {
        retrievalLines(tsv, "definitionToHeadword." + m.name(), m);
      }
      for (final RetrievalMetrics m : halfPassage) {
        retrievalLines(tsv, "halfPassage." + m.name(), m);
      }
      return tsv.toString();
    }

    /**
     * Appends the three metric lines of one retrieval outcome.
     *
     * @param tsv    The TSV under construction.
     * @param prefix The metric key prefix.
     * @param m      The metrics.
     */
    private static void retrievalLines(StringBuilder tsv, String prefix, RetrievalMetrics m) {
      line(tsv, prefix + ".queries", m.queries());
      line(tsv, prefix + ".mrr", format(m.mrr()));
      line(tsv, prefix + ".recallAt1", format(m.recallAt1()));
      line(tsv, prefix + ".recallAtK", format(m.recallAtK()));
    }

    /**
     * Appends one {@code key<TAB>value} line.
     *
     * @param tsv   The TSV under construction.
     * @param key   The metric key.
     * @param value The metric value.
     */
    private static void line(StringBuilder tsv, String key, Object value) {
      tsv.append(key).append('\t').append(value).append('\n');
    }

    /** {@return a ratio formatted with three decimals, locale-independent} */
    private static String format(double value) {
      return String.format(Locale.ROOT, "%.3f", value);
    }
  }

  /**
   * Runs the full evaluation.
   *
   * @param model      The embedding model. Must not be {@code null}.
   * @param passages   The passages to index and query. Must not be {@code null} or empty.
   * @param dictionary The dictionary entries for the definition-to-headword evaluation. Must
   *                   not be {@code null}; may be empty, which skips that evaluation's queries.
   * @param bits       The quantization bit width, as in
   *                   {@link TurboQuantIndex#TurboQuantIndex(int, int, long)}.
   * @param seed       The quantization rotation seed.
   * @param topK       The evaluation depth. Must be at least 1.
   * @return The report.
   * @throws IllegalArgumentException Thrown if an argument is {@code null} or out of range, or
   *     {@code passages} is empty.
   */
  public static Report run(StaticEmbeddingModel model, List<CasePassage> passages,
                           List<DictionaryEntry> dictionary, int bits, long seed, int topK) {
    if (model == null) {
      throw new IllegalArgumentException("Model must not be null");
    }
    if (passages == null || passages.isEmpty()) {
      throw new IllegalArgumentException("Passages must not be null or empty");
    }
    if (dictionary == null) {
      throw new IllegalArgumentException("Dictionary must not be null");
    }
    if (topK < 1) {
      throw new IllegalArgumentException("TopK must be at least 1, got " + topK);
    }

    final long embedStart = System.nanoTime();
    final float[][] passageVectors = new float[passages.size()][];
    for (int i = 0; i < passages.size(); i++) {
      passageVectors[i] = model.embed(passages.get(i).text());
    }
    final long embedMillis = millisSince(embedStart);

    final FlatFloatIndex flat = new FlatFloatIndex(model.dimension());
    final TurboQuantIndex quantized = new TurboQuantIndex(model.dimension(), bits, seed);
    for (int i = 0; i < passages.size(); i++) {
      flat.add(passages.get(i).id(), passageVectors[i]);
      quantized.add(passages.get(i).id(), passageVectors[i]);
    }
    final long flatBuildStart = System.nanoTime();
    flat.freeze();
    final long flatBuildMillis = millisSince(flatBuildStart);
    final long quantizedBuildStart = System.nanoTime();
    quantized.freeze();
    final long quantizedBuildMillis = millisSince(quantizedBuildStart);

    // Fidelity: quantized against exact on every passage vector; this pass also warms both
    // indexes for the throughput measurement after it.
    double overlap = 0;
    int agreement = 0;
    for (final float[] query : passageVectors) {
      final List<VectorIndex.Hit> exact = flat.topK(query, topK);
      final List<VectorIndex.Hit> approximate = quantized.topK(query, topK);
      final Set<String> truth = new HashSet<>(exact.size() * 2);
      for (final VectorIndex.Hit hit : exact) {
        truth.add(hit.id());
      }
      for (final VectorIndex.Hit hit : approximate) {
        if (truth.contains(hit.id())) {
          overlap++;
        }
      }
      if (!exact.isEmpty() && !approximate.isEmpty()
          && exact.get(0).id().equals(approximate.get(0).id())) {
        agreement++;
      }
    }
    final double fidelityRecall = overlap / ((double) passages.size() * topK);
    final double fidelityAgreement = agreement / (double) passages.size();

    final IndexMetrics flatMetrics = new IndexMetrics("exact", flat.size(),
        model.dimension() * (double) Float.BYTES, flatBuildMillis,
        queriesPerSecond(flat, passageVectors, topK));
    final IndexMetrics quantizedMetrics = new IndexMetrics("turboquant", quantized.size(),
        quantized.bytesPerVector(), quantizedBuildMillis,
        queriesPerSecond(quantized, passageVectors, topK));

    // Definition to headword: an index of headword embeddings queried by definitions.
    final List<RetrievalMetrics> definitionToHeadword = new ArrayList<>(2);
    if (!dictionary.isEmpty()) {
      final FlatFloatIndex flatHeadwords = new FlatFloatIndex(model.dimension());
      final TurboQuantIndex quantizedHeadwords =
          new TurboQuantIndex(model.dimension(), bits, seed);
      for (final DictionaryEntry entry : dictionary) {
        final float[] vector = model.embed(entry.headword());
        flatHeadwords.add(entry.headword(), vector);
        quantizedHeadwords.add(entry.headword(), vector);
      }
      flatHeadwords.freeze();
      quantizedHeadwords.freeze();
      final List<String> targets = new ArrayList<>(dictionary.size());
      final List<float[]> queries = new ArrayList<>(dictionary.size());
      for (final DictionaryEntry entry : dictionary) {
        queries.add(model.embed(entry.definition()));
        targets.add(entry.headword());
      }
      definitionToHeadword.add(retrieval("exact", flatHeadwords, queries, targets, topK));
      definitionToHeadword.add(
          retrieval("turboquant", quantizedHeadwords, queries, targets, topK));
    }

    // Half passage: the first half of each passage queries the passage index.
    final List<String> passageTargets = new ArrayList<>(passages.size());
    final List<float[]> halfQueries = new ArrayList<>(passages.size());
    for (final CasePassage passage : passages) {
      halfQueries.add(model.embed(firstHalf(passage.text())));
      passageTargets.add(passage.id());
    }
    final List<RetrievalMetrics> halfPassage = List.of(
        retrieval("exact", flat, halfQueries, passageTargets, topK),
        retrieval("turboquant", quantized, halfQueries, passageTargets, topK));

    return new Report(passages.size(), dictionary.size(), model.dimension(),
        model.vocabularySize(), model.termCount(), bits, topK, embedMillis,
        flatMetrics, quantizedMetrics, fidelityRecall, fidelityAgreement,
        List.copyOf(definitionToHeadword), halfPassage);
  }

  /**
   * Scores one retrieval evaluation: each query's target contributes its reciprocal rank when
   * it appears in the index's top {@code topK}, zero when it does not. Queries whose embedding
   * has no direction (the index answers nothing) are dropped from the denominator and counted
   * out of {@code queries}.
   *
   * @param name    The index's display name.
   * @param index   The index to query.
   * @param queries The query vectors.
   * @param targets The relevant id of each query, aligned with {@code queries}.
   * @param topK    The evaluation depth.
   * @return The metrics.
   */
  private static RetrievalMetrics retrieval(String name, VectorIndex index,
                                            List<float[]> queries, List<String> targets,
                                            int topK) {
    int usable = 0;
    double reciprocalRankSum = 0;
    int atOne = 0;
    int withinK = 0;
    for (int i = 0; i < queries.size(); i++) {
      final List<VectorIndex.Hit> hits = index.topK(queries.get(i), topK);
      if (hits.isEmpty()) {
        continue;
      }
      usable++;
      final String target = targets.get(i);
      for (int rank = 0; rank < hits.size(); rank++) {
        if (hits.get(rank).id().equals(target)) {
          reciprocalRankSum += 1.0 / (rank + 1);
          withinK++;
          if (rank == 0) {
            atOne++;
          }
          break;
        }
      }
    }
    final double denominator = Math.max(usable, 1);
    return new RetrievalMetrics(name, usable, reciprocalRankSum / denominator,
        atOne / denominator, withinK / denominator);
  }

  /**
   * Measures single-thread throughput: one timed pass of every query. The caller warms the
   * index (and the JIT) with an untimed pass first.
   *
   * @param index   The index to measure.
   * @param queries The query vectors.
   * @param topK    The result depth per query.
   * @return Queries per second.
   */
  private static double queriesPerSecond(VectorIndex index, float[][] queries, int topK) {
    final long start = System.nanoTime();
    for (final float[] query : queries) {
      index.topK(query, topK);
    }
    final double seconds = (System.nanoTime() - start) / 1e9;
    return queries.length / Math.max(seconds, 1e-9);
  }

  /**
   * {@return the first half of a text, cut at the last space before the midpoint when there is
   * one} A passage's first half still describes the same case, so the passage itself is the
   * relevant answer for it.
   *
   * @param text The passage text.
   */
  private static String firstHalf(String text) {
    final int midpoint = text.length() / 2;
    if (midpoint == 0) {
      return text;
    }
    final int cut = text.lastIndexOf(' ', midpoint);
    return text.substring(0, cut > 0 ? cut : midpoint);
  }

  /**
   * {@return the whole milliseconds elapsed since a {@link System#nanoTime()} mark}
   *
   * @param startNanos The mark.
   */
  private static long millisSince(long startNanos) {
    return (System.nanoTime() - startNanos) / 1_000_000;
  }
}
