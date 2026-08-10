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

package opennlp.tools.pii;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import opennlp.tools.document.Document;

/**
 * What a scan found, in a form that is safe to log: how many mentions of each type, how many
 * of them were distinct, and a keyed token per distinct value instead of the value.
 *
 * <p>A report is the artefact a review actually needs. Whether a redaction pipeline is
 * working is a question about counts and about which types appear where, and answering it by
 * printing the mentions recreates the exposure the pipeline exists to prevent. Nothing here
 * carries a value or an offset: two reports of the same document are identical, and a report
 * of a document with one address repeated four times says exactly that without saying which
 * address.</p>
 *
 * <p>The tokens come from an {@link HmacTokenizer}, so they agree with the tokens of a
 * {@link HmacTokenizer#rewrite(CharSequence, List) tokenized} copy of the text and across
 * every report made under the same key. That is what lets a reviewer ask whether the value
 * behind {@code EMAIL-3f2a1c9d} in one report is the one in another, without either report
 * holding it.</p>
 *
 * <p>Instances are immutable and safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public final class PiiAuditReport {

  /** How many distinct values per type a report names unless asked otherwise. */
  private static final int DEFAULT_SAMPLES = 3;

  private final Map<String, Integer> counts;
  private final Map<String, Integer> distinctCounts;
  private final Map<String, List<String>> samples;
  private final int total;

  private PiiAuditReport(Map<String, Integer> counts, Map<String, Integer> distinctCounts,
      Map<String, List<String>> samples, int total) {
    this.counts = counts;
    this.distinctCounts = distinctCounts;
    this.samples = samples;
    this.total = total;
  }

  /**
   * Reports on a list of mentions.
   *
   * @param mentions The mentions to report on. Must not be {@code null} or contain
   *                 {@code null}; may be empty.
   * @param tokenizer Derives the token for each distinct value. Must not be {@code null}.
   * @return The report. Never {@code null}.
   * @throws IllegalArgumentException Thrown if an argument is {@code null} or a mention is
   *         {@code null}.
   */
  public static PiiAuditReport of(List<PiiMention> mentions, HmacTokenizer tokenizer) {
    return of(mentions, tokenizer, DEFAULT_SAMPLES);
  }

  /**
   * Reports on a list of mentions, naming at most {@code samplesPerType} distinct values of
   * each type.
   *
   * @param mentions The mentions to report on. Must not be {@code null} or contain
   *                 {@code null}; may be empty.
   * @param tokenizer Derives the token for each distinct value. Must not be {@code null}.
   * @param samplesPerType How many tokens to keep per type. Must not be negative; zero
   *                       reports counts only.
   * @return The report. Never {@code null}.
   * @throws IllegalArgumentException Thrown if an argument is {@code null}, a mention is
   *         {@code null}, or {@code samplesPerType} is negative.
   */
  public static PiiAuditReport of(List<PiiMention> mentions, HmacTokenizer tokenizer,
      int samplesPerType) {
    if (mentions == null) {
      throw new IllegalArgumentException("mentions must not be null");
    }
    if (tokenizer == null) {
      throw new IllegalArgumentException("tokenizer must not be null");
    }
    if (samplesPerType < 0) {
      throw new IllegalArgumentException("samplesPerType must not be negative");
    }
    final Map<String, Integer> counts = new TreeMap<>();
    final Map<String, Set<String>> distinct = new TreeMap<>();
    final Map<String, List<String>> samples = new TreeMap<>();
    int total = 0;
    for (final PiiMention mention : mentions) {
      if (mention == null) {
        throw new IllegalArgumentException("mentions must not contain null");
      }
      final String type = mention.type();
      counts.merge(type, 1, Integer::sum);
      total++;
      final String token = tokenizer.token(mention);
      final boolean unseen = distinct.computeIfAbsent(type, ignored -> new LinkedHashSet<>())
          .add(token);
      if (unseen) {
        final List<String> kept = samples.computeIfAbsent(type, ignored -> new ArrayList<>());
        if (kept.size() < samplesPerType) {
          kept.add(token);
        }
      }
    }
    final Map<String, Integer> distinctCounts = new TreeMap<>();
    final Map<String, List<String>> sampleView = new LinkedHashMap<>();
    for (final Map.Entry<String, Set<String>> entry : distinct.entrySet()) {
      distinctCounts.put(entry.getKey(), entry.getValue().size());
      sampleView.put(entry.getKey(), List.copyOf(
          samples.getOrDefault(entry.getKey(), List.of())));
    }
    return new PiiAuditReport(Collections.unmodifiableMap(counts),
        Collections.unmodifiableMap(distinctCounts),
        Collections.unmodifiableMap(sampleView), total);
  }

  /**
   * Reports on a document's {@link PiiAnnotator#PII} layer.
   *
   * @param document The document to report on. Must not be {@code null} and must carry the
   *                 {@link PiiAnnotator#PII} layer.
   * @param tokenizer Derives the token for each distinct value. Must not be {@code null}.
   * @return The report. Never {@code null}.
   * @throws IllegalArgumentException Thrown if an argument is {@code null} or the document
   *         does not carry the PII layer.
   */
  public static PiiAuditReport of(Document document, HmacTokenizer tokenizer) {
    return of(PiiLayer.mentions(document), tokenizer);
  }

  /**
   * Returns how many mentions of each type were found.
   *
   * @return The counts by type, in type order. Never {@code null}; immutable. A type that
   *         was not found is absent rather than zero.
   */
  public Map<String, Integer> counts() {
    return counts;
  }

  /**
   * Returns how many distinct values of each type were found. One address mentioned four
   * times counts once here and four times in {@link #counts()}, and the gap between the two
   * is often the interesting part: a form letter with one recipient looks quite different
   * from a leaked list.
   *
   * @return The distinct value counts by type, in type order. Never {@code null};
   *         immutable.
   */
  public Map<String, Integer> distinctCounts() {
    return distinctCounts;
  }

  /**
   * Returns the types that were found.
   *
   * @return The types, in type order. Never {@code null}; immutable.
   */
  public Set<String> types() {
    return counts.keySet();
  }

  /**
   * Returns the total number of mentions.
   *
   * @return The total, {@code 0} for a text without PII.
   */
  public int total() {
    return total;
  }

  /**
   * Returns the kept tokens for one type, in the order the values were first seen.
   *
   * @param type The mention type. Must not be {@code null}.
   * @return The tokens, at most as many as the report was asked to keep. Never
   *         {@code null}; empty for a type that was not found. Immutable.
   * @throws IllegalArgumentException Thrown if {@code type} is {@code null}.
   */
  public List<String> samples(String type) {
    if (type == null) {
      throw new IllegalArgumentException("type must not be null");
    }
    return samples.getOrDefault(type, List.of());
  }

  /**
   * Formats the report as one line per type, for a log or a build output.
   *
   * @return The formatted report, ending without a line separator. Never {@code null}.
   */
  @Override
  public String toString() {
    if (total == 0) {
      return "no pii found";
    }
    final StringBuilder out = new StringBuilder();
    for (final Map.Entry<String, Integer> entry : counts.entrySet()) {
      final String type = entry.getKey();
      if (out.length() > 0) {
        out.append(System.lineSeparator());
      }
      out.append(type).append(": ").append(entry.getValue()).append(" mentions, ")
          .append(distinctCounts.get(type)).append(" distinct");
      final List<String> kept = samples(type);
      if (!kept.isEmpty()) {
        out.append(' ').append(kept);
      }
    }
    return out.toString();
  }
}
