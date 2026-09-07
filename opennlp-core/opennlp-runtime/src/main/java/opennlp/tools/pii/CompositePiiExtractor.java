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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;

/**
 * Combines PII extractors into one non-overlapping result in text order.
 *
 * <p>Candidates are selected by start offset, then descending length, then
 * {@link PiiTypePriority type priority}, then delegate order. A candidate overlapping an
 * accepted mention is omitted without truncating its span.</p>
 *
 * <p>Nested composites contribute their individual extractors to the same overlap pass,
 * in depth-first, left-to-right order. Grouping the same ordered extractors does not
 * change the result. Other extractors contribute their returned mentions; their internal
 * filtering is unchanged. {@link #extractors()} retains the configured nested structure.</p>
 *
 * <p>Thread safety depends on the delegates. See {@link PiiPacks} for built-in
 * combinations.</p>
 *
 * @since 3.0.0
 */
public final class CompositePiiExtractor implements PiiExtractor {

  /**
   * One candidate with its delegate order and type priority.
   *
   * @param start The candidate start offset, inclusive.
   * @param end The candidate end offset, exclusive.
   * @param order The index of the contributing individual extractor.
   * @param priority The {@link PiiTypePriority} rank of the candidate's type.
   * @param mention The candidate mention.
   */
  private record Hit(int start, int end, int order, int priority, PiiMention mention) {
  }

  private final List<PiiExtractor> extractors;

  /**
   * Initializes a composite over the given extractors.
   *
   * @param extractors The extractors to merge, in the order that breaks overlap ties.
   *                   Must not be {@code null} or empty and must not contain
   *                   {@code null}.
   * @throws IllegalArgumentException Thrown if {@code extractors} is {@code null} or
   *         empty, or contains {@code null}.
   */
  public CompositePiiExtractor(PiiExtractor... extractors) {
    this(extractors == null ? null : Arrays.asList(extractors));
  }

  /**
   * Initializes a composite over the given extractors.
   *
   * @param extractors The extractors to merge, in the order that breaks overlap ties.
   *                   Must not be {@code null} or empty and must not contain
   *                   {@code null}.
   * @throws IllegalArgumentException Thrown if {@code extractors} is {@code null} or
   *         empty, or contains {@code null}.
   */
  public CompositePiiExtractor(List<PiiExtractor> extractors) {
    if (extractors == null || extractors.isEmpty()) {
      throw new IllegalArgumentException("extractors must not be null or empty");
    }
    for (final PiiExtractor extractor : extractors) {
      if (extractor == null) {
        throw new IllegalArgumentException("extractors must not contain null");
      }
    }
    this.extractors = List.copyOf(extractors);
  }

  /**
   * Returns the configured delegates, including nested composites.
   *
   * @return The immutable delegate list in the supplied order.
   */
  public List<PiiExtractor> extractors() {
    return extractors;
  }

  /**
   * {@inheritDoc}
   *
   * <p>Scans the individual extractors and resolves all their candidates together.</p>
   *
   * @throws IllegalArgumentException Thrown if {@code text} is null, or a delegate
   *         returns a null result, a null mention or a mention outside the input text.
   */
  @Override
  public List<PiiMention> extract(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<Hit> hits = new ArrayList<>();
    final Deque<PiiExtractor> pending = new ArrayDeque<>(extractors);
    int order = 0;
    while (!pending.isEmpty()) {
      final PiiExtractor extractor = pending.removeFirst();
      if (extractor instanceof CompositePiiExtractor composite) {
        for (int i = composite.extractors.size() - 1; i >= 0; i--) {
          pending.addFirst(composite.extractors.get(i));
        }
      } else {
        for (final PiiMention mention : PiiExtraction.extract(extractor, text)) {
          hits.add(new Hit(mention.span().getStart(), mention.span().getEnd(), order,
              PiiTypePriority.rank(mention.type()), mention));
        }
        order++;
      }
    }
    hits.sort((a, b) -> {
      if (a.start() != b.start()) {
        return Integer.compare(a.start(), b.start());
      }
      if (a.end() != b.end()) {
        return Integer.compare(b.end(), a.end());
      }
      if (a.priority() != b.priority()) {
        return Integer.compare(a.priority(), b.priority());
      }
      return Integer.compare(a.order(), b.order());
    });
    final List<PiiMention> mentions = new ArrayList<>();
    int lastEnd = 0;
    for (final Hit hit : hits) {
      if (hit.start() >= lastEnd) {
        mentions.add(hit.mention());
        lastEnd = hit.end();
      }
    }
    return mentions;
  }
}
