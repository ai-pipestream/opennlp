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
import java.util.Arrays;
import java.util.List;

/**
 * A {@link PiiExtractor} that runs several extractors over the same text and merges what
 * they report into one non-overlapping list in text order.
 *
 * <p>Each delegate scans the full text independently, so delegates never influence one
 * another's decisions and a delegate may be reused in any number of composites. When
 * candidates from different delegates overlap, the same rule the single extractors use
 * decides: the leftmost candidate wins, then the longest, then the
 * {@link PiiTypePriority type priority}, and only then the delegate that was supplied
 * first. A candidate that overlaps an already accepted one is dropped, never truncated, so
 * every reported span is exactly what its extractor found.</p>
 *
 * <p>Deciding by type before delegate order is what makes a composite independent of how it
 * was assembled: a text in which two types claim the same span, an NHS number that is also
 * a validly formatted phone number for instance, yields the same mention whichever pack was
 * listed first. The delegate order only settles ties between types of equal priority, which
 * in practice means types this package does not name.</p>
 *
 * <p>Delegates that report the same span for the same type, for example two packs that
 * both carry the card scanner, therefore yield one mention rather than a duplicate.</p>
 *
 * <p>This extractor is as thread safe as its delegates; it holds no per-call state of its
 * own. See {@link PiiPacks} for ready-made combinations.</p>
 *
 * @since 3.0.0
 */
public final class CompositePiiExtractor implements PiiExtractor {

  /**
   * One candidate contributed by a delegate, held until overlap resolution decides which
   * candidates survive.
   *
   * @param start The candidate start offset, inclusive.
   * @param end The candidate end offset, exclusive.
   * @param order The index of the contributing delegate.
   * @param priority The {@link PiiTypePriority} rank of the candidate's type.
   * @param mention The mention to report if this candidate survives.
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
   * @return The delegates in the order they were supplied. Never {@code null}; the list
   *         is unmodifiable.
   */
  public List<PiiExtractor> extractors() {
    return extractors;
  }

  /**
   * {@inheritDoc}
   *
   * <p>Every delegate scans the whole text; the union of what they report is reduced to
   * the non-overlapping set this class describes.</p>
   */
  @Override
  public List<PiiMention> extract(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<Hit> hits = new ArrayList<>();
    for (int order = 0; order < extractors.size(); order++) {
      for (final PiiMention mention : extractors.get(order).extract(text)) {
        hits.add(new Hit(mention.span().getStart(), mention.span().getEnd(), order,
            PiiTypePriority.rank(mention.type()), mention));
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
