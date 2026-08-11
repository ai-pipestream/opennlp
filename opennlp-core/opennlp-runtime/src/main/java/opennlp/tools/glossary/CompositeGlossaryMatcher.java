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
import java.util.Comparator;
import java.util.List;

/**
 * A {@link GlossaryMatcher} that merges the hits of several matchers into one
 * non-overlapping result, so a pipeline can combine, for example, the exact
 * {@link AhoCorasickGlossaryMatcher} with the inflected
 * {@link TermAnalyzingGlossaryMatcher} behind a single
 * {@link GlossaryAnnotator}.
 *
 * <p>Matchers are consulted in the order given, and earlier matchers win: a hit
 * is accepted only when its span intersects no hit already accepted from an
 * earlier matcher (or an earlier hit of the same matcher). Register the exact
 * matcher first and the inflected matcher second, and an inflected hit survives
 * only where no exact hit covers that stretch of text. Spans that merely touch,
 * where one ends exactly where the other starts, do not intersect and both
 * survive. Accepted hits are reported in text order.</p>
 *
 * <p>The composite holds no per-call state; it is safe to share across threads
 * when every delegate is.</p>
 *
 * @since 3.0.0
 */
public final class CompositeGlossaryMatcher implements GlossaryMatcher {

  /** The delegates in priority order; earlier matchers win on overlap. */
  private final List<GlossaryMatcher> matchers;

  /**
   * Builds a composite over delegate matchers.
   *
   * @param matchers The delegates in priority order, highest first. Must not be
   *                 {@code null} or empty, and no element may be {@code null}.
   *                 The list is copied.
   * @throws IllegalArgumentException Thrown if {@code matchers} is {@code null},
   *         empty, or contains {@code null}.
   */
  public CompositeGlossaryMatcher(List<GlossaryMatcher> matchers) {
    if (matchers == null || matchers.isEmpty()) {
      throw new IllegalArgumentException("matchers must not be null or empty");
    }
    for (final GlossaryMatcher matcher : matchers) {
      if (matcher == null) {
        throw new IllegalArgumentException("matchers must not contain null elements");
      }
    }
    this.matchers = List.copyOf(matchers);
  }

  /**
   * {@inheritDoc}
   *
   * <p>Each delegate runs in priority order and a hit is accepted only when its
   * span intersects no already-accepted hit. The merged hits are returned sorted
   * by start offset.</p>
   */
  @Override
  public List<GlossaryMatch> match(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    final List<GlossaryMatch> accepted = new ArrayList<>();
    for (final GlossaryMatcher matcher : matchers) {
      for (final GlossaryMatch hit : matcher.match(text)) {
        if (intersectsNone(accepted, hit)) {
          accepted.add(hit);
        }
      }
    }
    accepted.sort(Comparator.comparingInt(hit -> hit.span().getStart()));
    return accepted;
  }

  private static boolean intersectsNone(List<GlossaryMatch> accepted, GlossaryMatch hit) {
    for (final GlossaryMatch kept : accepted) {
      if (kept.span().intersects(hit.span())) {
        return false;
      }
    }
    return true;
  }
}
