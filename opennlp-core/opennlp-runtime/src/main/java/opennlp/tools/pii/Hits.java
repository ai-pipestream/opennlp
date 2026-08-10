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
import java.util.List;

import opennlp.tools.util.Span;

/**
 * The candidate collection and overlap resolution shared by the PII extractors: each
 * scanner reports every candidate it finds, and one pass reduces them to the
 * non-overlapping set that is reported.
 *
 * <p>The rule is leftmost, then longest, then the more specific type, so an extractor
 * that scans each type independently reports the same mentions whatever order its
 * scanners ran in.</p>
 */
final class Hits {

  /**
   * One candidate found by a scanner, held until overlap resolution decides which
   * candidates survive.
   *
   * @param start The candidate start offset in the scanned text, inclusive.
   * @param end The candidate end offset in the scanned text, exclusive.
   * @param priority The type priority that breaks exact-span ties; a lower value is the
   *                 more specific type.
   * @param mention The mention to report if this candidate survives.
   */
  record Hit(int start, int end, int priority, PiiMention mention) {
  }

  private Hits() {
    // This class holds static methods only and is never instantiated.
  }

  /**
   * Records one candidate.
   *
   * @param hits The candidate collector.
   * @param start The candidate start offset, inclusive.
   * @param end The candidate end offset, exclusive.
   * @param type The mention type.
   * @param normalized The normalized form of the mention.
   */
  static void add(List<Hit> hits, int start, int end, String type, String normalized) {
    hits.add(new Hit(start, end, PiiTypePriority.rank(type),
        new PiiMention(new Span(start, end), type, normalized)));
  }

  /**
   * Resolves overlapping candidates: leftmost first, then longest, then the more
   * specific type.
   *
   * @param hits The raw candidates; this list is sorted in place.
   * @return The surviving mentions in text order. Never {@code null}.
   */
  static List<PiiMention> resolve(List<Hit> hits) {
    hits.sort((a, b) -> {
      if (a.start() != b.start()) {
        return Integer.compare(a.start(), b.start());
      }
      if (a.end() != b.end()) {
        return Integer.compare(b.end(), a.end());
      }
      return Integer.compare(a.priority(), b.priority());
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
