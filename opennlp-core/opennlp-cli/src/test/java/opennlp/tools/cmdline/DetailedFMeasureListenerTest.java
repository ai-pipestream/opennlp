/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package opennlp.tools.cmdline;

import java.util.Locale;

import org.junit.jupiter.api.Test;

import opennlp.tools.util.Span;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DetailedFMeasureListenerTest {

  private final DetailedFMeasureListener<Span[]> listener =
      new DetailedFMeasureListener<>() {
        @Override
        protected Span[] asSpanArray(Span[] sample) {
          return sample;
        }
      };

  @Test
  void createsEmptyReport() {
    assertEquals("""
        Evaluated 0 samples with 0 entities; found: 0 entities; correct: 0.
               TOTAL: precision:    0.00%;  recall:    0.00%; F1:    0.00%.
        """, listener.createReport());
  }

  @Test
  void usesRequestedLocaleAndPreservesLongSupplementaryLabel() {
    String label = "LONG-\uD801\uDC00-LABEL";
    listener.correctlyClassified(new Span[] {new Span(0, 1, label)}, new Span[0]);

    String expected = "Evaluated 1 samples with 1 entities; found: 1 entities; correct: 1.\n"
        + "       TOTAL: precision:  100,00%;  recall:  100,00%; F1:  100,00%.\n"
        + "LONG-\uD801\uDC00-LABEL: precision:  100,00%;  recall:  100,00%; "
        + "F1:  100,00%. [target:   1; tp:   1; fp:   0]\n";
    assertEquals(expected, listener.createReport(Locale.GERMANY));
  }

  @Test
  void localizesCountsWithRequestedLocale() {
    listener.correctlyClassified(new Span[] {new Span(0, 1, "X")}, new Span[0]);

    String report = listener.createReport(Locale.forLanguageTag("ar-EG"));

    org.junit.jupiter.api.Assertions.assertTrue(
        report.contains("[target:   ١; tp:   ١; fp:   ٠]"), report);
  }

  @Test
  void padsSupplementaryLabelByCodePoint() {
    listener.correctlyClassified(new Span[] {new Span(0, 1, "\uD801\uDC00")}, new Span[0]);

    String typeLine = listener.createReport().lines().skip(2).findFirst().orElseThrow();

    org.junit.jupiter.api.Assertions.assertTrue(typeLine.startsWith("           \uD801\uDC00:"), typeLine);
  }

  @Test
  void rejectsNullLocale() {
    assertThrows(IllegalArgumentException.class, () -> listener.createReport(null));
  }
}
