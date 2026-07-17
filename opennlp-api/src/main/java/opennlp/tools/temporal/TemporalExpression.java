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

package opennlp.tools.temporal;

import opennlp.tools.util.Span;

/**
 * One temporal mention in a text: the {@link Span} it covers in the original text, the
 * normalized calendar value, and its granularity.
 *
 * <p>The value follows the ISO 8601 style at the mention's granularity:
 * {@code 2026-07-14} for a day, {@code 2026-W29} for an ISO week, {@code 2026-07} for
 * a month, {@code 2024-Q3} for a quarter, and {@code 2026} for a year. Day-level
 * values are calendar-validated, so {@code February 30} is never reported.</p>
 *
 * @param span The location of the mention in the original text. Must not be {@code null}.
 * @param value The normalized calendar value. Must not be {@code null} or blank.
 * @param granularity The granularity of the value. Must not be {@code null}.
 *
 * @since 3.0.0
 */
public record TemporalExpression(Span span, String value, Granularity granularity) {

  /**
   * The granularities a mention can normalize to.
   */
  public enum Granularity {
    /** A calendar day, valued as {@code yyyy-MM-dd}. */
    DAY,
    /** An ISO 8601 week, valued as {@code yyyy-Www} in the week-based year. */
    WEEK,
    /** A calendar month, valued as {@code yyyy-MM}. */
    MONTH,
    /** A calendar quarter, valued as {@code yyyy-Qn}. */
    QUARTER,
    /** A calendar year, valued as {@code yyyy}. */
    YEAR
  }

  /**
   * Validates the mention.
   *
   * @throws IllegalArgumentException Thrown if {@code span} or {@code granularity} is
   *         {@code null}, or {@code value} is {@code null} or blank.
   */
  public TemporalExpression {
    if (span == null) {
      throw new IllegalArgumentException("span must not be null");
    }
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("value must not be null or blank");
    }
    if (granularity == null) {
      throw new IllegalArgumentException("granularity must not be null");
    }
  }
}
