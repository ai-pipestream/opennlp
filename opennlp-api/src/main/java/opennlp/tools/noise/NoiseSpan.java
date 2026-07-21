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

import opennlp.tools.util.Span;
import opennlp.tools.util.StringUtil;

/**
 * One noisy stretch of a text: the {@link Span} it covers in the original text, how bad
 * it is, and a score within its severity.
 *
 * <p>The severity is an open string so scorers can introduce new tiers without an API
 * change; the constants on this record name the tiers the built-in scorer reports, from
 * mildest to worst. The score orders findings within one severity: higher means the
 * signals agreed more strongly. Scores of different severities are not comparable.</p>
 *
 * @param span The location of the noise in the original text. Must not be
 *             {@code null}.
 * @param severity The severity tier, for example {@link #SEVERITY_GIBBERISH}. Must not
 *                 be {@code null} or blank.
 * @param score The strength within the severity, in {@code (0, 1]}.
 *
 * @since 3.0.0
 */
public record NoiseSpan(Span span, String severity, double score) {

  /**
   * A recognizable word damaged by a known confusion, the shape optical recognition
   * errors take; reported only when a dictionary is available to recognize the repair.
   */
  public static final String SEVERITY_MISSPELLED = "misspelled";

  /** Partially readable text with depressed structure. */
  public static final String SEVERITY_DAMAGED = "damaged";

  /** Text without linguistic structure. */
  public static final String SEVERITY_GIBBERISH = "gibberish";

  /** A run shaped like an encoded binary rather than language. */
  public static final String SEVERITY_BINARYISH = "binaryish";

  /**
   * Validates the noise span.
   *
   * @throws IllegalArgumentException Thrown if {@code span} is {@code null},
   *         {@code severity} is {@code null} or blank, or {@code score} is outside
   *         {@code (0, 1]}.
   */
  public NoiseSpan {
    if (span == null) {
      throw new IllegalArgumentException("span must not be null");
    }
    if (severity == null || StringUtil.isBlank(severity)) {
      throw new IllegalArgumentException("severity must not be null or blank");
    }
    if (!(score > 0.0 && score <= 1.0)) {
      throw new IllegalArgumentException("score must be in (0, 1], got: " + score);
    }
  }
}
