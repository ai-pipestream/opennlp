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

package opennlp.tools.extraction;

import java.text.DecimalFormatSymbols;
import java.util.Locale;

/**
 * The written conventions for grouping digits and marking a decimal fraction that the
 * numeric extractors read.
 *
 * <p>The two conventions are mirror images of each other, which is why a number written
 * in one and read in the other silently means something else: {@code 1.234,56} is a
 * little over a thousand where a dot groups digits and just over one where a dot marks
 * the fraction. A caller therefore states which convention a document follows, either
 * directly or through {@link #forLocale(Locale)}, instead of the extractors guessing per
 * number.</p>
 *
 * <p>Text that cannot be read in the stated convention is rejected rather than
 * approximated: the scan of an ambiguous grouping such as {@code 1,23} in
 * {@link #LATIN_US} fails, so a document in the wrong convention yields no mentions
 * instead of wrong ones.</p>
 *
 * @see NumberScan
 * @since 3.0.0
 */
public enum NumberNotation {

  /**
   * Commas group digits and a dot marks the fraction, as in {@code 1,234.56}. The
   * convention of English-speaking regions and of most of Asia.
   */
  LATIN_US(',', '.'),

  /**
   * Dots group digits and a comma marks the fraction, as in {@code 1.234,56}. The
   * convention of most of continental Europe and of South America.
   */
  LATIN_EU('.', ',');

  private final char groupSeparator;
  private final char decimalSeparator;

  NumberNotation(char groupSeparator, char decimalSeparator) {
    this.groupSeparator = groupSeparator;
    this.decimalSeparator = decimalSeparator;
  }

  /**
   * @return The character that separates groups of three digits.
   */
  public char groupSeparator() {
    return groupSeparator;
  }

  /**
   * @return The character that separates the integer part from the fraction.
   */
  public char decimalSeparator() {
    return decimalSeparator;
  }

  /**
   * Resolves the notation a locale writes numbers in.
   *
   * <p>The decision is taken from the locale's own {@link DecimalFormatSymbols}, so it
   * follows the JDK's locale data rather than a hard-coded list of countries: a locale
   * whose decimal separator is a comma is read as {@link #LATIN_EU}, every other locale
   * as {@link #LATIN_US}. Locales that group digits with neither a comma nor a dot, with
   * a narrow no-break space for instance, are read in the notation their decimal
   * separator names, and their grouped numbers are then rejected rather than
   * misread.</p>
   *
   * @param locale The locale to resolve. Must not be {@code null}.
   * @return The notation of {@code locale}. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code locale} is {@code null}.
   */
  public static NumberNotation forLocale(Locale locale) {
    if (locale == null) {
      throw new IllegalArgumentException("locale must not be null");
    }
    return DecimalFormatSymbols.getInstance(locale).getDecimalSeparator()
        == LATIN_EU.decimalSeparator ? LATIN_EU : LATIN_US;
  }
}
