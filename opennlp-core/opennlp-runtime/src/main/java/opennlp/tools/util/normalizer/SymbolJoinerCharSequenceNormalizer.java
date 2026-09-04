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
package opennlp.tools.util.normalizer;

import java.util.Map;

/**
 * A {@link CharSequenceNormalizer} that spells out a symbol-joiner token as its
 * word: a text consisting of exactly an ampersand ({@code "&"}) normalizes to
 * {@code "and"}. A document writing "Dungeons &amp; Dragons" and a query writing
 * "dungeons and dragons" then agree on term identity once terms are folded and
 * stemmed, because the ampersand token and the word "and" collapse onto the same
 * term.
 *
 * <p>The table covers the symbols that appear as standalone tokens in running
 * prose and have an unambiguous spelled-out English form: the joiners
 * ({@code &}, {@code +}, {@code @}), the unit and reference marks of legal and
 * technical writing ({@code %} percent, {@code §} section, {@code ¶} paragraph,
 * {@code °} degree), and the IP marks ({@code ©} copyright, {@code ®}
 * registered, {@code ™} trademark). Currency symbols are deliberately absent:
 * their words differ by locale and they almost never appear as standalone
 * tokens.</p>
 *
 * <p>Only a text consisting of exactly the symbol is rewritten; a symbol
 * embedded in a larger token ("R&amp;D", "AT&amp;T", "TSR®") is left unchanged,
 * because expanding it inside the token would invent a word that appears in
 * neither the document nor the query. Comparison is by whole-string equality,
 * so it is exact under UTF-16 and needs no pattern machinery.</p>
 *
 * <p>Texts that are not a known symbol are returned without copying, like the
 * sibling normalizers.</p>
 *
 * @since 3.0.0
 */
public class SymbolJoinerCharSequenceNormalizer implements CharSequenceNormalizer {

  private static final long serialVersionUID = -6772513786580257420L;

  private static final Map<String, String> WORD_BY_SYMBOL = Map.of(
      "&", "and",
      "+", "plus",
      "@", "at",
      "%", "percent",
      "§", "section",
      "¶", "paragraph",
      "°", "degree",
      "©", "copyright",
      "®", "registered",
      "™", "trademark");

  private static final SymbolJoinerCharSequenceNormalizer INSTANCE =
      new SymbolJoinerCharSequenceNormalizer();

  /** {@return the shared, stateless instance} */
  public static SymbolJoinerCharSequenceNormalizer getInstance() {
    return INSTANCE;
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public CharSequence normalize(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("The text must not be null.");
    }
    final String word = WORD_BY_SYMBOL.get(text.toString());
    return word != null ? word : text;
  }
}
