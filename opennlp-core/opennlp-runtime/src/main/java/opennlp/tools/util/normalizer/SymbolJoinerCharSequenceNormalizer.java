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
 * {@code "and"}. This is what lets a document writing "Dungeons &amp; Dragons"
 * and a query writing "dungeons and dragons" agree on term identity once terms
 * are folded and stemmed — the ampersand token and the word "and" must collapse
 * onto the same term before that can happen.
 *
 * <p>Only a text consisting of exactly the symbol is rewritten; an ampersand
 * embedded in a larger token ("R&amp;D", "AT&amp;T") is left unchanged, because
 * expanding it inside the token would invent a word that appears in neither the
 * document nor the query. The table is deliberately one entry deep: add a
 * symbol only when a corpus shows the mismatch.</p>
 *
 * <p>Texts that are not a known symbol are returned without copying, like the
 * sibling normalizers.</p>
 *
 * @since 3.0.0
 */
public class SymbolJoinerCharSequenceNormalizer implements CharSequenceNormalizer {

  private static final long serialVersionUID = -3081219175592047788L;

  private static final Map<String, String> WORD_BY_SYMBOL = Map.of("&", "and");

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
    return WORD_BY_SYMBOL.getOrDefault(text.toString(), text.toString());
  }
}
