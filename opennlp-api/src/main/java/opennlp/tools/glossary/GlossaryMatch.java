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

import opennlp.tools.util.Span;
import opennlp.tools.util.StringUtil;

/**
 * One glossary hit in a text: the {@link Span} it covers in the original text, the
 * identifier of the matched entry, and the glossary term that matched.
 *
 * <p>The term is the surface form as registered in the glossary; the text covered by the
 * span may differ from it in case when the matcher ignores case.</p>
 *
 * @param span The location of the hit in the original text. Must not be {@code null}.
 * @param id The identifier of the matched {@link GlossaryEntry}. Must not be
 *           {@code null} or blank.
 * @param term The registered term that matched. Must not be {@code null} or blank.
 *
 * @since 3.0.0
 */
public record GlossaryMatch(Span span, String id, String term) {

  /**
   * Validates the hit.
   *
   * @throws IllegalArgumentException Thrown if {@code span} is {@code null}, or
   *         {@code id} or {@code term} is {@code null} or blank.
   */
  public GlossaryMatch {
    if (span == null) {
      throw new IllegalArgumentException("span must not be null");
    }
    if (id == null || blank(id)) {
      throw new IllegalArgumentException("id must not be null or blank");
    }
    if (term == null || blank(term)) {
      throw new IllegalArgumentException("term must not be null or blank");
    }
  }

  /**
   * Reports whether a value is blank under the project whitespace definition, which
   * unlike the JDK's includes no-break spaces, so a value spelled entirely from them
   * cannot pass as content.
   */
  private static boolean blank(String value) {
    for (int i = 0; i < value.length(); ) {
      final int cp = value.codePointAt(i);
      if (!StringUtil.isWhitespace(cp)) {
        return false;
      }
      i += Character.charCount(cp);
    }
    return true;
  }
}
