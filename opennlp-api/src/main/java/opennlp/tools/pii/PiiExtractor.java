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

import java.util.List;

/**
 * The interface for extractors of personally identifiable information, which find PII
 * mentions in a text and report each as a {@link PiiMention} with its span in the
 * original text.
 *
 * @see PiiMention
 * @since 3.0.0
 */
public interface PiiExtractor {

  /**
   * Extracts all PII mentions from a text.
   *
   * @param text The text to scan. Must not be {@code null}.
   * @return The mentions in text order, non-overlapping. Never {@code null}; empty when
   *         the text contains no PII mention.
   * @throws IllegalArgumentException Thrown if {@code text} is {@code null}.
   */
  List<PiiMention> extract(CharSequence text);
}
