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

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import opennlp.tools.document.Document;

/**
 * Replaces PII mentions with numbered labels such as {@code EMAIL-1} and {@code EMAIL-2}.
 *
 * <p>Mentions with the same type and {@link PiiMention#normalized() normalized form}
 * share a label within one rewrite. Types are case-sensitive. Labels uppercase ASCII
 * letters in the type, so custom types such as {@code id} and {@code ID} share a sequence
 * of numbers but receive distinct labels.</p>
 *
 * <p>Numbering restarts with every rewrite. Matching labels in separate documents do not
 * establish a shared identity. Use {@link HmacTokenizer} for cross-document tokens.</p>
 *
 * <p>Labels are rarely as long as the values they replace, so offsets move; see
 * {@link PiiRewrite} for mapping annotations onto the rewritten text.</p>
 *
 * <p>Instances are immutable and safe to share between threads: the counters that number
 * the labels live for the duration of one {@code rewrite} call.</p>
 *
 * @since 3.0.0
 */
public final class Pseudonymizer {

  /** One unambiguous map key for a mention identity. */
  private record Identity(String type, String normalized) {
  }

  private final String prefix;
  private final String suffix;

  /**
   * Initializes a pseudonymizer producing bare labels such as {@code EMAIL-1}.
   */
  public Pseudonymizer() {
    this("", "");
  }

  /**
   * Initializes a pseudonymizer with a prefix and suffix around each label.
   *
   * @param prefix Placed before each label. Must not be {@code null}; may be empty.
   * @param suffix Placed after each label. Must not be {@code null}; may be empty.
   * @throws IllegalArgumentException Thrown if {@code prefix} or {@code suffix} is
   *         {@code null}.
   */
  public Pseudonymizer(String prefix, String suffix) {
    if (prefix == null || suffix == null) {
      throw new IllegalArgumentException("prefix and suffix must not be null");
    }
    this.prefix = prefix;
    this.suffix = suffix;
  }

  /**
   * Rewrites a text, replacing each mention with its label.
   *
   * @param text The original text. Must not be {@code null}.
   * @param mentions The mentions to replace, as reported by a {@link PiiExtractor}. Must
   *                 not be {@code null} or contain {@code null}, every span must lie
   *                 within {@code text}, and no two spans may overlap.
   * @return The rewrite. Never {@code null}.
   * @throws IllegalArgumentException Thrown if an argument is {@code null}, a mention is
   *         {@code null}, a span lies outside the text, or two spans overlap.
   */
  public PiiRewrite rewrite(CharSequence text, List<PiiMention> mentions) {
    final Map<Identity, String> labels = new HashMap<>();
    final Map<String, Integer> counters = new HashMap<>();
    return PiiRewrite.replace(text, mentions,
        mention -> label(mention, labels, counters));
  }

  /**
   * Rewrites a document's text, replacing every mention of its {@link PiiAnnotator#PII}
   * layer.
   *
   * @param document The document to rewrite. Must be non-null and have a
   *                 {@link PiiAnnotator#PII} layer with matching annotation and mention
   *                 offsets.
   * @return The rewrite. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is null, lacks the PII
   *         layer, or contains a mention with offsets that differ from its annotation.
   */
  public PiiRewrite rewrite(Document document) {
    final List<PiiMention> mentions = PiiLayer.mentions(document);
    return rewrite(document.text(), mentions);
  }

  /**
   * Returns the stable per-text label for one normalized mention.
   *
   * @param mention The mention being replaced.
   * @param labels Labels already assigned by type and normalized value.
   * @param counters The sequence numbers by displayed type prefix.
   * @return The existing or newly assigned label.
   */
  private String label(PiiMention mention, Map<Identity, String> labels,
      Map<String, Integer> counters) {
    final Identity key = new Identity(mention.type(), mention.normalized());
    return labels.computeIfAbsent(key, ignored -> {
      final String typePrefix = Ascii.toUpper(mention.type());
      final int number = counters.merge(typePrefix, 1, Integer::sum);
      return prefix + typePrefix + '-' + number + suffix;
    });
  }
}
