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
 * Replaces PII mentions with numbered labels, {@code EMAIL-1} and {@code EMAIL-2}, so that
 * a text stays readable and internally consistent while revealing nothing.
 *
 * <p>Within one rewrite the same value always gets the same label: a support ticket that
 * mentions one address in the greeting and again in the signature keeps the connection
 * between the two, which masking to {@code *****} destroys. The numbering restarts with
 * every rewrite and carries no meaning beyond one text, so the same address in two
 * documents will usually get different labels. Use {@link HmacTokenizer} where labels have
 * to agree across documents.</p>
 *
 * <p>Two mentions count as the same value when they have the same type and the same
 * {@link PiiMention#normalized() normalized form}, so formatting differences do not split
 * a label: {@code 4111 1111 1111 1111} and {@code 4111-1111-1111-1111} share one.</p>
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
   * Initializes a pseudonymizer that surrounds each label, for instance with brackets to
   * set the labels apart from the surrounding prose.
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
   * @param document The document to rewrite. Must not be {@code null} and must carry the
   *                 {@link PiiAnnotator#PII} layer.
   * @return The rewrite. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code document} is {@code null} or does
   *         not carry the PII layer.
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
   * @param counters The next sequence number by type.
   * @return The existing or newly assigned label.
   */
  private String label(PiiMention mention, Map<Identity, String> labels,
      Map<String, Integer> counters) {
    final Identity key = new Identity(mention.type(), mention.normalized());
    return labels.computeIfAbsent(key, ignored -> {
      final int number = counters.merge(mention.type(), 1, Integer::sum);
      return prefix + Ascii.toUpper(mention.type()) + '-' + number + suffix;
    });
  }
}
