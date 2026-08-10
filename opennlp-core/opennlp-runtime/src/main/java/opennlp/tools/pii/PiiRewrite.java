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

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;

import opennlp.tools.document.Annotation;
import opennlp.tools.util.Span;

/**
 * The result of replacing PII mentions with labels: the rewritten text, where each label
 * now is, and a mapping from offsets in the original text to offsets in the rewritten one.
 *
 * <p>Unlike {@link Masker}, which is length preserving and therefore leaves every other
 * layer's spans valid, a label is rarely as long as the value it replaces. Offsets shift,
 * so annotations from the original text have to be moved before they can be used with the
 * rewritten text. {@link #mapOffset(int)}, {@link #mapSpan(Span)}, and
 * {@link #remap(List)} do that moving.</p>
 *
 * <p>Produced by {@link Pseudonymizer} and {@link HmacTokenizer}. Instances are immutable
 * and safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public final class PiiRewrite {

  private final String text;
  private final List<PiiMention> mentions;
  private final int[] originalStarts;
  private final int[] originalEnds;
  private final int[] rewrittenStarts;
  private final int[] rewrittenEnds;
  private final int originalLength;

  private PiiRewrite(String text, List<PiiMention> mentions, int[] originalStarts,
      int[] originalEnds, int[] rewrittenStarts, int[] rewrittenEnds, int originalLength) {
    this.text = text;
    this.mentions = mentions;
    this.originalStarts = originalStarts;
    this.originalEnds = originalEnds;
    this.rewrittenStarts = rewrittenStarts;
    this.rewrittenEnds = rewrittenEnds;
    this.originalLength = originalLength;
  }

  /**
   * Replaces every mention with the label the labeler assigns to it.
   *
   * @param text The original text. Must not be {@code null}.
   * @param mentions The mentions to replace. Must not be {@code null} or contain
   *                 {@code null}, every span must lie within {@code text}, and no two
   *                 spans may overlap. Order does not matter.
   * @param labeler Assigns the replacement for a mention. Must not be {@code null} and
   *                must not return {@code null} or an empty label.
   * @return The rewrite. Never {@code null}.
   * @throws IllegalArgumentException Thrown if an argument is {@code null}, a mention is
   *         {@code null}, a span lies outside the text, two spans overlap, or the labeler
   *         returns {@code null} or an empty label.
   */
  static PiiRewrite replace(CharSequence text, List<PiiMention> mentions,
      Function<PiiMention, String> labeler) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    if (mentions == null) {
      throw new IllegalArgumentException("mentions must not be null");
    }
    if (labeler == null) {
      throw new IllegalArgumentException("labeler must not be null");
    }
    final List<PiiMention> ordered = ordered(text, mentions);
    final int count = ordered.size();
    final int[] originalStarts = new int[count];
    final int[] originalEnds = new int[count];
    final int[] rewrittenStarts = new int[count];
    final int[] rewrittenEnds = new int[count];
    final List<PiiMention> labelled = new ArrayList<>(count);
    final StringBuilder rewritten = new StringBuilder(text.length());
    int copied = 0;
    for (int i = 0; i < count; i++) {
      final PiiMention mention = ordered.get(i);
      final int start = mention.span().getStart();
      final int end = mention.span().getEnd();
      final String label = labeler.apply(mention);
      if (label == null || label.isEmpty()) {
        throw new IllegalArgumentException("labeler must not return a null or empty label");
      }
      rewritten.append(text, copied, start);
      originalStarts[i] = start;
      originalEnds[i] = end;
      rewrittenStarts[i] = rewritten.length();
      rewritten.append(label);
      rewrittenEnds[i] = rewritten.length();
      labelled.add(new PiiMention(new Span(rewrittenStarts[i], rewrittenEnds[i]),
          mention.type(), label));
      copied = end;
    }
    rewritten.append(text, copied, text.length());
    return new PiiRewrite(rewritten.toString(), Collections.unmodifiableList(labelled),
        originalStarts, originalEnds, rewrittenStarts, rewrittenEnds, text.length());
  }

  private static List<PiiMention> ordered(CharSequence text, List<PiiMention> mentions) {
    final List<PiiMention> ordered = new ArrayList<>(mentions.size());
    for (final PiiMention mention : mentions) {
      if (mention == null) {
        throw new IllegalArgumentException("mentions must not contain null");
      }
      if (mention.span().getStart() < 0 || mention.span().getEnd() > text.length()) {
        throw new IllegalArgumentException("span lies outside the text: " + mention.span());
      }
      ordered.add(mention);
    }
    ordered.sort(Comparator.comparingInt(mention -> mention.span().getStart()));
    for (int i = 1; i < ordered.size(); i++) {
      if (ordered.get(i).span().getStart() < ordered.get(i - 1).span().getEnd()) {
        throw new IllegalArgumentException("mentions must not overlap: "
            + ordered.get(i - 1).span() + " and " + ordered.get(i).span());
      }
    }
    return ordered;
  }

  /**
   * Returns the rewritten text.
   *
   * @return The text with every mention replaced by its label. Never {@code null}.
   */
  public String text() {
    return text;
  }

  /**
   * Returns where the labels are in the rewritten text, in text order. The
   * {@link PiiMention#normalized() normalized form} of each is the label itself, so the
   * list can be annotated onto the rewritten text as a {@link PiiAnnotator#PII} layer
   * without revealing anything that was replaced.
   *
   * @return The labels as mentions of the rewritten text. Never {@code null}; immutable.
   */
  public List<PiiMention> mentions() {
    return mentions;
  }

  /**
   * Maps an offset in the original text to the matching offset in the rewritten text.
   *
   * <p>An offset in unreplaced text maps exactly. An offset inside a replaced value maps
   * to the start of its label, since no finer answer exists: the characters it pointed at
   * are gone.</p>
   *
   * @param offset The offset in the original text. Must be between {@code 0} and the
   *               length of the original text.
   * @return The offset in {@link #text()}.
   * @throws IndexOutOfBoundsException Thrown if {@code offset} is outside the original
   *         text.
   */
  public int mapOffset(int offset) {
    return map(offset, true);
  }

  /**
   * Maps a span of the original text to the matching span of the rewritten text.
   *
   * <p>A span that contains a replaced value grows or shrinks with the label. A span that
   * is contained in a replaced value collapses onto the whole label, so the returned span
   * covers at least the label. Sentence and token spans, which never straddle a value
   * partially, map exactly.</p>
   *
   * @param span The span in the original text. Must not be {@code null} and must lie
   *             within the original text.
   * @return The matching span of {@link #text()}. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code span} is {@code null}.
   * @throws IndexOutOfBoundsException Thrown if {@code span} lies outside the original
   *         text.
   */
  public Span mapSpan(Span span) {
    if (span == null) {
      throw new IllegalArgumentException("span must not be null");
    }
    final int start = map(span.getStart(), true);
    final int end = Math.max(start, map(span.getEnd(), false));
    return new Span(start, end, span.getType(), span.getProb());
  }

  /**
   * Maps the spans of a layer's annotations onto the rewritten text, keeping each
   * annotation's value.
   *
   * <p>An annotation whose span collapses to nothing, one that covered only part of a
   * replaced value and nothing else, is left out: it no longer describes anything in the
   * rewritten text.</p>
   *
   * @param annotations The annotations of the original text. Must not be {@code null} or
   *                    contain {@code null}.
   * @param <T> The annotation value type.
   * @return The annotations with mapped spans, in the given order. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code annotations} is {@code null} or
   *         contains {@code null}.
   * @throws IndexOutOfBoundsException Thrown if an annotation lies outside the original
   *         text.
   */
  public <T> List<Annotation<T>> remap(List<Annotation<T>> annotations) {
    if (annotations == null) {
      throw new IllegalArgumentException("annotations must not be null");
    }
    final List<Annotation<T>> mapped = new ArrayList<>(annotations.size());
    for (final Annotation<T> annotation : annotations) {
      if (annotation == null) {
        throw new IllegalArgumentException("annotations must not contain null");
      }
      final Span span = mapSpan(annotation.span());
      if (span.length() > 0) {
        mapped.add(new Annotation<>(span, annotation.value()));
      }
    }
    return mapped;
  }

  private int map(int offset, boolean towardsStart) {
    if (offset < 0 || offset > originalLength) {
      throw new IndexOutOfBoundsException("offset out of range: " + offset);
    }
    int shift = 0;
    for (int i = 0; i < originalStarts.length; i++) {
      if (offset <= originalStarts[i]) {
        break;
      }
      if (offset < originalEnds[i]) {
        return towardsStart ? rewrittenStarts[i] : rewrittenEnds[i];
      }
      shift += (rewrittenEnds[i] - rewrittenStarts[i]) - (originalEnds[i] - originalStarts[i]);
    }
    return offset + shift;
  }
}
