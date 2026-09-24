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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Exact Unicode Emoji 17.0 fully-qualified emoji sequence matcher. */
final class UnicodeEmojiSequences {

  private static final String RESOURCE = "EmojiSequences.txt";

  /** Prefix of a record holding one fully-qualified sequence. */
  private static final String SEQUENCE_RECORD = "S;";

  /** Prefix of a record holding one {@code Emoji_Component} code point range. */
  private static final String COMPONENT_RECORD = "C;";

  private static final char COMMENT = '#';

  private static final int VARIATION_SELECTOR_TEXT = 0xFE0E;
  private static final int VARIATION_SELECTOR_EMOJI = 0xFE0F;

  /** The bases of the keycap sequences: {@code #}, {@code *} and the ASCII digits. */
  private static final CodePointSet KEYCAP_BASES =
      CodePointSet.of('#', '*').union(CodePointSet.ofRange('0', '9'));

  // Volatile so the lazily loaded instance is safely published: the double-checked accessor
  // reads the field once, and a fully built trie becomes visible to every thread that observes
  // the non-null reference.
  private static volatile UnicodeEmojiSequences instance;

  private final Node root;
  private final int[][] componentRanges;

  /**
   * Every code point a candidate can start with: the first code point of a sequence or a
   * structural component. One bit test rejects nearly every position of ordinary text before
   * the trie is touched.
   */
  private final BitSet candidateStarts;

  private UnicodeEmojiSequences(Node root, int[][] componentRanges) {
    this.root = root;
    this.componentRanges = componentRanges;
    this.candidateStarts = new BitSet();
    for (int codePoint : root.children.keySet()) {
      candidateStarts.set(codePoint);
    }
    for (int[] range : componentRanges) {
      candidateStarts.set(range[0], range[1] + 1);
    }
    candidateStarts.set(VARIATION_SELECTOR_TEXT);
  }

  /**
   * {@return the matcher built from the bundled data, loaded on first use}
   *
   * @throws IllegalStateException Thrown if the bundled data resource is missing.
   * @throws UncheckedIOException Thrown if the bundled data resource cannot be read.
   * @throws IllegalArgumentException Thrown if the bundled data is malformed.
   */
  static UnicodeEmojiSequences getInstance() {
    UnicodeEmojiSequences sequences = instance;
    if (sequences == null) {
      synchronized (UnicodeEmojiSequences.class) {
        sequences = instance;
        if (sequences == null) {
          sequences = load();
          instance = sequences;
        }
      }
    }
    return sequences;
  }

  Candidate candidateAt(CharSequence text, int start) {
    int first = Character.codePointAt(text, start);
    if (!candidateStarts.get(first)) {
      return null;
    }
    Walk walk = walk(text, start);
    int end = walk.matchEnd();
    boolean valid = end >= 0;
    if (!valid) {
      int prefixEnd = walk.prefixEnd();
      if (!KEYCAP_BASES.contains(first) && prefixEnd > start && prefixEnd < text.length()
          && isStructuralComponent(Character.codePointAt(text, prefixEnd))) {
        end = prefixEnd;
      }
      else if (isStructuralComponent(first)) {
        end = start + Character.charCount(first);
      }
      else {
        return null;
      }
    }
    // The start of the most recent complete sequence of the run, and whether the item before
    // the current position is a stray component. A component connects only to the complete
    // sequence right before it and to the one right after it.
    int lastSequenceStart = start;
    boolean afterComponent = !valid;
    int position = end;
    while (position < text.length()) {
      int next = match(text, position);
      if (next >= 0) {
        if (!valid && !afterComponent) {
          break;
        }
        lastSequenceStart = position;
        position = next;
        afterComponent = false;
        continue;
      }
      int codePoint = Character.codePointAt(text, position);
      if (!isStructuralComponent(codePoint)) {
        break;
      }
      if (valid && lastSequenceStart > start) {
        // Only the last complete sequence is connected to the stray component: the run ends
        // before it, and the next call at that sequence yields the malformed candidate.
        return new Candidate(lastSequenceStart, true);
      }
      valid = false;
      afterComponent = true;
      position += Character.charCount(codePoint);
    }
    return new Candidate(position, valid);
  }

  private int match(CharSequence text, int start) {
    return walk(text, start).matchEnd();
  }

  private Walk walk(CharSequence text, int start) {
    Node node = root;
    int longest = -1;
    int i = start;
    while (i < text.length()) {
      int codePoint = Character.codePointAt(text, i);
      Node child = node.children.get(codePoint);
      if (child == null) {
        break;
      }
      node = child;
      i += Character.charCount(codePoint);
      if (node.terminal) {
        longest = i;
      }
    }
    if (longest >= 0 && longest < text.length()
        && Character.codePointAt(text, longest) == VARIATION_SELECTOR_EMOJI) {
      // One redundant emoji presentation selector after a complete sequence belongs to it.
      longest++;
    }
    return new Walk(longest, i);
  }

  private boolean isStructuralComponent(int codePoint) {
    if (KEYCAP_BASES.contains(codePoint)) {
      return false;
    }
    if (codePoint == VARIATION_SELECTOR_TEXT) {
      return true;
    }
    int low = 0;
    int high = componentRanges.length - 1;
    while (low <= high) {
      int middle = (low + high) >>> 1;
      int[] range = componentRanges[middle];
      if (codePoint < range[0]) {
        high = middle - 1;
      }
      else if (codePoint > range[1]) {
        low = middle + 1;
      }
      else {
        return true;
      }
    }
    return false;
  }

  private static UnicodeEmojiSequences load() {
    try (InputStream in = UnicodeEmojiSequences.class.getResourceAsStream(RESOURCE)) {
      if (in == null) {
        throw new IllegalStateException("Missing emoji sequence data resource: " + RESOURCE);
      }
      return parse(in);
    } catch (IOException e) {
      throw new UncheckedIOException("Unable to read emoji sequence data resource " + RESOURCE, e);
    }
  }

  /**
   * Parses {@code S;} sequence and {@code C;} component range records into a matcher.
   * Package-visible so the malformed-data handling can be exercised without the bundled
   * resource.
   *
   * @param in The records to read, one per line; {@code #} lines are comments.
   * @return The matcher.
   * @throws IOException Thrown if reading {@code in} fails.
   * @throws IllegalArgumentException Thrown if a record is malformed.
   */
  static UnicodeEmojiSequences parse(InputStream in) throws IOException {
    Node root = new Node();
    int sequences = 0;
    List<int[]> ranges = new ArrayList<>();
    try (BufferedReader reader = new BufferedReader(
        new InputStreamReader(in, StandardCharsets.US_ASCII))) {
      String line;
      int lineNumber = 0;
      while ((line = reader.readLine()) != null) {
        lineNumber++;
        if (line.isBlank() || line.charAt(0) == COMMENT) {
          continue;
        }
        try {
          if (line.startsWith(SEQUENCE_RECORD)) {
            addSequence(root, line.substring(SEQUENCE_RECORD.length()));
            sequences++;
          } else if (line.startsWith(COMPONENT_RECORD)) {
            ranges.add(HexCodePoints.parseRange(line.substring(COMPONENT_RECORD.length())));
          } else {
            throw new IllegalArgumentException("neither a comment nor a record");
          }
        } catch (IllegalArgumentException e) {
          // Fail loud naming the bad line, the same way the sibling loaders do.
          throw new IllegalArgumentException("Malformed emoji sequence data in " + RESOURCE
              + " at line " + lineNumber + ": " + line, e);
        }
      }
    }
    if (sequences == 0) {
      throw new IllegalArgumentException("No " + SEQUENCE_RECORD + " sequence record in "
          + RESOURCE);
    }
    if (ranges.isEmpty()) {
      throw new IllegalArgumentException("No " + COMPONENT_RECORD + " component range record in "
          + RESOURCE);
    }
    return new UnicodeEmojiSequences(root, ranges.toArray(int[][]::new));
  }

  private static void addSequence(Node root, String sequence) {
    Node node = root;
    int tokenStart = 0;
    for (int i = 0; i <= sequence.length(); i++) {
      if (i == sequence.length() || sequence.charAt(i) == HexCodePoints.SEQUENCE_SEPARATOR) {
        int codePoint = HexCodePoints.parseCodePoint(sequence, tokenStart, i);
        node = node.children.computeIfAbsent(codePoint, ignored -> new Node());
        tokenStart = i + 1;
      }
    }
    node.terminal = true;
  }

  record Candidate(int end, boolean valid) {
  }

  private record Walk(int matchEnd, int prefixEnd) {
  }

  private static final class Node {
    private final Map<Integer, Node> children = new HashMap<>();
    private boolean terminal;
  }
}
