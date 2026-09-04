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

package opennlp.tools.formats;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.function.Consumer;

/**
 * Reads an FSA5 finite-state automaton and enumerates the byte sequences it accepts.
 *
 * <p>FSA5 is the older of the two morfologik automaton formats (the newer being
 * {@link CFSA2Reader}); its arcs use a fixed-width goto address rather than a variable-length one.
 * This reader is written from the published format and adds no third-party dependency.
 * Interpreting the accepted sequences as morphological entries is left to the caller.</p>
 *
 * <p>Instances hold only immutable state, so {@link #forEachSequence(Consumer)} may be called
 * concurrently.</p>
 */
public final class FSA5Reader implements FsaSequenceReader {

  private static final int BIT_FINAL_ARC = 0x01;
  private static final int BIT_LAST_ARC = 0x02;
  private static final int BIT_TARGET_NEXT = 0x04;

  /** Offset of the flags/goto field within an arc; the label occupies the byte before it. */
  private static final int ADDRESS_OFFSET = 1;
  private static final int HEADER_SIZE = 8;
  private static final int TERMINAL_NODE = 0;
  private static final int NO_ARC = 0;

  /** Guards against runaway recursion on a malformed automaton. */
  private static final int MAX_SEQUENCE_LENGTH = 8192;

  private final byte[] arcs;
  private final int gotoLength;
  private final int nodeDataLength;
  private final int rootNode;

  /**
   * Initializes the reader over the automaton's arc block.
   *
   * @param arcs           The arc block, the automaton bytes after the header.
   * @param gotoLength     The width in bytes of an arc's flags and goto address field.
   * @param nodeDataLength The width in bytes of the optional data preceding a node's arcs.
   */
  private FSA5Reader(byte[] arcs, int gotoLength, int nodeDataLength) {
    this.arcs = arcs;
    this.gotoLength = gotoLength;
    this.nodeDataLength = nodeDataLength;
    // FSA5 keeps a dummy node ahead of the epsilon node: skip the dummy's arc to reach the
    // epsilon node, then the root is its single arc's destination.
    final int epsilonNode = skipArc(firstArc(TERMINAL_NODE));
    this.rootNode = destinationNode(firstArc(epsilonNode));
  }

  /**
   * Reads an FSA5 automaton from a stream.
   *
   * @param in The automaton bytes, referenced by an open {@link InputStream}. Must not be
   *           {@code null}.
   * @return A reader over the automaton.
   * @throws IllegalArgumentException Thrown if {@code in} is {@code null}.
   * @throws IOException Thrown on IO errors, or if the stream is not an FSA5 automaton.
   */
  public static FSA5Reader read(InputStream in) throws IOException {
    if (in == null) {
      throw new IllegalArgumentException("in must not be null");
    }
    return fromBytes(in.readAllBytes());
  }

  /**
   * Reads an FSA5 automaton from bytes already in memory.
   *
   * @param bytes The whole automaton, magic header included.
   * @return A reader over the automaton.
   * @throws IOException Thrown if {@code bytes} is not an FSA5 automaton, its header is
   *                     truncated, or it declares a goto address width of zero.
   */
  static FSA5Reader fromBytes(byte[] bytes) throws IOException {
    FsaSequenceReader.requireFsaHeader(bytes);
    if (bytes.length < HEADER_SIZE) {
      throw new IOException("truncated FSA5 header: fewer than " + HEADER_SIZE + " bytes");
    }
    if ((bytes[4] & 0xff) != VERSION_FSA5) {
      throw new IOException("unsupported FSA version 0x"
          + Integer.toHexString(bytes[4] & 0xff) + "; only FSA5 (0x05) is read here");
    }
    // One header byte packs the per-node data width in its high nibble and the goto address
    // width in its low nibble.
    final int widths = bytes[7] & 0xff;
    final int gotoLength = widths & 0x0f;
    final int nodeDataLength = (widths >>> 4) & 0x0f;
    if (gotoLength < 1) {
      throw new IOException("invalid FSA5 goto length: " + gotoLength);
    }
    final byte[] arcs = Arrays.copyOfRange(bytes, HEADER_SIZE, bytes.length);
    return new FSA5Reader(arcs, gotoLength, nodeDataLength);
  }

  /**
   * {@inheritDoc}
   *
   * <p>Sequences are produced in the automaton's stored, lexicographic order.</p>
   *
   * @throws IllegalStateException Thrown if a path exceeds {@value #MAX_SEQUENCE_LENGTH} bytes,
   *                               which indicates a malformed automaton.
   */
  @Override
  public void forEachSequence(Consumer<byte[]> action) {
    if (action == null) {
      throw new IllegalArgumentException("action must not be null");
    }
    enumerate(rootNode, new GrowableByteSequence(), action);
  }

  /**
   * Walks every arc reachable from {@code node} depth first, reporting each accepting path.
   *
   * @param node   The offset of the node to descend into.
   * @param path   The labels collected on the way down; pushed and popped in place.
   * @param action The action to run for each accepted sequence.
   * @throws IllegalStateException Thrown if the path grows past {@value #MAX_SEQUENCE_LENGTH}
   *                               bytes.
   */
  private void enumerate(int node, GrowableByteSequence path, Consumer<byte[]> action) {
    if (path.length() > MAX_SEQUENCE_LENGTH) {
      throw new IllegalStateException(
          "FSA5 sequence exceeds " + MAX_SEQUENCE_LENGTH + " bytes; automaton may be malformed");
    }
    for (int arc = firstArc(node); arc != NO_ARC; arc = nextArc(arc)) {
      path.push(arcs[arc]);
      if ((arcs[arc + ADDRESS_OFFSET] & BIT_FINAL_ARC) != 0) {
        action.accept(path.toByteArray());
      }
      final int destination = destinationNode(arc);
      if (destination != TERMINAL_NODE) {
        enumerate(destination, path, action);
      }
      path.pop();
    }
  }

  /**
   * @param node The offset of a node.
   * @return The offset of that node's first arc, skipping the per-node data the header declares.
   */
  private int firstArc(int node) {
    return nodeDataLength + node;
  }

  /**
   * @param arc The offset of an arc.
   * @return The offset of the following arc of the same node, or {@value #NO_ARC} if {@code arc}
   *         is the last one.
   */
  private int nextArc(int arc) {
    return (arcs[arc + ADDRESS_OFFSET] & BIT_LAST_ARC) != 0 ? NO_ARC : skipArc(arc);
  }

  /**
   * @param arc The offset of an arc.
   * @return The offset of the node the arc points at, which is either the node laid out directly
   *         after the arc or the goto address stored in the arc, whose low three bits carry the
   *         arc flags; {@value #TERMINAL_NODE} if the arc ends a word without continuing.
   */
  private int destinationNode(int arc) {
    if ((arcs[arc + ADDRESS_OFFSET] & BIT_TARGET_NEXT) != 0) {
      return skipArc(arc);
    }
    int value = 0;
    for (int i = gotoLength - 1; i >= 0; i--) {
      value = (value << 8) | (arcs[arc + ADDRESS_OFFSET + i] & 0xff);
    }
    return value >>> 3;
  }

  /**
   * @param arc The offset of an arc.
   * @return The offset just past that arc, that is, the start of whatever follows it.
   */
  private int skipArc(int arc) {
    return (arcs[arc + ADDRESS_OFFSET] & BIT_TARGET_NEXT) != 0
        ? arc + ADDRESS_OFFSET + 1
        : arc + ADDRESS_OFFSET + gotoLength;
  }
}
