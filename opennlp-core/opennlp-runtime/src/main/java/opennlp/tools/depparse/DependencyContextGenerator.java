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

package opennlp.tools.depparse;

import opennlp.tools.commons.ThreadSafe;
import opennlp.tools.util.ParamChecks;

/**
 * Generates the classification features for one arc-standard configuration: words and
 * tags of the topmost stack and frontmost buffer positions, their pairings, the partial
 * structure built so far (tags and relations of the leftmost and rightmost dependents,
 * valency counts), and a bucketed distance between stack top and buffer front.
 *
 * <p>Words, tags, and relations are escaped with a backslash so that distinct
 * configurations yield distinct features: a value equal to a marker value gets a
 * backslash in front, a backslash in a value is doubled, and inside a feature combining
 * several values each {@code /} and {@code |} gets a backslash in front. A value with
 * no backslash and no separator that is not a marker appears unchanged.</p>
 *
 * <p>Instances hold no state and are safe to share between threads.</p>
 *
 * @since 3.0.0
 */
@ThreadSafe
class DependencyContextGenerator {

  /** The feature value standing for the artificial root position. */
  private static final String ROOT_VALUE = "*ROOT*";

  /** The feature value standing for a position that does not exist. */
  private static final String NONE_VALUE = "*NULL*";

  /** Separates a word from the tag of the same position within one feature. */
  private static final char WORD_TAG_SEPARATOR = '/';

  /** Separates the parts of a feature combining several positions. */
  private static final char POSITION_SEPARATOR = '|';

  /** Marks the following character of a value as part of the value. */
  private static final char ESCAPE = '\\';

  /** The number of features {@link #getContext(ArcStandardState, String[], String[])} emits. */
  private static final int FEATURE_COUNT = 37;

  /** Feature prefix: the word on top of the stack. */
  private static final String S0_WORD = "s0w=";

  /** Feature prefix: the tag on top of the stack. */
  private static final String S0_TAG = "s0t=";

  /** Feature prefix: the second stack word. */
  private static final String S1_WORD = "s1w=";

  /** Feature prefix: the second stack tag. */
  private static final String S1_TAG = "s1t=";

  /** Feature prefix: the third stack tag. */
  private static final String S2_TAG = "s2t=";

  /** Feature prefix: the first buffer word. */
  private static final String B0_WORD = "b0w=";

  /** Feature prefix: the first buffer tag. */
  private static final String B0_TAG = "b0t=";

  /** Feature prefix: the second buffer word. */
  private static final String B1_WORD = "b1w=";

  /** Feature prefix: the second buffer tag. */
  private static final String B1_TAG = "b1t=";

  /** Feature prefix: the third buffer tag. */
  private static final String B2_TAG = "b2t=";

  /** Feature prefix: word and tag on top of the stack. */
  private static final String S0_WORD_TAG = "s0wt=";

  /** Feature prefix: word and tag of the second stack position. */
  private static final String S1_WORD_TAG = "s1wt=";

  /** Feature prefix: word and tag of the first buffer position. */
  private static final String B0_WORD_TAG = "b0wt=";

  /** Feature prefix: stack top word with first buffer word. */
  private static final String S0_WORD_B0_WORD = "s0w,b0w=";

  /** Feature prefix: stack top tag with first buffer tag. */
  private static final String S0_TAG_B0_TAG = "s0t,b0t=";

  /** Feature prefix: stack top word with first buffer tag. */
  private static final String S0_WORD_B0_TAG = "s0w,b0t=";

  /** Feature prefix: stack top tag with first buffer word. */
  private static final String S0_TAG_B0_WORD = "s0t,b0w=";

  /** Feature prefix: stack top word and tag with first buffer tag. */
  private static final String S0_WORD_TAG_B0_TAG = "s0wt,b0t=";

  /** Feature prefix: second stack tag with stack top tag. */
  private static final String S1_TAG_S0_TAG = "s1t,s0t=";

  /** Feature prefix: second stack tag with stack top word. */
  private static final String S1_TAG_S0_WORD = "s1t,s0w=";

  /** Feature prefix: second stack word with stack top tag. */
  private static final String S1_WORD_S0_TAG = "s1w,s0t=";

  /** Feature prefix: second stack, stack top and first buffer tags. */
  private static final String S1_TAG_S0_TAG_B0_TAG = "s1t,s0t,b0t=";

  /** Feature prefix: stack top, first and second buffer tags. */
  private static final String S0_TAG_B0_TAG_B1_TAG = "s0t,b0t,b1t=";

  /** Feature prefix: the three topmost stack tags. */
  private static final String S2_TAG_S1_TAG_S0_TAG = "s2t,s1t,s0t=";

  /** Feature prefix: tag of the leftmost dependent of the stack top. */
  private static final String S0_LEFT_DEPENDENT_TAG = "s0lct=";

  /** Feature prefix: tag of the rightmost dependent of the stack top. */
  private static final String S0_RIGHT_DEPENDENT_TAG = "s0rct=";

  /** Feature prefix: tag of the leftmost dependent of the second stack position. */
  private static final String S1_LEFT_DEPENDENT_TAG = "s1lct=";

  /** Feature prefix: tag of the rightmost dependent of the second stack position. */
  private static final String S1_RIGHT_DEPENDENT_TAG = "s1rct=";

  /** Feature prefix: relation of the leftmost dependent of the stack top. */
  private static final String S0_LEFT_DEPENDENT_RELATION = "s0lcl=";

  /** Feature prefix: relation of the rightmost dependent of the stack top. */
  private static final String S0_RIGHT_DEPENDENT_RELATION = "s0rcl=";

  /** Feature prefix: relation of the rightmost dependent of the second stack position. */
  private static final String S1_RIGHT_DEPENDENT_RELATION = "s1rcl=";

  /** Feature prefix: second stack tag, its rightmost dependent tag, and stack top tag. */
  private static final String S1_TAG_S1_RIGHT_DEPENDENT_TAG_S0_TAG = "s1t,s1rct,s0t=";

  /** Feature prefix: stack top tag, its leftmost dependent tag, and first buffer tag. */
  private static final String S0_TAG_S0_LEFT_DEPENDENT_TAG_B0_TAG = "s0t,s0lct,b0t=";

  /** Feature prefix: dependent count of the stack top. */
  private static final String S0_DEPENDENTS = "s0deps=";

  /** Feature prefix: dependent count of the second stack position. */
  private static final String S1_DEPENDENTS = "s1deps=";

  /** Feature prefix: distance between stack top and first buffer position. */
  private static final String DISTANCE = "dist=";

  /** Feature prefix: distance with stack top and first buffer tags. */
  private static final String DISTANCE_S0_TAG_B0_TAG = "dist,s0t,b0t=";

  /** Valency counts at or above this bound share one feature value. */
  private static final int MAX_VALENCY = 3;

  /** Distances at or above this bound share the {@link #LONG_DISTANCE} feature value. */
  private static final int MAX_DISTANCE = 4;

  /** The feature value standing for every distance of {@link #MAX_DISTANCE} or more. */
  private static final String LONG_DISTANCE = "4+";

  /**
   * Generates the features of the current configuration.
   *
   * @param state The configuration to describe. Must not be {@code null}.
   * @param tokens The input tokens. Must satisfy the token contract of
   *               {@link DependencyParser#parse(String[], String[])} and match the state.
   * @param tags The part-of-speech tags aligned with {@code tokens}. Must not be
   *             {@code null} and must match the state.
   * @return The feature strings. Never {@code null}.
   * @throws IllegalArgumentException Thrown if a parameter is invalid or the arrays do
   *         not match the state.
   */
  String[] getContext(ArcStandardState state, String[] tokens, String[] tags) {
    ParamChecks.requireNonNullArg(state, "state");
    // Sentence contents are validated once at the parser or sample boundary.
    if (tokens == null || tags == null || tokens.length != state.tokenCount()
        || tags.length != state.tokenCount()) {
      throw new IllegalArgumentException("tokens and tags must match the state token count");
    }
    final int s0 = state.stack(0);
    final int s1 = state.stack(1);
    final int s2 = state.stack(2);
    final int b0 = state.buffer(0);
    final int b1 = state.buffer(1);
    final int b2 = state.buffer(2);

    final String s0w = valueAt(tokens, s0);
    final String s0t = valueAt(tags, s0);
    final String s1w = valueAt(tokens, s1);
    final String s1t = valueAt(tags, s1);
    final String s2t = valueAt(tags, s2);
    final String b0w = valueAt(tokens, b0);
    final String b0t = valueAt(tags, b0);
    final String b1w = valueAt(tokens, b1);
    final String b1t = valueAt(tags, b1);
    final String b2t = valueAt(tags, b2);

    final String s0lct = dependentTag(state, tags, s0, true);
    final String s0rct = dependentTag(state, tags, s0, false);
    final String s1lct = dependentTag(state, tags, s1, true);
    final String s1rct = dependentTag(state, tags, s1, false);
    final String s0lcl = dependentRelation(state, s0, true);
    final String s0rcl = dependentRelation(state, s0, false);
    final String s1rcl = dependentRelation(state, s1, false);

    final String distance = distance(s0, b0);
    final String s0wPart = part(s0w);
    final String s0tPart = part(s0t);
    final String s1wPart = part(s1w);
    final String s1tPart = part(s1t);
    final String s2tPart = part(s2t);
    final String b0wPart = part(b0w);
    final String b0tPart = part(b0t);
    final String b1tPart = part(b1t);
    final String s0lctPart = part(s0lct);
    final String s1rctPart = part(s1rct);
    final String[] features = new String[FEATURE_COUNT];
    int f = 0;
    features[f++] = S0_WORD + s0w;
    features[f++] = S0_TAG + s0t;
    features[f++] = S1_WORD + s1w;
    features[f++] = S1_TAG + s1t;
    features[f++] = S2_TAG + s2t;
    features[f++] = B0_WORD + b0w;
    features[f++] = B0_TAG + b0t;
    features[f++] = B1_WORD + b1w;
    features[f++] = B1_TAG + b1t;
    features[f++] = B2_TAG + b2t;
    features[f++] = S0_WORD_TAG + s0wPart + WORD_TAG_SEPARATOR + s0tPart;
    features[f++] = S1_WORD_TAG + s1wPart + WORD_TAG_SEPARATOR + s1tPart;
    features[f++] = B0_WORD_TAG + b0wPart + WORD_TAG_SEPARATOR + b0tPart;
    features[f++] = S0_WORD_B0_WORD + s0wPart + POSITION_SEPARATOR + b0wPart;
    features[f++] = S0_TAG_B0_TAG + s0tPart + POSITION_SEPARATOR + b0tPart;
    features[f++] = S0_WORD_B0_TAG + s0wPart + POSITION_SEPARATOR + b0tPart;
    features[f++] = S0_TAG_B0_WORD + s0tPart + POSITION_SEPARATOR + b0wPart;
    features[f++] = S0_WORD_TAG_B0_TAG
        + s0wPart + WORD_TAG_SEPARATOR + s0tPart + POSITION_SEPARATOR + b0tPart;
    features[f++] = S1_TAG_S0_TAG + s1tPart + POSITION_SEPARATOR + s0tPart;
    features[f++] = S1_TAG_S0_WORD + s1tPart + POSITION_SEPARATOR + s0wPart;
    features[f++] = S1_WORD_S0_TAG + s1wPart + POSITION_SEPARATOR + s0tPart;
    features[f++] = S1_TAG_S0_TAG_B0_TAG
        + s1tPart + POSITION_SEPARATOR + s0tPart + POSITION_SEPARATOR + b0tPart;
    features[f++] = S0_TAG_B0_TAG_B1_TAG
        + s0tPart + POSITION_SEPARATOR + b0tPart + POSITION_SEPARATOR + b1tPart;
    features[f++] = S2_TAG_S1_TAG_S0_TAG
        + s2tPart + POSITION_SEPARATOR + s1tPart + POSITION_SEPARATOR + s0tPart;
    features[f++] = S0_LEFT_DEPENDENT_TAG + s0lct;
    features[f++] = S0_RIGHT_DEPENDENT_TAG + s0rct;
    features[f++] = S1_LEFT_DEPENDENT_TAG + s1lct;
    features[f++] = S1_RIGHT_DEPENDENT_TAG + s1rct;
    features[f++] = S0_LEFT_DEPENDENT_RELATION + s0lcl;
    features[f++] = S0_RIGHT_DEPENDENT_RELATION + s0rcl;
    features[f++] = S1_RIGHT_DEPENDENT_RELATION + s1rcl;
    features[f++] = S1_TAG_S1_RIGHT_DEPENDENT_TAG_S0_TAG
        + s1tPart + POSITION_SEPARATOR + s1rctPart + POSITION_SEPARATOR + s0tPart;
    features[f++] = S0_TAG_S0_LEFT_DEPENDENT_TAG_B0_TAG
        + s0tPart + POSITION_SEPARATOR + s0lctPart + POSITION_SEPARATOR + b0tPart;
    features[f++] = S0_DEPENDENTS + dependents(state, s0);
    features[f++] = S1_DEPENDENTS + dependents(state, s1);
    features[f++] = DISTANCE + distance;
    features[f++] = DISTANCE_S0_TAG_B0_TAG
        + distance + POSITION_SEPARATOR + s0tPart + POSITION_SEPARATOR + b0tPart;
    return features;
  }

  /**
   * Looks up the word or tag at a stack or buffer position.
   *
   * @param values The sentence tokens or their part-of-speech tags.
   * @param index The token index, {@link ArcStandardState#ROOT} or {@link ArcStandardState#NONE}.
   * @return The value, or the marker value for the root and absent positions.
   */
  private String valueAt(String[] values, int index) {
    if (index == ArcStandardState.ROOT) {
      return ROOT_VALUE;
    }
    return index == ArcStandardState.NONE ? NONE_VALUE : literal(values[index]);
  }

  /**
   * Encodes a word, tag, or relation so that it never reads like a marker value: the
   * escape character is doubled, and a value equal to a marker gets the escape character
   * in front. Values without the escape character that are not markers stay unchanged.
   *
   * @param value The value from the sentence or the configuration.
   * @return The encoded value.
   */
  private String literal(String value) {
    if (ROOT_VALUE.equals(value) || NONE_VALUE.equals(value)) {
      return ESCAPE + value;
    }
    return escape(value, ESCAPE, ESCAPE);
  }

  /**
   * Encodes a value returned by {@link #valueAt(String[], int)} or the dependent lookups
   * for use inside a feature combining several values: the escape character goes in front
   * of each separator. Since {@link #literal(String)} already doubled the escape
   * character, every escape character in the result escapes the character after it, and
   * every unescaped separator separates two values.
   *
   * @param value The encoded value.
   * @return The value with its separators escaped.
   */
  private String part(String value) {
    return escape(value, WORD_TAG_SEPARATOR, POSITION_SEPARATOR);
  }

  /**
   * Puts the escape character in front of each occurrence of the given characters.
   *
   * @param value The value to escape.
   * @param first A character to escape.
   * @param second Another character to escape, or {@code first} again.
   * @return The escaped value, or {@code value} itself if it holds neither of them.
   */
  private String escape(String value, char first, char second) {
    StringBuilder escaped = null;
    for (int i = 0; i < value.length(); i++) {
      final char c = value.charAt(i);
      final boolean needsEscape = c == first || c == second;
      if (needsEscape && escaped == null) {
        escaped = new StringBuilder(value.length() + 4).append(value, 0, i);
      }
      if (escaped != null) {
        if (needsEscape) {
          escaped.append(ESCAPE);
        }
        escaped.append(c);
      }
    }
    return escaped == null ? value : escaped.toString();
  }

  /**
   * Looks up the tag of a token's leftmost or rightmost dependent attached so far.
   *
   * @param state The current configuration.
   * @param tags The part-of-speech tags of the sentence.
   * @param index The token index, or a negative value for the root and absent positions.
   * @param leftmost {@code true} for the leftmost dependent, {@code false} for the rightmost.
   * @return The dependent's tag, or the marker value if there is no such dependent.
   */
  private String dependentTag(ArcStandardState state, String[] tags, int index,
      boolean leftmost) {
    if (index < 0) {
      return NONE_VALUE;
    }
    final int dependent =
        leftmost ? state.leftmostDependent(index) : state.rightmostDependent(index);
    return valueAt(tags, dependent);
  }

  /**
   * Looks up the relation of a token's leftmost or rightmost dependent attached so far.
   *
   * @param state The current configuration.
   * @param index The token index, or a negative value for the root and absent positions.
   * @param leftmost {@code true} for the leftmost dependent, {@code false} for the rightmost.
   * @return The dependent's relation, or the marker value if there is no such dependent.
   */
  private String dependentRelation(ArcStandardState state, int index,
      boolean leftmost) {
    if (index < 0) {
      return NONE_VALUE;
    }
    final int dependent =
        leftmost ? state.leftmostDependent(index) : state.rightmostDependent(index);
    if (dependent < 0) {
      return NONE_VALUE;
    }
    final String relation = state.assignedRelation(dependent);
    return relation == null ? NONE_VALUE : literal(relation);
  }

  /**
   * Counts the dependents a token has been assigned so far.
   *
   * @param state The current configuration.
   * @param index The token index, or a negative value for the root and absent positions.
   * @return The count, capped at {@link #MAX_VALENCY}, or the marker value for the root
   *         and absent positions.
   */
  private String dependents(ArcStandardState state, int index) {
    return index < 0 ? NONE_VALUE
        : Integer.toString(Math.min(state.assignedDependents(index), MAX_VALENCY));
  }

  /**
   * Buckets the distance between the stack top and the buffer front.
   *
   * @param s0 The token index on top of the stack, negative for the root or none.
   * @param b0 The token index at the buffer front, negative for none.
   * @return The distance, {@link #LONG_DISTANCE} at {@link #MAX_DISTANCE} or more, or the
   *         marker value if either position is not a token.
   */
  private String distance(int s0, int b0) {
    if (s0 < 0 || b0 < 0) {
      return NONE_VALUE;
    }
    final int distance = b0 - s0;
    return distance >= MAX_DISTANCE ? LONG_DISTANCE : Integer.toString(distance);
  }
}
