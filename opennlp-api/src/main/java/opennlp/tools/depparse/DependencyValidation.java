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

/**
 * Argument checks shared by {@link DependencySample} and the {@link DependencyParser}
 * implementations.
 *
 * @since 3.0.0
 */
final class DependencyValidation {

  /** Prevents construction of this utility class. */
  private DependencyValidation() {
  }

  /**
   * Validates the token and tag arrays of a sentence.
   *
   * @param tokens The token array.
   * @param tags The part-of-speech tags aligned with {@code tokens}.
   * @throws IllegalArgumentException Thrown if an array is {@code null}, {@code tokens}
   *         is empty, the lengths do not match, or an entry is {@code null}.
   */
  static void checkTokensAndTags(String[] tokens, String[] tags) {
    if (tokens == null || tags == null) {
      throw new IllegalArgumentException("tokens and tags must not be null");
    }
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (tokens.length != tags.length) {
      throw new IllegalArgumentException("tokens and tags must have the same length: "
          + tokens.length + " != " + tags.length);
    }
    for (int i = 0; i < tokens.length; i++) {
      if (tokens[i] == null) {
        throw new IllegalArgumentException("token must not be null at index " + i);
      }
      if (tags[i] == null) {
        throw new IllegalArgumentException("tag must not be null at index " + i);
      }
    }
  }
}
