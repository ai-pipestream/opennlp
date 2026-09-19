/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package opennlp.tools.tokenize;

import opennlp.tools.util.normalizer.CodePointSet;

/**
 * Generated Unicode 17.0.0 Latin tokenizer-policy ranges.
 *
 * <p>Derived from UnicodeData.txt, Scripts.txt, and ScriptExtensions.txt,
 * copyright Unicode, Inc., under Unicode License V3. See LICENSE and NOTICE.
 * Resolved inventory: 1453 letters and 51 continuation marks.
 * Regenerate with {@code dev/UnicodeLatinTokenizerPolicyGenerator.java}.
 */
final class Unicode17LatinTokenizerData {

  private static final int[] LETTER_RANGES = {
      0x0041, 0x005A, 0x0061, 0x007A, 0x00AA, 0x00AA, 0x00BA, 0x00BA,
      0x00C0, 0x00D6, 0x00D8, 0x00F6, 0x00F8, 0x02B8, 0x02E0, 0x02E4,
      0x1D00, 0x1D25, 0x1D2C, 0x1D5C, 0x1D62, 0x1D65, 0x1D6B, 0x1D77,
      0x1D79, 0x1DBE, 0x1E00, 0x1EFF, 0x2071, 0x2071, 0x207F, 0x207F,
      0x2090, 0x209C, 0x212A, 0x212B, 0x2132, 0x2132, 0x214E, 0x214E,
      0x2183, 0x2184, 0x2C60, 0x2C7F, 0xA722, 0xA787, 0xA78B, 0xA7DC,
      0xA7F1, 0xA7FF, 0xAB30, 0xAB5A, 0xAB5C, 0xAB64, 0xAB66, 0xAB69,
      0xFB00, 0xFB06, 0xFF21, 0xFF3A, 0xFF41, 0xFF5A, 0x10780, 0x10785,
      0x10787, 0x107B0, 0x107B2, 0x107BA, 0x1DF00, 0x1DF1E, 0x1DF25, 0x1DF2A
  };

  private static final int[] MARK_RANGES = {
      0x0300, 0x0311, 0x0313, 0x0313, 0x031B, 0x031B, 0x0323, 0x0328,
      0x032D, 0x032E, 0x0330, 0x0331, 0x0358, 0x0358, 0x035E, 0x035E,
      0x0363, 0x036F, 0x0485, 0x0486, 0x0951, 0x0952, 0x1DF8, 0x1DF8,
      0x20F0, 0x20F0
  };

  private Unicode17LatinTokenizerData() {
  }

  static CodePointSet letters() {
    return fromRanges(LETTER_RANGES);
  }

  static CodePointSet marks() {
    return fromRanges(MARK_RANGES);
  }

  private static CodePointSet fromRanges(int[] ranges) {
    CodePointSet result = CodePointSet.of();
    for (int i = 0; i < ranges.length; i += 2) {
      result = result.union(CodePointSet.ofRange(ranges[i], ranges[i + 1]));
    }
    return result;
  }
}
