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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Generates the checked-in Unicode 17 Latin tokenizer-policy ranges from official UCD files. */
public final class UnicodeLatinTokenizerPolicyGenerator {

  private static final String VERSION = "17.0.0";
  private static final String UNICODE_DATA_SHA256 =
      "2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c";
  private static final String SCRIPTS_SHA256 =
      "9f5e50d3abaee7d6ce09480f325c706f485ae3240912527e651954d2d6b035bf";
  private static final String SCRIPT_EXTENSIONS_SHA256 =
      "ec2107e58825a1586acee8e0911ce18260394ac8b87e535ca325f1ccbeb06bc6";

  private UnicodeLatinTokenizerPolicyGenerator() {
  }

  public static void main(String[] args) throws Exception {
    if (args.length != 4) {
      throw new IllegalArgumentException("Usage: UnicodeLatinTokenizerPolicyGenerator "
          + "UnicodeData.txt Scripts.txt ScriptExtensions.txt output.java");
    }
    final Path unicodeDataPath = Path.of(args[0]);
    final Path scriptsPath = Path.of(args[1]);
    final Path scriptExtensionsPath = Path.of(args[2]);
    verifySource(unicodeDataPath, UNICODE_DATA_SHA256, null);
    verifySource(scriptsPath, SCRIPTS_SHA256, "# Scripts-" + VERSION + ".txt");
    verifySource(scriptExtensionsPath, SCRIPT_EXTENSIONS_SHA256,
        "# ScriptExtensions-" + VERSION + ".txt");

    final UnicodeData unicodeData = readUnicodeData(unicodeDataPath);
    final BitSet latinScript = readProperty(scriptsPath, "Latin");
    final BitSet latinExtensions = readScriptExtensions(scriptExtensionsPath, "Latn");
    final BitSet letters = (BitSet) latinScript.clone();
    letters.and(unicodeData.letters);

    final BitSet marks = (BitSet) latinScript.clone();
    marks.or(latinExtensions);
    marks.and(unicodeData.marks);
    for (int letter = letters.nextSetBit(0); letter >= 0;
         letter = letters.nextSetBit(letter + 1)) {
      addCanonicalMarks(letter, unicodeData, marks, new BitSet());
    }
    if (letters.intersects(marks)) {
      throw new IllegalStateException("Generated letter and mark sets overlap");
    }

    Files.createDirectories(Path.of(args[3]).toAbsolutePath().getParent());
    Files.writeString(Path.of(args[3]), render(letters, marks), StandardCharsets.UTF_8);
  }

  private static void verifySource(Path path, String expectedSha256, String versionHeader)
      throws IOException, NoSuchAlgorithmException {
    final byte[] bytes = Files.readAllBytes(path);
    final String actual = hex(MessageDigest.getInstance("SHA-256").digest(bytes));
    if (!expectedSha256.equals(actual)) {
      throw new IllegalArgumentException(path + " SHA-256 was " + actual
          + ", expected " + expectedSha256);
    }
    if (versionHeader != null) {
      final String firstLine = Files.readAllLines(path, StandardCharsets.UTF_8).get(0);
      if (!versionHeader.equals(firstLine)) {
        throw new IllegalArgumentException(path + " has unexpected version header " + firstLine);
      }
    }
  }

  private static UnicodeData readUnicodeData(Path path) throws IOException {
    final BitSet letters = new BitSet();
    final BitSet marks = new BitSet();
    final Map<Integer, int[]> decompositions = new HashMap<>();
    int pendingFirst = -1;
    String pendingCategory = null;
    for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
      final List<String> fields = split(line, ';');
      if (fields.size() != 15) {
        throw new IllegalArgumentException("Malformed UnicodeData line: " + line);
      }
      final int codePoint = Integer.parseInt(fields.get(0), 16);
      final String name = fields.get(1);
      final String category = fields.get(2);
      if (name.endsWith(", First>")) {
        pendingFirst = codePoint;
        pendingCategory = category;
        continue;
      }
      if (name.endsWith(", Last>")) {
        if (pendingFirst < 0 || !category.equals(pendingCategory)) {
          throw new IllegalArgumentException("Unpaired UnicodeData range: " + line);
        }
        setCategory(pendingFirst, codePoint, category, letters, marks);
        pendingFirst = -1;
        pendingCategory = null;
        continue;
      }
      setCategory(codePoint, codePoint, category, letters, marks);
      final String decomposition = fields.get(5);
      if (!decomposition.isEmpty() && decomposition.charAt(0) != '<') {
        final List<String> values = split(decomposition, ' ');
        final int[] decoded = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
          decoded[i] = Integer.parseInt(values.get(i), 16);
        }
        decompositions.put(codePoint, decoded);
      }
    }
    if (pendingFirst >= 0) {
      throw new IllegalArgumentException("Unclosed UnicodeData range");
    }
    return new UnicodeData(letters, marks, decompositions);
  }

  private static void setCategory(int first, int last, String category,
      BitSet letters, BitSet marks) {
    if (category.charAt(0) == 'L') {
      letters.set(first, last + 1);
    } else if (category.charAt(0) == 'M') {
      marks.set(first, last + 1);
    }
  }

  private static BitSet readProperty(Path path, String wanted) throws IOException {
    final BitSet result = new BitSet();
    for (String raw : Files.readAllLines(path, StandardCharsets.UTF_8)) {
      final String line = stripComment(raw).strip();
      if (line.isEmpty()) {
        continue;
      }
      final int semicolon = line.indexOf(';');
      if (semicolon < 0) {
        throw new IllegalArgumentException("Malformed property line: " + raw);
      }
      if (wanted.equals(line.substring(semicolon + 1).strip())) {
        addRange(result, line.substring(0, semicolon).strip());
      }
    }
    return result;
  }

  private static BitSet readScriptExtensions(Path path, String wanted) throws IOException {
    final BitSet result = new BitSet();
    for (String raw : Files.readAllLines(path, StandardCharsets.UTF_8)) {
      final String line = stripComment(raw).strip();
      if (line.isEmpty()) {
        continue;
      }
      final int semicolon = line.indexOf(';');
      if (semicolon < 0) {
        throw new IllegalArgumentException("Malformed ScriptExtensions line: " + raw);
      }
      if (split(line.substring(semicolon + 1).strip(), ' ').contains(wanted)) {
        addRange(result, line.substring(0, semicolon).strip());
      }
    }
    return result;
  }

  private static void addCanonicalMarks(int codePoint, UnicodeData data,
      BitSet marks, BitSet visiting) {
    final int[] decomposition = data.decompositions.get(codePoint);
    if (decomposition == null || visiting.get(codePoint)) {
      return;
    }
    visiting.set(codePoint);
    for (int part : decomposition) {
      if (data.marks.get(part)) {
        marks.set(part);
      }
      addCanonicalMarks(part, data, marks, visiting);
    }
    visiting.clear(codePoint);
  }

  private static void addRange(BitSet set, String value) {
    final int dots = value.indexOf("..");
    final int first = Integer.parseInt(dots < 0 ? value : value.substring(0, dots), 16);
    final int last = Integer.parseInt(dots < 0 ? value : value.substring(dots + 2), 16);
    set.set(first, last + 1);
  }

  private static String render(BitSet letters, BitSet marks) {
    final StringBuilder out = new StringBuilder(32_000);
    out.append("/*\n * Licensed to the Apache Software Foundation (ASF) under one or more\n")
        .append(" * contributor license agreements.  See the NOTICE file distributed with\n")
        .append(" * this work for additional information regarding copyright ownership.\n")
        .append(" * The ASF licenses this file to You under the Apache License, Version 2.0\n")
        .append(" * (the \"License\"); you may not use this file except in compliance with\n")
        .append(" * the License.  You may obtain a copy of the License at\n *\n")
        .append(" *     http://www.apache.org/licenses/LICENSE-2.0\n *\n")
        .append(" * Unless required by applicable law or agreed to in writing, software\n")
        .append(" * distributed under the License is distributed on an \"AS IS\" BASIS,\n")
        .append(" * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n")
        .append(" * See the License for the specific language governing permissions and\n")
        .append(" * limitations under the License.\n")
        .append(" */\npackage opennlp.tools.tokenize;\n\n")
        .append("import opennlp.tools.util.normalizer.CodePointSet;\n\n")
        .append("/**\n * Generated Unicode 17.0.0 Latin tokenizer-policy ranges.\n")
        .append(" *\n * <p>Derived from UnicodeData.txt, Scripts.txt, and ScriptExtensions.txt,\n")
        .append(" * copyright Unicode, Inc., under Unicode License V3. See LICENSE and NOTICE.\n")
        .append(" * Resolved inventory: ").append(letters.cardinality()).append(" letters and ")
        .append(marks.cardinality()).append(" continuation marks.\n")
        .append(" * Regenerate with {@code dev/UnicodeLatinTokenizerPolicyGenerator.java}.\n")
        .append(" */\nfinal class Unicode17LatinTokenizerData {\n\n")
        .append("  private static final int[] LETTER_RANGES = {\n");
    appendRanges(out, letters);
    out.append("  };\n\n  private static final int[] MARK_RANGES = {\n");
    appendRanges(out, marks);
    out.append("  };\n\n  private Unicode17LatinTokenizerData() {\n  }\n\n")
        .append("  static CodePointSet letters() {\n")
        .append("    return fromRanges(LETTER_RANGES);\n  }\n\n")
        .append("  static CodePointSet marks() {\n")
        .append("    return fromRanges(MARK_RANGES);\n  }\n\n")
        .append("  private static CodePointSet fromRanges(int[] ranges) {\n")
        .append("    CodePointSet result = CodePointSet.of();\n")
        .append("    for (int i = 0; i < ranges.length; i += 2) {\n")
        .append("      result = result.union(CodePointSet.ofRange(ranges[i], ranges[i + 1]));\n")
        .append("    }\n    return result;\n  }\n}\n");
    return out.toString();
  }

  private static void appendRanges(StringBuilder out, BitSet values) {
    int column = 0;
    for (int start = values.nextSetBit(0); start >= 0;) {
      final int end = values.nextClearBit(start) - 1;
      if (column == 0) {
        out.append("      ");
      }
      out.append(String.format("0x%04X, 0x%04X", start, end));
      start = values.nextSetBit(end + 1);
      if (start >= 0) {
        out.append(',');
      }
      column++;
      if (column == 4 || start < 0) {
        out.append('\n');
        column = 0;
      } else {
        out.append(' ');
      }
    }
  }

  private static List<String> split(String value, char delimiter) {
    final List<String> parts = new ArrayList<>();
    int start = 0;
    for (int i = 0; i <= value.length(); i++) {
      if (i == value.length() || value.charAt(i) == delimiter) {
        if (i > start) {
          parts.add(value.substring(start, i));
        } else if (delimiter != ' ') {
          parts.add("");
        }
        start = i + 1;
      }
    }
    return parts;
  }

  private static String stripComment(String value) {
    final int hash = value.indexOf('#');
    return hash < 0 ? value : value.substring(0, hash);
  }

  private static String hex(byte[] bytes) {
    final StringBuilder out = new StringBuilder(bytes.length * 2);
    for (byte value : bytes) {
      out.append(String.format("%02x", value & 0xff));
    }
    return out.toString();
  }

  private record UnicodeData(BitSet letters, BitSet marks,
      Map<Integer, int[]> decompositions) { }
}
