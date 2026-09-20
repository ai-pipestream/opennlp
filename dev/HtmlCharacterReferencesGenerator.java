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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;

/** Generates OpenNLP's HTML named-character-reference table from WHATWG entities.json. */
public final class HtmlCharacterReferencesGenerator {
  private static final char DELIMITER = '\uffff';
  private record Entity(String name, String value) {}

  public static void main(String[] args) throws Exception {
    if (args.length != 3) throw new IllegalArgumentException("input, Java output, vector output required");
    List<Entity> entities = new ArrayList<>();
    for (String line : Files.readAllLines(Path.of(args[0]), StandardCharsets.UTF_8)) {
      String text = line.trim();
      if (text.equals("{") || text.equals("}")) continue;
      if (!text.startsWith("\"&") || !text.contains("\"codepoints\": [")
          || !text.contains("\"characters\":")) {
        throw new IllegalArgumentException("Unexpected entities.json line: " + line);
      }
      int nameEnd = text.indexOf('"', 2);
      String name = text.substring(2, nameEnd);
      int arrayStart = text.indexOf('[', nameEnd) + 1;
      int arrayEnd = text.indexOf(']', arrayStart);
      String values = text.substring(arrayStart, arrayEnd).trim();
      StringBuilder decoded = new StringBuilder(2);
      int start = 0;
      while (start < values.length()) {
        int comma = values.indexOf(',', start);
        int end = comma < 0 ? values.length() : comma;
        int codePoint = Integer.parseInt(values.substring(start, end).trim());
        if (!Character.isValidCodePoint(codePoint)) throw new IllegalArgumentException("Invalid code point");
        decoded.appendCodePoint(codePoint);
        start = end + 1;
      }
      entities.add(new Entity(name, decoded.toString()));
    }
    entities.sort(Comparator.comparing(Entity::name));
    if (entities.size() != 2231) throw new IllegalArgumentException("Expected 2231 entities");
    writeJava(Path.of(args[1]), entities);
    writeVectors(Path.of(args[2]), entities);
  }

  private static void writeJava(Path output, List<Entity> entities) throws Exception {
    StringBuilder out = new StringBuilder(160_000);
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
        .append(" * limitations under the License.\n */\n")
        .append("package opennlp.tools.util;\n\n")
        .append("/** Generated from WHATWG entities.json. Do not edit manually. */\n")
        .append("final class HtmlCharacterReferenceData {\n")
        .append("  static final String[] NAMES = split(\n");
    appendPacked(out, entities, true);
    out.append("  );\n  static final String[] VALUES = split(\n");
    appendPacked(out, entities, false);
    out.append("  );\n\n  private static String[] split(String data) {\n")
        .append("    String[] values = new String[2231];\n    int start = 0;\n")
        .append("    for (int i = 0; i < values.length; i++) {\n")
        .append("      int end = data.indexOf('").append(DELIMITER).append("', start);\n")
        .append("      values[i] = data.substring(start, end);\n      start = end + 1;\n    }\n")
        .append("    return values;\n  }\n\n  private HtmlCharacterReferenceData() {}\n}\n");
    Files.createDirectories(output.getParent());
    Files.writeString(output, out, StandardCharsets.UTF_8);
  }

  private static void appendPacked(StringBuilder out, List<Entity> entities, boolean names) {
    StringBuilder chunk = new StringBuilder(120);
    for (Entity entity : entities) {
      String value = escape(names ? entity.name() : entity.value()) + DELIMITER;
      if (chunk.length() + value.length() > 100) {
        out.append("      \"").append(chunk).append("\" +\n");
        chunk.setLength(0);
      }
      chunk.append(value);
    }
    out.append("      \"").append(chunk).append("\"\n");
  }

  private static void writeVectors(Path output, List<Entity> entities) throws Exception {
    StringBuilder out = new StringBuilder(80_000);
    for (Entity entity : entities) {
      out.append(entity.name()).append('\t');
      entity.value().codePoints().forEach(cp -> out.append(Integer.toHexString(cp)).append(','));
      out.append('\n');
    }
    Files.createDirectories(output.getParent());
    Files.writeString(output, out, StandardCharsets.UTF_8);
  }

  private static String escape(String value) {
    StringBuilder result = new StringBuilder(value.length());
    for (int i = 0; i < value.length(); i++) {
      char ch = value.charAt(i);
      if (ch == '\\' || ch == '"') result.append('\\').append(ch);
      else if (ch == '\n') result.append("\\n");
      else if (ch == '\t') result.append("\\t");
      else if (ch == '\r') result.append("\\r");
      else if (ch == '\b') result.append("\\b");
      else if (ch == '\f') result.append("\\f");
      else if (ch == DELIMITER) result.append(ch);
      else if (ch < 0x20 || ch > 0x7e) result.append("\\u")
          .append(HexFormat.of().withUpperCase().toHexDigits(ch));
      else result.append(ch);
    }
    return result.toString();
  }
}
