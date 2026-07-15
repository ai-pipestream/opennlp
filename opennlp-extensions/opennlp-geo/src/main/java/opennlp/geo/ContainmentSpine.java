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

package opennlp.geo;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import opennlp.tools.geo.PlaceAncestor;
import opennlp.tools.geo.PlaceHierarchy;

/**
 * An immutable, in-memory containment hierarchy over user-supplied place tables: each
 * place carries its parent, name, and type, and {@link #ancestors(String)} walks the
 * chain outward. No hierarchy data is bundled; the user supplies the tables and thereby
 * accepts their licenses, the pattern shared with the gazetteers and dictionaries.
 *
 * <p>Two table formats load through the builder. The neutral tab-separated format
 * carries {@code id}, {@code parent_id}, {@code name}, and {@code type} columns, one
 * place per line, empty parent for roots; a containment table derived from any source,
 * for example an administrative-territory query result, fits it. The Who's On First
 * meta CSV format is read directly by its header columns, so the published per-placetype
 * tables load without conversion; non-positive parent identifiers mean no usable
 * parent, as in the source data.</p>
 *
 * <p>Instances are immutable and safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public final class ContainmentSpine implements PlaceHierarchy {

  /** The longest chain {@link #ancestors(String)} follows before assuming a cycle. */
  private static final int MAX_DEPTH = 64;

  private record Node(String parentId, String name, String type) {
  }

  private final Map<String, Node> places;

  private ContainmentSpine(Map<String, Node> places) {
    this.places = places;
  }

  /**
   * @return A builder collecting places from tables and direct additions. Never
   *         {@code null}.
   */
  public static Builder builder() {
    return new Builder();
  }

  @Override
  public List<PlaceAncestor> ancestors(String id) {
    if (id == null) {
      throw new IllegalArgumentException("id must not be null");
    }
    final List<PlaceAncestor> chain = new ArrayList<>();
    final Set<String> visited = new HashSet<>();
    visited.add(id);
    Node node = places.get(id);
    String parentId = node == null ? null : node.parentId();
    while (parentId != null && chain.size() < MAX_DEPTH && visited.add(parentId)) {
      final Node parent = places.get(parentId);
      if (parent == null) {
        break;
      }
      chain.add(new PlaceAncestor(parentId, parent.name(), parent.type()));
      parentId = parent.parentId();
    }
    return chain;
  }

  /**
   * Collects places for a {@link ContainmentSpine}.
   */
  public static final class Builder {

    private final Map<String, Node> places = new HashMap<>();

    private Builder() {
    }

    /**
     * Adds one place.
     *
     * @param id The place identifier. Must not be {@code null} or blank.
     * @param parentId The parent identifier, or {@code null} for a root.
     * @param name The place name. Must not be {@code null} or blank.
     * @param type The place type. Must not be {@code null} or blank.
     * @return This builder.
     * @throws IllegalArgumentException Thrown if a required value is {@code null} or
     *         blank.
     */
    public Builder add(String id, String parentId, String name, String type) {
      if (id == null || id.isBlank()) {
        throw new IllegalArgumentException("id must not be null or blank");
      }
      if (name == null || name.isBlank()) {
        throw new IllegalArgumentException("name must not be null or blank");
      }
      if (type == null || type.isBlank()) {
        throw new IllegalArgumentException("type must not be null or blank");
      }
      places.put(id, new Node(parentId, name, type));
      return this;
    }

    /**
     * Loads a neutral tab-separated containment table: {@code id}, {@code parent_id},
     * {@code name}, {@code type} per line, empty parent for roots, {@code #} comment
     * lines skipped.
     *
     * @param table The table file, UTF-8. Must not be {@code null}.
     * @return This builder.
     * @throws IOException Thrown if reading fails or a line is malformed.
     * @throws IllegalArgumentException Thrown if {@code table} is {@code null}.
     */
    public Builder addTable(Path table) throws IOException {
      if (table == null) {
        throw new IllegalArgumentException("table must not be null");
      }
      int lineNumber = 0;
      for (final String line : readLines(table)) {
        lineNumber++;
        if (line.isEmpty() || line.startsWith("#")) {
          continue;
        }
        final List<String> fields = splitOn(line, '\t');
        if (fields.size() < 4) {
          throw new IOException("malformed containment line " + lineNumber
              + " in " + table);
        }
        final String parent = fields.get(1).trim();
        add(fields.get(0).trim(), parent.isEmpty() ? null : parent,
            fields.get(2).trim(), fields.get(3).trim());
      }
      return this;
    }

    /**
     * Loads a Who's On First meta CSV by its header columns {@code id},
     * {@code parent_id}, {@code name}, and {@code placetype}. Rows with a non-positive
     * parent identifier become roots, matching the source convention for places
     * without a usable parent.
     *
     * @param metaCsv The meta CSV file, UTF-8. Must not be {@code null}.
     * @return This builder.
     * @throws IOException Thrown if reading fails, a required column is missing, or a
     *         row is malformed.
     * @throws IllegalArgumentException Thrown if {@code metaCsv} is {@code null}.
     */
    public Builder addWofMeta(Path metaCsv) throws IOException {
      if (metaCsv == null) {
        throw new IllegalArgumentException("metaCsv must not be null");
      }
      final List<String> lines = readLines(metaCsv);
      if (lines.isEmpty()) {
        throw new IOException("empty meta CSV: " + metaCsv);
      }
      final List<String> header = parseCsvRecord(lines.get(0));
      final int idColumn = header.indexOf("id");
      final int parentColumn = header.indexOf("parent_id");
      final int nameColumn = header.indexOf("name");
      final int typeColumn = header.indexOf("placetype");
      if (idColumn < 0 || parentColumn < 0 || nameColumn < 0 || typeColumn < 0) {
        throw new IOException(
            "meta CSV lacks id, parent_id, name, or placetype columns: " + metaCsv);
      }
      for (int i = 1; i < lines.size(); i++) {
        if (lines.get(i).isEmpty()) {
          continue;
        }
        final List<String> fields = parseCsvRecord(lines.get(i));
        if (fields.size() <= Math.max(Math.max(idColumn, parentColumn),
            Math.max(nameColumn, typeColumn))) {
          throw new IOException("short row " + (i + 1) + " in " + metaCsv);
        }
        final String name = fields.get(nameColumn).trim();
        final String type = fields.get(typeColumn).trim();
        if (name.isEmpty() || type.isEmpty()) {
          continue;
        }
        final String parent = fields.get(parentColumn).trim();
        add(fields.get(idColumn).trim(), positiveOrNull(parent), name, type);
      }
      return this;
    }

    /**
     * Builds the spine.
     *
     * @return The immutable hierarchy. Never {@code null}.
     * @throws IllegalArgumentException Thrown if no place was added.
     */
    public ContainmentSpine build() {
      if (places.isEmpty()) {
        throw new IllegalArgumentException("no places were added");
      }
      return new ContainmentSpine(Map.copyOf(places));
    }

    private static String positiveOrNull(String parent) {
      if (parent.isEmpty() || parent.startsWith("-") || "0".equals(parent)) {
        return null;
      }
      return parent;
    }
  }

  private static List<String> readLines(Path file) throws IOException {
    final String content;
    try (InputStream in = Files.newInputStream(file)) {
      content = new String(in.readAllBytes(), StandardCharsets.UTF_8);
    }
    final List<String> lines = new ArrayList<>();
    int start = 0;
    for (int i = 0; i <= content.length(); i++) {
      if (i == content.length() || content.charAt(i) == '\n') {
        int end = i;
        if (end > start && content.charAt(end - 1) == '\r') {
          end--;
        }
        lines.add(content.substring(start, end));
        start = i + 1;
      }
    }
    return lines;
  }

  private static List<String> splitOn(String line, char separator) {
    final List<String> fields = new ArrayList<>();
    int start = 0;
    for (int i = 0; i <= line.length(); i++) {
      if (i == line.length() || line.charAt(i) == separator) {
        fields.add(line.substring(start, i));
        start = i + 1;
      }
    }
    return fields;
  }

  /** Parses one CSV record with double-quote quoting and doubled-quote escapes. */
  private static List<String> parseCsvRecord(String line) {
    final List<String> fields = new ArrayList<>();
    final StringBuilder field = new StringBuilder();
    boolean quoted = false;
    for (int i = 0; i < line.length(); i++) {
      final char c = line.charAt(i);
      if (quoted) {
        if (c == '"') {
          if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
            field.append('"');
            i++;
          } else {
            quoted = false;
          }
        } else {
          field.append(c);
        }
      } else if (c == '"') {
        quoted = true;
      } else if (c == ',') {
        fields.add(field.toString());
        field.setLength(0);
      } else {
        field.append(c);
      }
    }
    fields.add(field.toString());
    return fields;
  }
}
