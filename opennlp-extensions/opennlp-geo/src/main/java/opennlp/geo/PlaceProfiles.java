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
import java.util.List;
import java.util.Map;

/**
 * Metadata-grounded place similarity: each place carries a numeric profile from a
 * user-supplied table, population density, income, whatever the caller measures, and
 * places compare by the cosine of their standardized profiles. Two places are similar
 * when their measured profiles are, independent of their names or text co-occurrence.
 *
 * <p>The table is tab-separated: a header line with {@code id} followed by metric
 * names, then one place per line with its identifier and one value per metric. The
 * user assembles and supplies the table and thereby accepts the licenses of the
 * sources it was derived from; nothing is bundled. Columns are standardized to mean
 * zero and unit variance at load, so heterogeneous units contribute comparably; a
 * constant column contributes nothing.</p>
 *
 * <p>Instances are immutable and safe to share between threads.</p>
 *
 * @since 3.0.0
 */
public final class PlaceProfiles {

  /**
   * One similarity result: a place paired with its score against the query place.
   *
   * @param id The identifier of the neighboring place, as listed in the table.
   * @param similarity The cosine similarity of the neighbor against the query place.
   */
  public record Neighbor(String id, double similarity) {
  }

  private final Map<String, double[]> profiles;
  private final List<String> metrics;

  private PlaceProfiles(Map<String, double[]> profiles, List<String> metrics) {
    this.profiles = profiles;
    this.metrics = metrics;
  }

  /**
   * Loads a profile table.
   *
   * @param table The tab-separated table, UTF-8: a header with {@code id} and metric
   *              names, then one place per line. Must not be {@code null}.
   * @return The loaded profiles. Never {@code null}.
   * @throws IOException Thrown if reading fails or the table is malformed.
   * @throws IllegalArgumentException Thrown if {@code table} is {@code null}.
   */
  public static PlaceProfiles load(Path table) throws IOException {
    if (table == null) {
      throw new IllegalArgumentException("table must not be null");
    }
    try (InputStream in = Files.newInputStream(table)) {
      return load(in);
    }
  }

  /**
   * Loads a profile table from a stream. Lines may end with LF or CRLF; blank lines
   * and lines whose first character is {@code #} are ignored. If an identifier
   * appears on more than one row, the last row wins.
   *
   * @param tableStream The table content. Must not be {@code null}. Not closed.
   * @return The loaded profiles. Never {@code null}.
   * @throws IOException Thrown if reading fails or the table is malformed.
   * @throws IllegalArgumentException Thrown if {@code tableStream} is {@code null}.
   */
  public static PlaceProfiles load(InputStream tableStream) throws IOException {
    if (tableStream == null) {
      throw new IllegalArgumentException("tableStream must not be null");
    }
    final List<String> lines =
        splitLines(new String(tableStream.readAllBytes(), StandardCharsets.UTF_8));
    if (lines.isEmpty() || lines.get(0).isEmpty()) {
      throw new IOException("the profile table has no header");
    }
    final List<String> header = splitTabs(lines.get(0));
    if (header.size() < 2 || !"id".equals(header.get(0).trim())) {
      throw new IOException("the header must be: id, then at least one metric");
    }
    final int width = header.size() - 1;
    final Map<String, double[]> raw = new HashMap<>();
    for (int i = 1; i < lines.size(); i++) {
      final String line = lines.get(i);
      if (line.isEmpty() || line.startsWith("#")) {
        continue;
      }
      final List<String> fields = splitTabs(line);
      if (fields.size() != header.size()) {
        throw new IOException("row " + (i + 1) + " has " + fields.size()
            + " fields, expected " + header.size());
      }
      final double[] profile = new double[width];
      for (int m = 0; m < width; m++) {
        try {
          profile[m] = Double.parseDouble(fields.get(m + 1).trim());
        } catch (NumberFormatException e) {
          throw new IOException("malformed value in row " + (i + 1), e);
        }
      }
      raw.put(fields.get(0).trim(), profile);
    }
    if (raw.isEmpty()) {
      throw new IOException("the profile table lists no places");
    }
    standardize(raw, width);
    return new PlaceProfiles(Map.copyOf(raw),
        List.copyOf(header.subList(1, header.size())));
  }

  /**
   * Rescales every column to mean zero and unit variance, in place, using the
   * population standard deviation over the listed places. A constant column has zero
   * deviation and standardizes to all zeros instead of dividing by zero, so it
   * contributes nothing to any similarity.
   *
   * @param profiles The raw profiles, keyed by place identifier; mutated in place.
   * @param width The number of metric columns in every profile.
   */
  private static void standardize(Map<String, double[]> profiles, int width) {
    final double[] mean = new double[width];
    for (final double[] profile : profiles.values()) {
      for (int m = 0; m < width; m++) {
        mean[m] += profile[m];
      }
    }
    for (int m = 0; m < width; m++) {
      mean[m] /= profiles.size();
    }
    final double[] variance = new double[width];
    for (final double[] profile : profiles.values()) {
      for (int m = 0; m < width; m++) {
        final double centered = profile[m] - mean[m];
        variance[m] += centered * centered;
      }
    }
    for (final double[] profile : profiles.values()) {
      for (int m = 0; m < width; m++) {
        final double deviation = Math.sqrt(variance[m] / profiles.size());
        profile[m] = deviation == 0.0 ? 0.0 : (profile[m] - mean[m]) / deviation;
      }
    }
  }

  /**
   * @return The metric names in column order, as an immutable list. Never
   *         {@code null}.
   */
  public List<String> metrics() {
    return metrics;
  }

  /**
   * Checks whether a place is profiled.
   *
   * @param id The place identifier.
   * @return {@code true} if the table lists the place.
   */
  public boolean contains(String id) {
    return profiles.containsKey(id);
  }

  /**
   * Compares two places by the cosine of their standardized profiles.
   *
   * @param id The first place identifier. Must be listed.
   * @param otherId The second place identifier. Must be listed.
   * @return The cosine similarity, nominally in {@code [-1, 1]} although rounding can
   *         place it up to one ulp outside; {@code 0} when either profile has no
   *         variance at all.
   * @throws IllegalArgumentException Thrown if an identifier is {@code null} or not
   *         listed.
   */
  public double similarity(String id, String otherId) {
    return cosine(profile(id), profile(otherId));
  }

  /**
   * Finds the places most similar to one place.
   *
   * @param id The query place identifier. Must be listed.
   * @param count The maximum number of neighbors. Must be positive.
   * @return The other places ordered by descending similarity, at most {@code count};
   *         the query place itself is never included, and places with equal scores
   *         appear in no particular relative order. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code id} is {@code null} or not
   *         listed, or {@code count} is not positive.
   */
  public List<Neighbor> mostSimilar(String id, int count) {
    if (count <= 0) {
      throw new IllegalArgumentException("count must be positive: " + count);
    }
    final double[] query = profile(id);
    final List<Neighbor> neighbors = new ArrayList<>();
    for (final Map.Entry<String, double[]> candidate : profiles.entrySet()) {
      if (!candidate.getKey().equals(id)) {
        neighbors.add(new Neighbor(candidate.getKey(),
            cosine(query, candidate.getValue())));
      }
    }
    neighbors.sort((a, b) -> Double.compare(b.similarity(), a.similarity()));
    return List.copyOf(neighbors.subList(0, Math.min(count, neighbors.size())));
  }

  /**
   * Resolves a place identifier to its standardized profile, failing loud on
   * identifiers that are {@code null} or absent from the table.
   *
   * @param id The place identifier to resolve.
   * @return The standardized profile of the place. Never {@code null}.
   * @throws IllegalArgumentException Thrown if {@code id} is {@code null} or not
   *         listed.
   */
  private double[] profile(String id) {
    if (id == null) {
      throw new IllegalArgumentException("id must not be null");
    }
    final double[] profile = profiles.get(id);
    if (profile == null) {
      throw new IllegalArgumentException("the table does not list place: " + id);
    }
    return profile;
  }

  /**
   * Computes the cosine of two equal-length vectors: the dot product divided by the
   * product of the norms.
   *
   * @param a The first vector.
   * @param b The second vector, of the same length as {@code a}.
   * @return The cosine, or {@code 0.0} when either vector is all zeros, which is the
   *         standardized form of a profile without any variance.
   */
  private static double cosine(double[] a, double[] b) {
    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    if (normA == 0.0 || normB == 0.0) {
      return 0.0;
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Splits text into lines on LF, dropping one trailing CR per line so both LF and
   * CRLF endings are accepted.
   *
   * @param content The full table text.
   * @return The lines in order, possibly including empty ones. Never {@code null}.
   */
  private static List<String> splitLines(String content) {
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

  /**
   * Splits one line into fields on tab characters, keeping empty fields so a field
   * count mismatch is detected rather than silently repaired.
   *
   * @param line The line to split.
   * @return The fields in order. Never {@code null} and never empty.
   */
  private static List<String> splitTabs(String line) {
    final List<String> fields = new ArrayList<>();
    int start = 0;
    for (int i = 0; i <= line.length(); i++) {
      if (i == line.length() || line.charAt(i) == '\t') {
        fields.add(line.substring(start, i));
        start = i + 1;
      }
    }
    return fields;
  }
}
