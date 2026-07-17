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
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import opennlp.tools.util.StringUtil;

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
 * <p>Every value the table holds must be a finite number, and every column must
 * standardize within the range of a double. A column of finite values whose sum or
 * whose summed squared deviations exceed that range is rejected at load, naming the
 * metric, rather than being silently reduced to zeros or to {@code NaN}: such a column
 * carries signal that the arithmetic cannot express, and a caller who rescales it into
 * range gets the same similarities back. Scores are therefore always real numbers.</p>
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
   * @throws IOException Thrown if reading fails, the table has no header, a row does
   *         not list exactly one value per metric, a value is not a finite number, or a
   *         metric column overflows a double.
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
   * Loads a profile table from a stream. Lines may end with LF or CRLF. A line is
   * ignored wherever it appears, ahead of the header as readily as between rows, when
   * it is blank or when its first non-blank character is {@code #}, so a table may
   * carry an attribution comment above its header. The header is the first line that is
   * not ignored. Identifiers and values are stripped of surrounding whitespace. If an
   * identifier appears on more than one row, the last row wins.
   *
   * <p>Columns are standardized once every row has been read, so a column that
   * overflows a double is reported by metric name rather than by row: no single row is
   * at fault when every cell in the column is finite.</p>
   *
   * @param tableStream The table content. Must not be {@code null}. Not closed.
   * @return The loaded profiles. Never {@code null}.
   * @throws IOException Thrown if reading fails, the table has no header, a row does
   *         not list exactly one value per metric, a value is not a finite number, or a
   *         metric column overflows a double.
   * @throws IllegalArgumentException Thrown if {@code tableStream} is {@code null}.
   */
  public static PlaceProfiles load(InputStream tableStream) throws IOException {
    if (tableStream == null) {
      throw new IllegalArgumentException("tableStream must not be null");
    }
    final List<String> lines =
        splitLines(new String(tableStream.readAllBytes(), StandardCharsets.UTF_8));
    int headerLine = -1;
    for (int i = 0; i < lines.size(); i++) {
      if (!isIgnored(lines.get(i))) {
        headerLine = i;
        break;
      }
    }
    if (headerLine < 0) {
      throw new IOException("the profile table has no header");
    }
    final List<String> header = splitTabs(lines.get(headerLine));
    if (header.size() < 2 || !"id".equals(strip(header.get(0)))) {
      throw new IOException("the header must be: id, then at least one metric");
    }
    final int width = header.size() - 1;
    final Map<String, double[]> raw = new HashMap<>();
    for (int i = headerLine + 1; i < lines.size(); i++) {
      final String line = lines.get(i);
      if (isIgnored(line)) {
        continue;
      }
      final List<String> fields = splitTabs(line);
      if (fields.size() != header.size()) {
        throw new IOException("row " + (i + 1) + " has " + fields.size()
            + " fields, expected " + header.size());
      }
      final double[] profile = new double[width];
      for (int m = 0; m < width; m++) {
        profile[m] = metricValue(strip(fields.get(m + 1)), i + 1);
      }
      raw.put(strip(fields.get(0)), profile);
    }
    if (raw.isEmpty()) {
      throw new IOException("the profile table lists no places");
    }
    final List<String> metricNames = List.copyOf(header.subList(1, header.size()));
    standardize(raw, metricNames);
    return new PlaceProfiles(Map.copyOf(raw), metricNames);
  }

  /**
   * Parses one metric cell, accepting only the finite numbers a metric table can
   * meaningfully hold.
   *
   * <p>{@link Double#parseDouble} is deliberately not trusted on its own. It accepts
   * {@code NaN}, {@code Infinity} and {@code -Infinity}, none of which is a measurement,
   * and a single such cell would not stay in its own row: standardization takes the mean
   * and deviation over the whole column, so one non-finite cell turns every profile in
   * the table into non-finite values, and the guards downstream, which compare against
   * {@code 0.0}, do not fire for them. It also accepts the Java {@code f} and {@code d}
   * literal suffixes, which are Java source syntax rather than table data and indicate
   * that whatever generated the table leaked its own literal syntax into the output.
   * Both are rejected here, at the only point where the offending row is still known.</p>
   *
   * @param text The cell content, already stripped of surrounding whitespace.
   * @param row The one-based line number of the row, for the failure message.
   * @return The parsed value, always finite.
   * @throws IOException Thrown if the cell is not a number, carries a Java literal
   *         suffix, or is not finite.
   */
  private static double metricValue(String text, int row) throws IOException {
    if (hasJavaLiteralSuffix(text)) {
      throw new IOException("malformed value in row " + row + ": " + text);
    }
    final double value;
    try {
      value = Double.parseDouble(text);
    } catch (NumberFormatException e) {
      throw new IOException("malformed value in row " + row + ": " + text, e);
    }
    if (!Double.isFinite(value)) {
      throw new IOException("non-finite value in row " + row + ": " + text);
    }
    return value;
  }

  /**
   * Checks whether a cell ends in a Java float or double literal suffix. No decimal or
   * hexadecimal number written as data ends in one of these letters, so testing the
   * final character is enough and no grammar has to be restated here.
   *
   * @param text The cell content, already stripped of surrounding whitespace.
   * @return {@code true} if the final character is {@code f}, {@code F}, {@code d} or
   *         {@code D}.
   */
  private static boolean hasJavaLiteralSuffix(String text) {
    if (text.isEmpty()) {
      return false;
    }
    final char last = text.charAt(text.length() - 1);
    return last == 'f' || last == 'F' || last == 'd' || last == 'D';
  }

  /**
   * Checks whether a line carries no data: one that is blank or whose first non-blank
   * character opens a comment.
   *
   * @param line The raw line, without its line ending.
   * @return {@code true} if the line must be skipped.
   */
  private static boolean isIgnored(String line) {
    final String stripped = strip(line);
    return stripped.isEmpty() || stripped.charAt(0) == '#';
  }

  /**
   * Removes leading and trailing whitespace, using the toolkit's whitespace definition
   * rather than {@link String#trim()}, which leaves the no-break spaces that tables
   * copied out of a PDF or a rendered web table routinely carry.
   *
   * @param text The text to strip.
   * @return The text without surrounding whitespace. Never {@code null}.
   */
  private static String strip(String text) {
    int start = 0;
    int end = text.length();
    while (start < end && StringUtil.isWhitespace(text.charAt(start))) {
      start++;
    }
    while (end > start && StringUtil.isWhitespace(text.charAt(end - 1))) {
      end--;
    }
    return text.substring(start, end);
  }

  /**
   * Rescales every column to mean zero and unit variance, in place, using the
   * population standard deviation over the listed places.
   *
   * <p>A constant column has zero deviation and standardizes to all zeros instead of
   * dividing by zero, so it contributes nothing to any similarity. That is the honest
   * answer for such a column, because a metric that never varies genuinely
   * distinguishes no two places.</p>
   *
   * <p>A column that overflows a double is rejected instead, even though every cell in
   * it is finite. Zeroing it would report the same thing as a constant column, that the
   * metric carries no signal, about a column that demonstrably does carry signal and
   * merely exceeds the arithmetic. Overflow is caught in both of the places it appears.
   * The summed column can exceed the range, leaving an infinite mean, an infinite
   * centered value, and a {@code NaN} profile that no downstream guard rejects because
   * those guards compare against {@code 0.0}. The summed squared deviations can exceed
   * it too, leaving an infinite deviation that would divide every value in the column
   * down to zero. Both describe a table the caller can trivially rescale, so both say
   * so rather than answering from a discarded metric.</p>
   *
   * @param profiles The raw profiles, keyed by place identifier; mutated in place.
   * @param metrics The metric names in column order, used to name a rejected column.
   * @throws IOException Thrown if the mean or the standard deviation of a column is not
   *         finite, which means the column overflows a double.
   */
  private static void standardize(Map<String, double[]> profiles, List<String> metrics)
      throws IOException {
    final int width = metrics.size();
    final double[] mean = new double[width];
    for (final double[] profile : profiles.values()) {
      for (int m = 0; m < width; m++) {
        mean[m] += profile[m];
      }
    }
    for (int m = 0; m < width; m++) {
      mean[m] /= profiles.size();
      checkColumnFinite(mean[m], metrics.get(m));
    }
    final double[] variance = new double[width];
    for (final double[] profile : profiles.values()) {
      for (int m = 0; m < width; m++) {
        final double centered = profile[m] - mean[m];
        variance[m] += centered * centered;
      }
    }
    final double[] deviation = new double[width];
    for (int m = 0; m < width; m++) {
      deviation[m] = Math.sqrt(variance[m] / profiles.size());
      checkColumnFinite(deviation[m], metrics.get(m));
    }
    for (final double[] profile : profiles.values()) {
      for (int m = 0; m < width; m++) {
        profile[m] = deviation[m] == 0.0 ? 0.0 : (profile[m] - mean[m]) / deviation[m];
      }
    }
  }

  /**
   * Checks one computed column statistic, a mean or a standard deviation, for
   * finiteness.
   *
   * @param statistic The computed column statistic.
   * @param metric The name of the column the statistic was computed over, for the
   *               failure message.
   * @throws IOException Thrown if the statistic is not finite, which means the column
   *         overflows a double.
   */
  private static void checkColumnFinite(double statistic, String metric)
      throws IOException {
    if (!Double.isFinite(statistic)) {
      throw new IOException("metric column overflows a double: " + metric);
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
   * <p>The score is always a real number. Standardized profiles are finite and bounded,
   * because {@link #load(InputStream)} rejects both non-finite cells and columns that
   * overflow a double, so neither the profiles nor the sums taken over them here can
   * reach an infinity or a {@code NaN}.</p>
   *
   * @param id The first place identifier. Must be listed.
   * @param otherId The second place identifier. Must be listed.
   * @return The cosine similarity, never {@code NaN}, nominally in {@code [-1, 1]}
   *         although rounding can place it up to one ulp outside; {@code 0} when either
   *         profile has no variance at all.
   * @throws IllegalArgumentException Thrown if an identifier is {@code null} or not
   *         listed.
   */
  public double similarity(String id, String otherId) {
    return cosine(profile(id), profile(otherId));
  }

  /**
   * Finds the places most similar to one place.
   *
   * <p>Places with equal scores are ordered by ascending identifier. The tie-break is
   * part of the contract rather than an implementation detail: it decides which places
   * survive the {@code count} cut when more of them tie than there is room for, so the
   * same query against the same table always answers with the same places in the same
   * order.</p>
   *
   * @param id The query place identifier. Must be listed.
   * @param count The maximum number of neighbors. Must be positive.
   * @return The other places ordered by descending similarity and, among equal scores,
   *         by ascending identifier, at most {@code count}; the query place itself is
   *         never included. Never {@code null}.
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
    neighbors.sort(Comparator.comparingDouble(Neighbor::similarity).reversed()
        .thenComparing(Neighbor::id));
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
   * <p>The sums here need no overflow guard of their own, because standardization
   * bounds what can reach them. Over the {@code n} listed places a standardized column
   * sums its squares to exactly {@code n}, so no single standardized value exceeds
   * {@code sqrt(n)} in magnitude, and each of the three sums below is bounded by
   * {@code n} times the number of metrics. Both factors are sizes of in-memory
   * structures, so their product cannot approach the range of a double; a table large
   * enough to overflow these sums could not be held in the first place. This holds only
   * because the caller cannot reach here with an infinite or {@code NaN} profile, which
   * is what the load-time cell and column checks guarantee.</p>
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
