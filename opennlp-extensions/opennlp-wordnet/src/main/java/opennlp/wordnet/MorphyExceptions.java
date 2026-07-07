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
package opennlp.wordnet;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import opennlp.tools.commons.ThreadSafe;
import opennlp.tools.wordnet.WordNetPos;

/**
 * The Morphy exception lists: the per-part-of-speech tables of irregular inflected forms
 * ({@code mice} to {@code mouse}, {@code went} to {@code go}) that the Morphy algorithm
 * consults before its detachment rules.
 *
 * <p>{@link #load(Path)} reads the four {@code *.exc} files ({@code noun.exc},
 * {@code verb.exc}, {@code adj.exc}, {@code adv.exc}) in the documented WNDB format: one entry
 * per line, the inflected form followed by one or more base forms, space separated, with
 * underscores standing for spaces in multiword entries. All four files must be present; note
 * that Princeton's database-only download ({@code WNdb}) does not include them, while the full
 * WordNet package does. No exception data is bundled with this module; see
 * {@link MorphyLemmatizer} for the tiering rationale.</p>
 *
 * <p>Lookups fold the queried word the same way the lexicon seam folds lemmas: lowercase with
 * the root locale, underscore as space.</p>
 *
 * <p>Instances are immutable after loading and safe for concurrent lookups.</p>
 */
@ThreadSafe
public final class MorphyExceptions {

  private final Map<WordNetPos, Map<String, List<String>>> byPos;

  private MorphyExceptions(Map<WordNetPos, Map<String, List<String>>> byPos) {
    this.byPos = byPos;
  }

  /**
   * Loads the four exception lists from a directory.
   *
   * @param directory The directory containing {@code noun.exc}, {@code verb.exc},
   *                  {@code adj.exc}, and {@code adv.exc}. Must not be {@code null} and must
   *                  exist.
   * @return The loaded exception lists.
   * @throws IllegalArgumentException Thrown if {@code directory} is {@code null} or not a
   *     directory, one of the four files is missing, or a line is malformed; the message names
   *     the file and line.
   * @throws UncheckedIOException Thrown if reading a file fails.
   */
  public static MorphyExceptions load(Path directory) {
    if (directory == null) {
      throw new IllegalArgumentException("Directory must not be null");
    }
    if (!Files.isDirectory(directory)) {
      throw new IllegalArgumentException(
          "Directory does not exist or is not a directory: " + directory);
    }
    final Map<WordNetPos, Map<String, List<String>>> byPos = new EnumMap<>(WordNetPos.class);
    byPos.put(WordNetPos.NOUN, loadFile(directory, "noun.exc"));
    byPos.put(WordNetPos.VERB, loadFile(directory, "verb.exc"));
    byPos.put(WordNetPos.ADJECTIVE, loadFile(directory, "adj.exc"));
    byPos.put(WordNetPos.ADVERB, loadFile(directory, "adv.exc"));
    return new MorphyExceptions(byPos);
  }

  /**
   * Finds the base forms of an irregular inflected form.
   *
   * @param word The inflected form; folded before lookup. Must not be {@code null}.
   * @param pos  The part of speech. Must not be {@code null}.
   * @return The base forms in file order, never {@code null}; empty when the word has no entry.
   * @throws IllegalArgumentException Thrown if {@code word} or {@code pos} is {@code null}.
   */
  public List<String> lookup(String word, WordNetPos pos) {
    if (word == null) {
      throw new IllegalArgumentException("Word must not be null");
    }
    if (pos == null) {
      throw new IllegalArgumentException("Pos must not be null");
    }
    final List<String> lemmas = byPos.get(pos).get(fold(word));
    return lemmas == null ? List.of() : lemmas;
  }

  static String fold(String word) {
    return word.replace('_', ' ').toLowerCase(Locale.ROOT);
  }

  private static Map<String, List<String>> loadFile(Path directory, String fileName) {
    final Path file = directory.resolve(fileName);
    if (!Files.isRegularFile(file)) {
      throw new IllegalArgumentException("Missing exception list file: " + file);
    }
    final List<String> lines;
    try {
      lines = Files.readAllLines(file, StandardCharsets.ISO_8859_1);
    } catch (IOException e) {
      throw new UncheckedIOException("Unable to read exception list file " + file, e);
    }
    final Map<String, List<String>> entries = new HashMap<>(lines.size() * 2);
    for (int i = 0; i < lines.size(); i++) {
      final String line = lines.get(i);
      if (line.isEmpty()) {
        continue;
      }
      final List<String> fields = splitOnSpaces(line);
      if (fields.size() < 2) {
        throw new IllegalArgumentException("Malformed exception list " + fileName + " at line "
            + (i + 1) + ": expected an inflected form and at least one base form, got: " + line);
      }
      final List<String> lemmas = new ArrayList<>(fields.size() - 1);
      for (final String lemma : fields.subList(1, fields.size())) {
        lemmas.add(fold(lemma));
      }
      // A form listed twice keeps its first entry, matching first-match lookup semantics.
      entries.putIfAbsent(fold(fields.get(0)), List.copyOf(lemmas));
    }
    return Map.copyOf(entries);
  }

  private static List<String> splitOnSpaces(String line) {
    final List<String> fields = new ArrayList<>(3);
    int start = 0;
    while (start < line.length()) {
      final int space = line.indexOf(' ', start);
      if (space < 0) {
        fields.add(line.substring(start));
        break;
      }
      if (space > start) {
        fields.add(line.substring(start, space));
      }
      start = space + 1;
    }
    return fields;
  }
}
