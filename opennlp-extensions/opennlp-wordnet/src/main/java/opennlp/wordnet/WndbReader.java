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
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

import opennlp.tools.wordnet.Synset;
import opennlp.tools.wordnet.WordNetLexicon;
import opennlp.tools.wordnet.WordNetPos;
import opennlp.tools.wordnet.WordNetRelation;

/**
 * Reads a legacy Princeton WNDB database directory ({@code index.noun}, {@code data.noun}, and
 * the corresponding pairs for verbs, adjectives, and adverbs) into a {@link WordNetLexicon}.
 *
 * <p>The reader is clean-room, built from the published {@code wndb(5WN)} and
 * {@code wninput(5WN)} format documentation, with no third-party WordNet library involved. All
 * eight files must be present in the directory. License preamble lines (which begin with a
 * space in the released files) are skipped in index and data files. {@code index.sense} is not
 * read: the v1 contract has no sense-key surface, so sense keys belong to the later sense
 * inventory layer. The {@code *.exc} morphological exception lists in the same directory are
 * the {@link MorphyLemmatizer} companion input and are read separately.</p>
 *
 * <p>Synset ids are minted as {@code wndb-}<i>offset</i>{@code -}<i>pos</i> from the data
 * file's 8-digit byte offset and part-of-speech letter, for example {@code wndb-00001740-n}.
 * The id is opaque to consumers; the format mirrors WN-LMF-style ids only for readability.
 * Adjective satellite lines ({@code ss_type} {@code s}) normalize to
 * {@link WordNetPos#ADJECTIVE}, and the syntactic markers the adjective files append to some
 * words ({@code (p)}, {@code (a)}, {@code (ip)}) are stripped from lemmas. Underscores in
 * stored lemmas become spaces. Sense order per lemma follows the index file's offset order,
 * which the format documents as most frequent first.</p>
 *
 * <p><b>Errors.</b> Malformed content fails loud with an {@link IllegalArgumentException}
 * naming the file and line: a missing file, a truncated or overlong line, a data line whose
 * offset field disagrees with its actual byte position (the format's documented seek
 * contract), an index entry referencing an offset with no data line, an undeclared pointer
 * symbol, or a pointer to a nonexistent target.</p>
 *
 * <p>The returned lexicon is immutable and safe for concurrent lookups.</p>
 */
public final class WndbReader {

  private static final Map<String, WordNetRelation> POINTER_SYMBOLS = pointerSymbols();

  private WndbReader() {
  }

  /**
   * Reads a WNDB database directory.
   *
   * @param directory The directory containing the eight index and data files. Must not be
   *                  {@code null} and must exist.
   * @return The loaded lexicon.
   * @throws IllegalArgumentException Thrown if {@code directory} is {@code null} or not a
   *     directory, a database file is missing, or any file is malformed; the message names the
   *     file and line.
   * @throws UncheckedIOException Thrown if reading a file fails.
   */
  public static WordNetLexicon read(Path directory) {
    if (directory == null) {
      throw new IllegalArgumentException("Directory must not be null");
    }
    if (!Files.isDirectory(directory)) {
      throw new IllegalArgumentException(
          "Directory does not exist or is not a directory: " + directory);
    }
    final Map<String, RawSynset> rawSynsets = new LinkedHashMap<>();
    for (final FilePos filePos : FilePos.values()) {
      parseDataFile(directory, filePos, rawSynsets);
    }
    final Map<String, Synset> synsetsById = resolve(rawSynsets);
    final Map<InMemoryWordNetLexicon.LemmaKey, List<String>> senseOrder = new LinkedHashMap<>();
    for (final FilePos filePos : FilePos.values()) {
      parseIndexFile(directory, filePos, rawSynsets, senseOrder);
    }
    return new InMemoryWordNetLexicon(synsetsById, senseOrder);
  }

  // The four part-of-speech file pairs of a WNDB directory.
  private enum FilePos {
    NOUN("noun", 'n', WordNetPos.NOUN),
    VERB("verb", 'v', WordNetPos.VERB),
    ADJECTIVE("adj", 'a', WordNetPos.ADJECTIVE),
    ADVERB("adv", 'r', WordNetPos.ADVERB);

    private final String suffix;
    private final char posChar;
    private final WordNetPos pos;

    FilePos(String suffix, char posChar, WordNetPos pos) {
      this.suffix = suffix;
      this.posChar = posChar;
      this.pos = pos;
    }
  }

  private static void parseDataFile(Path directory, FilePos filePos,
                                    Map<String, RawSynset> rawSynsets) {
    final String fileName = "data." + filePos.suffix;
    final byte[] bytes = readAll(directory.resolve(fileName), fileName);
    int lineStart = 0;
    int lineNumber = 0;
    while (lineStart < bytes.length) {
      lineNumber++;
      int lineEnd = lineStart;
      while (lineEnd < bytes.length && bytes[lineEnd] != '\n') {
        lineEnd++;
      }
      // ISO-8859-1 decodes bytes one-to-one, keeping offsets exact for any released file.
      final String line =
          new String(bytes, lineStart, lineEnd - lineStart, StandardCharsets.ISO_8859_1);
      if (!line.isEmpty() && line.charAt(0) != ' ') {
        parseDataLine(line, lineStart, fileName, lineNumber, filePos, rawSynsets);
      }
      lineStart = lineEnd + 1;
    }
  }

  private static void parseDataLine(String line, int byteOffset, String fileName, int lineNumber,
                                    FilePos filePos, Map<String, RawSynset> rawSynsets) {
    final Tokenizer tokens = new Tokenizer(line, fileName, lineNumber);
    final String offsetField = tokens.next("synset_offset");
    if (parseOffset(offsetField, tokens) != byteOffset) {
      throw malformed(fileName, lineNumber, "Synset offset field " + offsetField
          + " disagrees with the actual byte position " + byteOffset);
    }
    tokens.next("lex_filenum");
    final String ssType = tokens.next("ss_type");
    final boolean validType = switch (filePos) {
      case ADJECTIVE -> "a".equals(ssType) || "s".equals(ssType);
      default -> ssType.length() == 1 && ssType.charAt(0) == filePos.posChar;
    };
    if (!validType) {
      throw malformed(fileName, lineNumber,
          "Synset type " + ssType + " does not belong in " + fileName);
    }
    final int wordCount = tokens.nextInt("w_cnt", 16);
    if (wordCount < 1) {
      throw malformed(fileName, lineNumber, "Word count must be at least 1, got: " + wordCount);
    }
    final List<String> lemmas = new ArrayList<>(wordCount);
    for (int i = 0; i < wordCount; i++) {
      final String lemma = cleanLemma(tokens.next("word"), fileName, lineNumber);
      tokens.nextInt("lex_id", 16);
      if (!lemmas.contains(lemma)) {
        lemmas.add(lemma);
      }
    }
    final int pointerCount = tokens.nextInt("p_cnt", 10);
    final List<RawPointer> pointers = new ArrayList<>(pointerCount);
    for (int i = 0; i < pointerCount; i++) {
      final String symbol = tokens.next("pointer_symbol");
      final WordNetRelation relation = POINTER_SYMBOLS.get(symbol);
      if (relation == null) {
        throw malformed(fileName, lineNumber, "Undeclared pointer symbol: " + symbol);
      }
      final String targetOffset = tokens.next("pointer synset_offset");
      parseOffset(targetOffset, tokens);
      final char targetPos = posChar(tokens.next("pointer pos"), tokens);
      tokens.next("pointer source/target");
      pointers.add(new RawPointer(relation, "wndb-" + targetOffset + "-" + targetPos, lineNumber));
    }
    if (filePos == FilePos.VERB) {
      final int frameCount = tokens.nextInt("f_cnt", 10);
      for (int i = 0; i < frameCount; i++) {
        tokens.next("frame marker");
        tokens.next("f_num");
        tokens.next("w_num");
      }
    }
    final String gloss = tokens.gloss();
    final String id = "wndb-" + offsetField + "-" + filePos.posChar;
    rawSynsets.put(id, new RawSynset(id, filePos.pos, lemmas, gloss, pointers,
        fileName, lineNumber));
  }

  private static Map<String, Synset> resolve(Map<String, RawSynset> rawSynsets) {
    final Map<String, Synset> synsetsById = new LinkedHashMap<>(rawSynsets.size() * 2);
    for (final RawSynset raw : rawSynsets.values()) {
      final Map<WordNetRelation, LinkedHashSet<String>> typed = new LinkedHashMap<>();
      for (final RawPointer pointer : raw.pointers) {
        if (!rawSynsets.containsKey(pointer.targetId)) {
          throw malformed(raw.fileName, raw.lineNumber, "Synset " + raw.id + " has a "
              + pointer.relation + " pointer to nonexistent synset " + pointer.targetId);
        }
        typed.computeIfAbsent(pointer.relation, unused -> new LinkedHashSet<>())
            .add(pointer.targetId);
      }
      final Map<WordNetRelation, List<String>> relations = new LinkedHashMap<>(typed.size() * 2);
      for (final Map.Entry<WordNetRelation, LinkedHashSet<String>> entry : typed.entrySet()) {
        relations.put(entry.getKey(), List.copyOf(entry.getValue()));
      }
      synsetsById.put(raw.id, new Synset(raw.id, raw.pos, raw.lemmas, raw.gloss, relations));
    }
    return synsetsById;
  }

  private static void parseIndexFile(Path directory, FilePos filePos,
                                     Map<String, RawSynset> rawSynsets,
                                     Map<InMemoryWordNetLexicon.LemmaKey, List<String>> senses) {
    final String fileName = "index." + filePos.suffix;
    final byte[] bytes = readAll(directory.resolve(fileName), fileName);
    final String content = new String(bytes, StandardCharsets.ISO_8859_1);
    int lineNumber = 0;
    int lineStart = 0;
    while (lineStart < content.length()) {
      lineNumber++;
      int lineEnd = content.indexOf('\n', lineStart);
      if (lineEnd < 0) {
        lineEnd = content.length();
      }
      final String line = content.substring(lineStart, lineEnd);
      if (!line.isEmpty() && line.charAt(0) != ' ') {
        parseIndexLine(line, fileName, lineNumber, filePos, rawSynsets, senses);
      }
      lineStart = lineEnd + 1;
    }
  }

  private static void parseIndexLine(String line, String fileName, int lineNumber,
                                     FilePos filePos, Map<String, RawSynset> rawSynsets,
                                     Map<InMemoryWordNetLexicon.LemmaKey, List<String>> senses) {
    final Tokenizer tokens = new Tokenizer(line, fileName, lineNumber);
    final String lemma = tokens.next("lemma");
    final String pos = tokens.next("pos");
    if (pos.length() != 1 || pos.charAt(0) != filePos.posChar) {
      throw malformed(fileName, lineNumber, "Index pos " + pos + " does not belong in "
          + fileName);
    }
    final int synsetCount = tokens.nextInt("synset_cnt", 10);
    if (synsetCount < 1) {
      throw malformed(fileName, lineNumber,
          "Synset count must be at least 1, got: " + synsetCount);
    }
    final int pointerTypeCount = tokens.nextInt("p_cnt", 10);
    for (int i = 0; i < pointerTypeCount; i++) {
      // The index file's pointer-type summary is informational; the typed pointers in the
      // data file are authoritative, so the summary symbols are not validated here.
      tokens.next("ptr_symbol");
    }
    tokens.next("sense_cnt");
    tokens.next("tagsense_cnt");
    final List<String> order = new ArrayList<>(synsetCount);
    for (int i = 0; i < synsetCount; i++) {
      final String offset = tokens.next("synset_offset");
      parseOffset(offset, tokens);
      final String synsetId = "wndb-" + offset + "-" + filePos.posChar;
      if (!rawSynsets.containsKey(synsetId)) {
        throw malformed(fileName, lineNumber, "Lemma " + lemma + " references offset " + offset
            + " with no data." + filePos.suffix + " line");
      }
      if (!order.contains(synsetId)) {
        order.add(synsetId);
      }
    }
    final InMemoryWordNetLexicon.LemmaKey key =
        InMemoryWordNetLexicon.LemmaKey.of(lemma, filePos.pos);
    final List<String> existing = senses.get(key);
    if (existing == null) {
      senses.put(key, order);
    } else {
      // Two index lemmas can fold to one key; keep first-listed order and append the rest.
      for (final String synsetId : order) {
        if (!existing.contains(synsetId)) {
          existing.add(synsetId);
        }
      }
    }
  }

  // Strips the documented adjective syntactic markers and turns underscores into spaces.
  private static String cleanLemma(String word, String fileName, int lineNumber) {
    String cleaned = word;
    if (cleaned.endsWith(")")) {
      final int open = cleaned.lastIndexOf('(');
      final String marker = open < 0 ? "" : cleaned.substring(open);
      if (!"(p)".equals(marker) && !"(a)".equals(marker) && !"(ip)".equals(marker)) {
        throw malformed(fileName, lineNumber, "Unknown syntactic marker on word: " + word);
      }
      cleaned = cleaned.substring(0, open);
    }
    if (cleaned.isEmpty()) {
      throw malformed(fileName, lineNumber, "Empty word field");
    }
    return cleaned.replace('_', ' ');
  }

  private static int parseOffset(String offset, Tokenizer tokens) {
    if (offset.length() != 8) {
      throw tokens.malformedToken("Synset offset must be 8 digits, got: " + offset);
    }
    int value = 0;
    for (int i = 0; i < 8; i++) {
      final char c = offset.charAt(i);
      if (c < '0' || c > '9') {
        throw tokens.malformedToken("Synset offset must be 8 digits, got: " + offset);
      }
      value = value * 10 + (c - '0');
    }
    return value;
  }

  private static char posChar(String pos, Tokenizer tokens) {
    if (pos.length() == 1) {
      final char c = pos.charAt(0);
      if (c == 'n' || c == 'v' || c == 'a' || c == 'r') {
        return c;
      }
    }
    throw tokens.malformedToken("Pointer pos must be one of n, v, a, r, got: " + pos);
  }

  private static byte[] readAll(Path file, String fileName) {
    if (!Files.isRegularFile(file)) {
      throw new IllegalArgumentException("Missing WNDB database file: " + file);
    }
    try {
      return Files.readAllBytes(file);
    } catch (IOException e) {
      throw new UncheckedIOException("Unable to read WNDB database file " + file, e);
    }
  }

  private static IllegalArgumentException malformed(String fileName, int lineNumber,
                                                    String message) {
    return new IllegalArgumentException(
        "Malformed WNDB file " + fileName + " at line " + lineNumber + ": " + message);
  }

  // A cursor over one line's space-separated fields; no regular expressions involved.
  private static final class Tokenizer {

    private final String line;
    private final String fileName;
    private final int lineNumber;
    private int position;

    Tokenizer(String line, String fileName, int lineNumber) {
      this.line = line;
      this.fileName = fileName;
      this.lineNumber = lineNumber;
    }

    String next(String field) {
      while (position < line.length() && line.charAt(position) == ' ') {
        position++;
      }
      if (position >= line.length()) {
        throw malformed(fileName, lineNumber, "Truncated line, missing field: " + field);
      }
      final int start = position;
      while (position < line.length() && line.charAt(position) != ' ') {
        position++;
      }
      return line.substring(start, position);
    }

    int nextInt(String field, int radix) {
      final String token = next(field);
      try {
        return Integer.parseInt(token, radix);
      } catch (NumberFormatException e) {
        throw new IllegalArgumentException(malformed(fileName, lineNumber,
            "Field " + field + " is not a base-" + radix + " integer: " + token).getMessage(), e);
      }
    }

    // The remainder after the pipe separator, trimmed of the surrounding spaces.
    String gloss() {
      final String separator = next("gloss separator");
      if (!"|".equals(separator)) {
        throw malformed(fileName, lineNumber, "Expected the | gloss separator, got: " + separator);
      }
      int start = position;
      while (start < line.length() && line.charAt(start) == ' ') {
        start++;
      }
      int end = line.length();
      while (end > start && line.charAt(end - 1) == ' ') {
        end--;
      }
      return line.substring(start, end);
    }

    IllegalArgumentException malformedToken(String message) {
      return malformed(fileName, lineNumber, message);
    }
  }

  private record RawPointer(WordNetRelation relation, String targetId, int lineNumber) {
  }

  private static final class RawSynset {
    private final String id;
    private final WordNetPos pos;
    private final List<String> lemmas;
    private final String gloss;
    private final List<RawPointer> pointers;
    private final String fileName;
    private final int lineNumber;

    RawSynset(String id, WordNetPos pos, List<String> lemmas, String gloss,
              List<RawPointer> pointers, String fileName, int lineNumber) {
      this.id = id;
      this.pos = pos;
      this.lemmas = lemmas;
      this.gloss = gloss;
      this.pointers = pointers;
      this.fileName = fileName;
      this.lineNumber = lineNumber;
    }
  }

  private static Map<String, WordNetRelation> pointerSymbols() {
    final Map<String, WordNetRelation> symbols = new HashMap<>();
    symbols.put("!", WordNetRelation.ANTONYM);
    symbols.put("@", WordNetRelation.HYPERNYM);
    symbols.put("@i", WordNetRelation.INSTANCE_HYPERNYM);
    symbols.put("~", WordNetRelation.HYPONYM);
    symbols.put("~i", WordNetRelation.INSTANCE_HYPONYM);
    symbols.put("#m", WordNetRelation.MEMBER_HOLONYM);
    symbols.put("#s", WordNetRelation.SUBSTANCE_HOLONYM);
    symbols.put("#p", WordNetRelation.PART_HOLONYM);
    symbols.put("%m", WordNetRelation.MEMBER_MERONYM);
    symbols.put("%s", WordNetRelation.SUBSTANCE_MERONYM);
    symbols.put("%p", WordNetRelation.PART_MERONYM);
    symbols.put("=", WordNetRelation.ATTRIBUTE);
    symbols.put("+", WordNetRelation.DERIVATIONALLY_RELATED);
    symbols.put("*", WordNetRelation.ENTAILMENT);
    symbols.put(">", WordNetRelation.CAUSE);
    symbols.put("^", WordNetRelation.ALSO_SEE);
    symbols.put("$", WordNetRelation.VERB_GROUP);
    symbols.put("&", WordNetRelation.SIMILAR_TO);
    symbols.put("<", WordNetRelation.PARTICIPLE);
    symbols.put("\\", WordNetRelation.PERTAINYM);
    symbols.put(";c", WordNetRelation.DOMAIN_TOPIC);
    symbols.put("-c", WordNetRelation.MEMBER_OF_DOMAIN_TOPIC);
    symbols.put(";r", WordNetRelation.DOMAIN_REGION);
    symbols.put("-r", WordNetRelation.MEMBER_OF_DOMAIN_REGION);
    symbols.put(";u", WordNetRelation.DOMAIN_USAGE);
    symbols.put("-u", WordNetRelation.MEMBER_OF_DOMAIN_USAGE);
    return Map.copyOf(symbols);
  }
}
