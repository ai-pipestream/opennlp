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

package opennlp.tools.formats.conllu;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ParagraphStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.StringUtil;

/**
 * The CoNLL-U Format is specified
 * <a href="https://universaldependencies.org/format.html">here</a>.
 */
public class ConlluStream implements ObjectStream<ConlluSentence> {

  private static final String TEXT_LANG_PREFIX = "text_";
  private static final String INVALID_ID = "Invalid CoNLL-U id: ";
  private static final String MISSING_MULTIWORD_LINE =
      "Multiword token %s has no word line for id %s";

  private record MultiwordRange(int start, int end) {
  }

  private final ObjectStream<String> sentenceStream;

  /**
   * Initializes a {@link ConlluStream}.
   *
   * @param in The {@link InputStreamFactory} to use. Characters will be interpreted in UTF-8.
   *
   * @throws IOException Thrown if IO errors occurred during initialization.
   * @throws IllegalArgumentException Thrown if {@code in} is {@code null}.
   */
  public ConlluStream(InputStreamFactory in) throws IOException {
    if (in == null) {
      throw new IllegalArgumentException("in must not be null");
    }
    this.sentenceStream = new ParagraphStream(new PlainTextByLineStream(in, StandardCharsets.UTF_8));
  }

  @Override
  public ConlluSentence read() throws IOException {
    String sentence = sentenceStream.read();

    if (sentence != null) {
      List<ConlluWordLine> wordLines = new ArrayList<>();

      BufferedReader reader = new BufferedReader(new StringReader(sentence));

      boolean newDocument = false;
      boolean newParagraph = false;
      String documentId = null;
      String paragraphId = null;
      String sentenceId = null;
      String text = null;
      Map<Locale, String> textLang = null;
      String translit = null;

      String line;
      while ((line = reader.readLine())  != null) {
        // # indicates a comment line and contains additional data
        if (line.trim().startsWith("#")) {
          String commentLine = line.trim().substring(1);

          int separator = commentLine.indexOf('=');

          if (separator != -1) {
            String firstPart = commentLine.substring(0, separator).trim();
            String secondPart = commentLine.substring(separator + 1, commentLine.length()).trim();

            if (!secondPart.isEmpty()) {
              switch (firstPart) {
                case "newdoc id":
                  newDocument = true;
                  documentId = secondPart;
                  break;
                case "newpar id":
                  newParagraph = true;
                  paragraphId = secondPart;
                  break;
                case "sent_id":
                  sentenceId = secondPart;
                  break;
                case "text":
                  text = secondPart;
                  break;
                case "translit":
                  translit = secondPart;
                  break;
              }
            }

            if (firstPart.startsWith(TEXT_LANG_PREFIX)) {
              if (textLang == null) {
                textLang = new HashMap<>();
              }
              addTextLang(firstPart, secondPart, textLang);
            }
          }
          else {
            switch (commentLine.trim()) {
              case "newdoc":
                newDocument = true;
                break;
              case "newpar":
                newParagraph = true;
                break;
            }
          }
        }
        else {
          wordLines.add(new ConlluWordLine(line));
        }
      }

      wordLines = postProcessContractions(wordLines);

      return new ConlluSentence(wordLines, sentenceId, text, newDocument, documentId, newParagraph,
              paragraphId, textLang, translit);
    }

    return null;
  }

  /**
   * Merges the word lines of each multiword token range into the range line and removes them.
   * Stops at the first missing id before allocating further entries in a range. A long
   * cursor permits an inclusive range ending at the largest integer without wrapping.
   *
   * @param lines The word lines of one sentence.
   * @return The lines with each range merged.
   * @throws InvalidFormatException Thrown if a range names a word id that has no line.
   */
  private List<ConlluWordLine> postProcessContractions(List<ConlluWordLine> lines)
      throws InvalidFormatException {

    // 1. Find contractions
    Map<String, Integer> index = new HashMap<>();
    Map<String, MultiwordRange> ranges = new HashMap<>();
    Map<String, List<String>> contractions = new HashMap<>();
    Set<String> linesToDelete = new HashSet<>();

    for (int i = 0; i < lines.size(); i++) {
      ConlluWordLine line = lines.get(i);
      MultiwordRange range = parseId(line.getId());
      if (range != null) {
        ranges.put(line.getId(), range);
      }
      if (index.put(line.getId(), i) != null) {
        throw new InvalidFormatException("Duplicate CoNLL-U id: " + line.getId());
      }
    }
    for (ConlluWordLine line : lines) {
      MultiwordRange range = ranges.get(line.getId());
      if (range != null) {
        List<String> expandedContractions = new ArrayList<>();
        for (long j = range.start(); j <= range.end(); j++) {
          String js = Long.toString(j);
          if (!index.containsKey(js)) {
            throw new InvalidFormatException(
                String.format(MISSING_MULTIWORD_LINE, line.getId(), js));
          }
          if (!linesToDelete.add(js)) {
            throw new InvalidFormatException("Overlapping multiword token range: " + line.getId());
          }
          expandedContractions.add(js);
        }
        contractions.put(line.getId(), expandedContractions);
      }
    }

    // 2. Merge annotation
    for (Entry<String, List<String>> entry : contractions.entrySet()) {
      final String contractionId = entry.getKey();
      final List<String> expandedContractions = entry.getValue();
      int contractionIndex = index.get(contractionId);
      ConlluWordLine contraction = lines.get(contractionIndex);
      List<ConlluWordLine> expandedParts = new ArrayList<>();
      for (String id : expandedContractions) {
        expandedParts.add(lines.get(index.get(id)));
      }
      ConlluWordLine merged = mergeAnnotation(contraction, expandedParts);
      lines.set(contractionIndex, merged);
    }

    // 3. Delete the expanded parts
    lines.removeIf(line -> linesToDelete.contains(line.getId()));
    return lines;
  }

  /**
   * Merges token level annotations.
   *
   * @param contraction The line that receives the annotation.
   * @param expandedParts The lines to get annotation.
   *
   * @return The {@link ConlluWordLine merged line}.
   */
  private ConlluWordLine mergeAnnotation(ConlluWordLine contraction,
                                         List<ConlluWordLine> expandedParts) {
    String id = contraction.getId();
    String form = contraction.getForm();
    String lemma = expandedParts.stream()
        .filter(p -> !"_".equals(p.getLemma()))
        .map(ConlluWordLine::getLemma)
        .collect(Collectors.joining("+"));

    String uPosTag = expandedParts.stream()
        .filter(p -> !"_".equals(p.getPosTag(ConlluTagset.U)))
        .map(p -> p.getPosTag(ConlluTagset.U))
        .collect(Collectors.joining("+"));

    String xPosTag = expandedParts.stream()
        .filter(p -> !"_".equals(p.getPosTag(ConlluTagset.X)))
        .map(p -> p.getPosTag(ConlluTagset.X))
        .collect(Collectors.joining("+"));

    String feats = expandedParts.stream()
        .filter(p -> !"_".equals(p.getFeats()))
        .map(ConlluWordLine::getFeats)
        .collect(Collectors.joining("+"));

    String head = contraction.getHead();
    String deprel = contraction.getDeprel();
    String deps = contraction.getDeps();
    String misc = contraction.getMisc();

    return new ConlluWordLine(id, form, lemma, uPosTag, xPosTag, feats,head, deprel, deps, misc);
  }

  private Map<Locale, String> addTextLang(String firstPart, String secondPart,
                                          Map<Locale, String> textLang) throws InvalidFormatException {
    String lang = firstPart.substring(TEXT_LANG_PREFIX.length());
    if ((lang.length() == 2 || lang.length() == 3) && isAsciiLowerCase(lang)) {
      textLang.put(Locale.of(lang), secondPart);
    }
    else {
      throw new InvalidFormatException("Locale language code is invalid: " + firstPart);
    }
    return textLang;
  }

  /**
   * Validates an integer word id, a nonempty multiword-token range, or an empty-node id.
   *
   * @param id The first field of a CoNLL-U word line.
   * @return The parsed multiword range, or {@code null} for a word or empty node.
   * @throws InvalidFormatException Thrown if the id does not follow the CoNLL-U grammar or an
   *         integer component exceeds the supported 32-bit range.
   */
  private MultiwordRange parseId(String id) throws InvalidFormatException {
    int hyphen = id.indexOf('-');
    int dot = id.indexOf('.');
    if (hyphen >= 0 && dot >= 0) {
      throw new InvalidFormatException(INVALID_ID + id);
    }
    if (hyphen >= 0) {
      int start = parseIdPart(id, 0, hyphen, false);
      int end = parseIdPart(id, hyphen + 1, id.length(), false);
      if (start >= end) {
        throw new InvalidFormatException(INVALID_ID + id);
      }
      return new MultiwordRange(start, end);
    }
    if (dot >= 0) {
      parseIdPart(id, 0, dot, true);
      parseIdPart(id, dot + 1, id.length(), false);
      return null;
    }
    parseIdPart(id, 0, id.length(), false);
    return null;
  }

  /** Reads an unsigned ASCII component, checking overflow before each multiplication. */
  private int parseIdPart(String id, int begin, int end, boolean allowZero)
      throws InvalidFormatException {
    if (begin >= end || end - begin > 1 && id.charAt(begin) == '0') {
      throw new InvalidFormatException(INVALID_ID + id);
    }
    int value = 0;
    for (int i = begin; i < end; i++) {
      char c = id.charAt(i);
      if (!StringUtil.isAsciiDigit(c) || value > (Integer.MAX_VALUE - (c - '0')) / 10) {
        throw new InvalidFormatException(INVALID_ID + id);
      }
      value = value * 10 + c - '0';
    }
    if (!allowZero && value == 0) {
      throw new InvalidFormatException(INVALID_ID + id);
    }
    return value;
  }

  /** Checks the ASCII language-code alphabet. */
  private static boolean isAsciiLowerCase(String value) {
    for (int i = 0; i < value.length(); i++) {
      if (!StringUtil.isAsciiLowerCase(value.charAt(i))) {
        return false;
      }
    }
    return true;
  }

  @Override
  public void close() throws IOException {
    sentenceStream.close();
  }

  @Override
  public void reset() throws IOException, UnsupportedOperationException {
    sentenceStream.reset();
  }
}
