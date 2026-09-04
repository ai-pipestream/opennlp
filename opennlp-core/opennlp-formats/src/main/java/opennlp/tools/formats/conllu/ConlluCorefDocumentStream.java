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
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import opennlp.tools.coref.CorefAnnotator;
import opennlp.tools.coref.CorefMention;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.Layers;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.Span;

/**
 * Reads documents with gold coreference chains from CoNLL-U with the {@code Entity}
 * attribute of the MISC column, the encoding of the Universal Anaphora and OntoGUM
 * releases: {@code (3} opens a mention of entity 3 on the token, {@code 3)} closes the
 * innermost open mention of that entity, and {@code (3)} does both. A rich id such as
 * {@code 3-person-new-...} counts by the part before its first hyphen, and each part of a
 * discontinuous mention such as {@code (e5[1/2]-person} is read as a mention of {@code e5}.
 *
 * <p>Each {@code # newdoc} comment starts a document; a file without one is a single
 * document. The document text is rebuilt from the word forms, separated by single
 * spaces except after a token marked {@code SpaceAfter=No}, with sentences separated by
 * newlines. Every document carries {@link Layers#SENTENCES}, {@link Layers#TOKENS},
 * {@link Layers#POS_TAGS} from the chosen tag set, {@link CorefAnnotator#GOLD_CHAINS}
 * with one chain per entity id in order of first mention, and, when {@code # speaker}
 * comments are present, {@link CorefAnnotator#SPEAKERS} over each such sentence.
 * Multiword token ranges and empty nodes are skipped.</p>
 *
 * @since 3.0.0
 */
public class ConlluCorefDocumentStream implements ObjectStream<Document> {

  private static final String NEWDOC = "# newdoc";
  private static final String SPEAKER = "# speaker =";
  private static final String ENTITY = "Entity";
  private static final String SPACE_AFTER_NO = "SpaceAfter=No";

  private final InputStreamFactory in;
  private final ConlluTagset tagset;
  private BufferedReader reader;
  private String pendingLine;

  /**
   * Initializes the stream.
   *
   * @param in The CoNLL-U input. Must not be {@code null}.
   * @param tagset Which tag column fills the POS layer. Must not be {@code null}.
   * @throws IOException Thrown if the input cannot be opened.
   * @throws IllegalArgumentException Thrown if an argument is {@code null}.
   */
  public ConlluCorefDocumentStream(InputStreamFactory in, ConlluTagset tagset)
      throws IOException {
    if (in == null) {
      throw new IllegalArgumentException("in must not be null");
    }
    if (tagset == null) {
      throw new IllegalArgumentException("tagset must not be null");
    }
    this.in = in;
    this.tagset = tagset;
    reset();
  }

  /** {@inheritDoc} Returns {@code null} once the input holds no further document. */
  @Override
  public Document read() throws IOException {
    final List<String> lines = new ArrayList<>();
    String line = pendingLine != null ? pendingLine : reader.readLine();
    pendingLine = null;
    while (line != null) {
      if (line.startsWith(NEWDOC) && !lines.isEmpty()) {
        pendingLine = line;
        break;
      }
      lines.add(line);
      line = reader.readLine();
    }
    boolean content = false;
    for (final String l : lines) {
      if (!l.isEmpty() && l.charAt(0) != '#') {
        content = true;
        break;
      }
    }
    return content ? parse(lines) : null;
  }

  /** {@inheritDoc} */
  @Override
  public void reset() throws IOException {
    close();
    reader = new BufferedReader(new InputStreamReader(in.createInputStream(),
        StandardCharsets.UTF_8));
    pendingLine = null;
  }

  /** {@inheritDoc} */
  @Override
  public void close() throws IOException {
    if (reader != null) {
      reader.close();
      reader = null;
    }
  }

  /** Builds one document from its lines. */
  private Document parse(List<String> lines) throws InvalidFormatException {
    final StringBuilder text = new StringBuilder();
    final List<Annotation<String>> sentences = new ArrayList<>();
    final List<Annotation<String>> tokens = new ArrayList<>();
    final List<Annotation<String>> tags = new ArrayList<>();
    final List<Annotation<String>> speakers = new ArrayList<>();
    final Map<String, List<Integer>> openStarts = new HashMap<>();
    final Map<String, List<Span>> entities = new LinkedHashMap<>();
    String speaker = null;
    int sentenceStart = -1;
    boolean glue = false;
    for (final String line : lines) {
      if (line.isEmpty()) {
        if (sentenceStart >= 0) {
          sentences.add(new Annotation<>(new Span(sentenceStart, text.length()),
              text.substring(sentenceStart)));
          if (speaker != null) {
            speakers.add(new Annotation<>(new Span(sentenceStart, text.length()), speaker));
          }
          sentenceStart = -1;
          speaker = null;
        }
        continue;
      }
      if (line.charAt(0) == '#') {
        if (line.startsWith(SPEAKER)) {
          speaker = line.substring(SPEAKER.length()).trim();
        }
        continue;
      }
      final String[] fields = fields(line);
      if (fields.length < 10) {
        throw new InvalidFormatException("expected 10 columns: " + line);
      }
      if (!plainId(fields[0])) {
        continue;
      }
      if (sentenceStart < 0) {
        if (text.length() > 0) {
          text.append('\n');
        }
        sentenceStart = text.length();
        glue = true;
      }
      if (!glue) {
        text.append(' ');
      }
      final int start = text.length();
      text.append(fields[1]);
      final int end = text.length();
      tokens.add(new Annotation<>(new Span(start, end), fields[1]));
      tags.add(new Annotation<>(new Span(start, end),
          tagset == ConlluTagset.U ? fields[3] : fields[4]));
      glue = fields[9].contains(SPACE_AFTER_NO);
      final String entity = misc(fields[9], ENTITY);
      if (entity != null) {
        brackets(entity, start, end, openStarts, entities);
      }
    }
    if (sentenceStart >= 0) {
      sentences.add(new Annotation<>(new Span(sentenceStart, text.length()),
          text.substring(sentenceStart)));
      if (speaker != null) {
        speakers.add(new Annotation<>(new Span(sentenceStart, text.length()), speaker));
      }
    }
    for (final Map.Entry<String, List<Integer>> open : openStarts.entrySet()) {
      if (!open.getValue().isEmpty()) {
        throw new InvalidFormatException("entity " + open.getKey() + " is never closed");
      }
    }
    Document document = Document.of(text.toString())
        .with(Layers.SENTENCES, sentences)
        .with(Layers.TOKENS, tokens)
        .with(Layers.POS_TAGS, tags)
        .with(CorefAnnotator.GOLD_CHAINS, chains(entities));
    if (!speakers.isEmpty()) {
      document = document.with(CorefAnnotator.SPEAKERS, speakers);
    }
    return document;
  }

  /** Numbers the entities in order of first mention and lists their mentions in text order. */
  private List<Annotation<CorefMention>> chains(Map<String, List<Span>> entities) {
    final List<Annotation<CorefMention>> layer = new ArrayList<>();
    int chain = 0;
    for (final List<Span> mentions : entities.values()) {
      for (final Span mention : mentions) {
        layer.add(new Annotation<>(mention,
            new CorefMention(chain, CorefMention.KIND_GOLD, CorefMention.NO_ENTITY)));
      }
      chain++;
    }
    layer.sort((a, b) -> a.span().getStart() != b.span().getStart()
        ? Integer.compare(a.span().getStart(), b.span().getStart())
        : Integer.compare(b.span().getEnd(), a.span().getEnd()));
    return layer;
  }

  /** Splits a line on tabs. */
  private String[] fields(String line) {
    final List<String> fields = new ArrayList<>(10);
    int start = 0;
    for (int i = 0; i <= line.length(); i++) {
      if (i == line.length() || line.charAt(i) == '\t') {
        fields.add(line.substring(start, i));
        start = i + 1;
      }
    }
    return fields.toArray(new String[0]);
  }

  /** Accepts a word id, rejecting multiword ranges and empty nodes. */
  private boolean plainId(String id) {
    if (id.isEmpty()) {
      return false;
    }
    for (int i = 0; i < id.length(); i++) {
      if (id.charAt(i) < '0' || id.charAt(i) > '9') {
        return false;
      }
    }
    return true;
  }

  /** Reads one {@code key=value} attribute of a MISC column, or {@code null}. */
  private String misc(String misc, String key) {
    int start = 0;
    while (start < misc.length()) {
      int end = misc.indexOf('|', start);
      if (end < 0) {
        end = misc.length();
      }
      final String attribute = misc.substring(start, end);
      if (attribute.startsWith(key) && attribute.length() > key.length()
          && attribute.charAt(key.length()) == '=') {
        return attribute.substring(key.length() + 1);
      }
      start = end + 1;
    }
    return null;
  }

  /** Applies the bracket notation of one token to the open mentions and entities. */
  private void brackets(String entity, int start, int end,
      Map<String, List<Integer>> openStarts, Map<String, List<Span>> entities)
      throws InvalidFormatException {
    int i = 0;
    while (i < entity.length()) {
      if (entity.charAt(i) == '(') {
        int j = i + 1;
        while (j < entity.length() && entity.charAt(j) != '(' && entity.charAt(j) != ')') {
          j++;
        }
        final String id = canonicalId(entity.substring(i + 1, j));
        if (j < entity.length() && entity.charAt(j) == ')') {
          entities.computeIfAbsent(id, key -> new ArrayList<>()).add(new Span(start, end));
          j++;
        } else {
          openStarts.computeIfAbsent(id, key -> new ArrayList<>()).add(start);
        }
        i = j;
      } else {
        int j = i;
        while (j < entity.length() && entity.charAt(j) != ')') {
          j++;
        }
        final String id = canonicalId(entity.substring(i, j));
        final List<Integer> starts = openStarts.get(id);
        if (starts == null || starts.isEmpty()) {
          throw new InvalidFormatException("entity " + id + " is closed but not open");
        }
        entities.computeIfAbsent(id, key -> new ArrayList<>())
            .add(new Span(starts.remove(starts.size() - 1), end));
        i = j + 1;
      }
    }
  }

  /**
   * Strips the attribute tail of a rich entity id and the part marker of a
   * discontinuous mention, so {@code e5[1/2]-person} and {@code e5[2/2]} are entity
   * {@code e5} and each part becomes a mention of it.
   */
  private String canonicalId(String id) {
    int end = id.length();
    for (int i = 0; i < id.length(); i++) {
      if (id.charAt(i) == '-' || id.charAt(i) == '[') {
        end = i;
        break;
      }
    }
    return id.substring(0, end);
  }
}
