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

package opennlp.tools.coref;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import opennlp.tools.chunker.ChunkerAnnotator;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;
import opennlp.tools.coref.CorefScores.Score;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.Layers;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.util.Span;

/**
 * Scores the coreference annotator on OntoGUM, the OntoNotes-scheme conversion of the
 * Georgetown University Multilayer corpus, with the CoNLL metrics of {@link CorefScorer}.
 *
 * <p>Runs only when {@code opennlp.coref.gum.dir} names a GUM checkout: its
 * {@code splits.md} selects the documents and {@code coref/ontogum/conllu} supplies
 * gold sentences, tokens, Penn tags, and coreference chains. The data is fetched by the
 * runner and never enters the repository; the annotations are CC-BY 4.0 and the texts
 * carry their own licenses, so check both before any use beyond measurement. Named
 * entities are predicted by the {@code en-ner-person}, {@code en-ner-location}, and
 * {@code en-ner-organization} models in {@code opennlp.coref.models.dir}, which
 * defaults to {@code ~/.opennlp}; noun phrase mentions come from the chunker model
 * {@code opennlp.coref.chunker} names, by default {@code en-chunker.bin} in the same
 * directory, and are absent when neither exists. {@code opennlp.coref.split} chooses
 * {@code dev} (the default), {@code test}, {@code test2}, or {@code train}.</p>
 *
 * <p>Sentences, tokens, and tags are gold, so the figures are not comparable one to
 * one with the fully predicted CoreNLP input of Zhu, Pradhan, and Zeldes (ACL 2021),
 * who report a CoNLL average of 39.7 for the deterministic Stanford system on the 2021
 * OntoGUM test set. Assertions are low regression floors; the logged scores and the
 * rows appended to {@code target/coref-eval-results.csv} are the measurement.</p>
 */
public class OntoGumCorefEvalTest {

  private static final Logger LOG = LoggerFactory.getLogger(OntoGumCorefEvalTest.class);

  private static final String GUM_DIR_PROPERTY = "opennlp.coref.gum.dir";
  private static final String MODELS_DIR_PROPERTY = "opennlp.coref.models.dir";
  private static final String SPLIT_PROPERTY = "opennlp.coref.split";
  private static final String CHUNKER_PROPERTY = "opennlp.coref.chunker";
  private static final Path RESULTS_FILE = Path.of("target", "coref-eval-results.csv");
  private static final String[] NER_MODELS =
      {"en-ner-person.bin", "en-ner-location.bin", "en-ner-organization.bin"};

  /** A mention identity: the character span it covers. */
  private record Mention(int start, int end) {
  }

  /** One gold-annotated document with its coreference key. */
  private record GoldDocument(String name, Document document, List<Set<Mention>> key) {
  }

  /** A read-only view over the per-split and per-genre scorers. */
  private record Result(String split, int documents, CorefScores scores,
      double corefSeconds) {
  }

  @Test
  @EnabledIfSystemProperty(named = GUM_DIR_PROPERTY, matches = ".+")
  void testScoresOntoGumSplit() throws IOException {
    final Path gum = Path.of(System.getProperty(GUM_DIR_PROPERTY));
    final Path models = Path.of(System.getProperty(MODELS_DIR_PROPERTY,
        System.getProperty("user.home") + "/.opennlp"));
    final String split = System.getProperty(SPLIT_PROPERTY, "dev");
    final List<String> names = splitDocuments(gum.resolve("splits.md"), split);
    Assertions.assertFalse(names.isEmpty(), "split " + split + " lists no documents");

    final List<NameFinderME> finders = new ArrayList<>();
    for (final String model : NER_MODELS) {
      finders.add(new NameFinderME(new TokenNameFinderModel(models.resolve(model))));
    }
    final Path chunkerModel = Path.of(System.getProperty(CHUNKER_PROPERTY,
        models.resolve("en-chunker.bin").toString()));
    final ChunkerAnnotator chunker = Files.exists(chunkerModel)
        ? new ChunkerAnnotator(new ChunkerME(new ChunkerModel(chunkerModel))) : null;
    final CorefAnnotator annotator = new CorefAnnotator();
    final CorefScorer all = new CorefScorer();
    final Map<String, CorefScorer> byGenre = new TreeMap<>();
    int documents = 0;
    long corefNanos = 0;
    for (final String name : names) {
      final Path file = gum.resolve("coref/ontogum/conllu").resolve(name + ".conllu");
      if (!Files.exists(file)) {
        LOG.info("skipping {}: not in this checkout", name);
        continue;
      }
      final GoldDocument gold = read(name, Files.readAllLines(file, StandardCharsets.UTF_8));
      final Document tagged = withEntities(gold.document(), finders);
      final Document input = chunker == null ? tagged : chunker.annotate(tagged);
      final long started = System.nanoTime();
      final Document output = annotator.annotate(input);
      corefNanos += System.nanoTime() - started;
      final List<Set<Mention>> response = responsePartition(output);
      all.add(gold.key(), response);
      byGenre.computeIfAbsent(genre(name), key -> new CorefScorer())
          .add(gold.key(), response);
      documents++;
    }
    Assertions.assertTrue(documents > 0, "no OntoGUM documents found for split " + split);

    final Result result = new Result(split, documents, all.scores(), corefNanos / 1e9);
    LOG.info("OntoGUM {} ({} documents): {}", split, documents, describe(result.scores()));
    LOG.info("coreference pass: {} documents/second",
        String.format(Locale.ROOT, "%.0f", documents / Math.max(result.corefSeconds(), 1e-9)));
    for (final Map.Entry<String, CorefScorer> genre : byGenre.entrySet()) {
      LOG.info("  {}: {}", genre.getKey(), describe(genre.getValue().scores()));
    }
    record(result);
    Assertions.assertTrue(result.scores().conll() > 0.05,
        "CoNLL average collapsed: " + result.scores().conll());
  }

  /** Formats the metric triples on one line. */
  private static String describe(CorefScores scores) {
    return String.format(Locale.ROOT,
        "CoNLL %.1f | MUC %s | B3 %s | CEAFe %s | mentions %s",
        100 * scores.conll(), triple(scores.muc()), triple(scores.bCubed()),
        triple(scores.ceafE()), triple(scores.mentions()));
  }

  private static String triple(Score score) {
    return String.format(Locale.ROOT, "P %.1f R %.1f F %.1f",
        100 * score.precision(), 100 * score.recall(), 100 * score.f1());
  }

  /** Appends one CSV row per run so sweeps accumulate in the build directory. */
  private static void record(Result result) throws IOException {
    Files.createDirectories(RESULTS_FILE.getParent());
    if (!Files.exists(RESULTS_FILE)) {
      Files.writeString(RESULTS_FILE, "time,split,documents,conll,muc_p,muc_r,muc_f,"
          + "b3_p,b3_r,b3_f,ceafe_p,ceafe_r,ceafe_f,mention_p,mention_r,mention_f,"
          + "coref_seconds\n");
    }
    final CorefScores s = result.scores();
    Files.writeString(RESULTS_FILE, String.format(Locale.ROOT,
        "%s,%s,%d,%.4f,%s,%s,%s,%s,%.3f%n", Instant.now(), result.split(),
        result.documents(), s.conll(), csv(s.muc()), csv(s.bCubed()), csv(s.ceafE()),
        csv(s.mentions()), result.corefSeconds()),
        java.nio.file.StandardOpenOption.APPEND);
  }

  private static String csv(Score score) {
    return String.format(Locale.ROOT, "%.4f,%.4f,%.4f",
        score.precision(), score.recall(), score.f1());
  }

  /** The genre is the middle segment of a GUM document name. */
  private static String genre(String name) {
    final int first = name.indexOf('_');
    final int second = name.indexOf('_', first + 1);
    return second < 0 ? name : name.substring(first + 1, second);
  }

  /**
   * Reads the document names listed under one {@code ## split} heading of
   * {@code splits.md}.
   */
  private static List<String> splitDocuments(Path splits, String split)
      throws IOException {
    final List<String> names = new ArrayList<>();
    boolean inSplit = false;
    for (final String line : Files.readAllLines(splits, StandardCharsets.UTF_8)) {
      if (line.startsWith("## ")) {
        inSplit = line.substring(3).trim().equals(split);
      } else if (inSplit && line.trim().startsWith("* ")) {
        names.add(line.trim().substring(2).trim());
      }
    }
    return names;
  }

  /**
   * Builds a document from the OntoGUM CoNLL-U lines: gold sentences, tokens, and Penn
   * tags as layers, and the {@code Entity} brackets of the MISC column as the key.
   */
  private static GoldDocument read(String name, List<String> lines) {
    final StringBuilder text = new StringBuilder();
    final List<Annotation<String>> sentences = new ArrayList<>();
    final List<Annotation<String>> tokens = new ArrayList<>();
    final List<Annotation<String>> tags = new ArrayList<>();
    final Map<String, List<Integer>> openStarts = new HashMap<>();
    final Map<String, Set<Mention>> entities = new LinkedHashMap<>();
    int sentenceStart = -1;
    boolean glue = false;
    for (final String line : lines) {
      if (line.isEmpty()) {
        if (sentenceStart >= 0) {
          sentences.add(new Annotation<>(new Span(sentenceStart, text.length()),
              text.substring(sentenceStart)));
          text.append('\n');
          sentenceStart = -1;
          glue = true;
        }
        continue;
      }
      if (line.charAt(0) == '#') {
        continue;
      }
      final String[] fields = fields(line);
      if (fields.length < 10 || !isPlainId(fields[0])) {
        continue;
      }
      if (sentenceStart < 0) {
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
      tags.add(new Annotation<>(new Span(start, end), fields[4]));
      glue = fields[9].contains("SpaceAfter=No");
      final String entity = misc(fields[9], "Entity");
      if (entity != null) {
        brackets(entity, start, end, openStarts, entities);
      }
    }
    if (sentenceStart >= 0) {
      sentences.add(new Annotation<>(new Span(sentenceStart, text.length()),
          text.substring(sentenceStart)));
    }
    Assertions.assertTrue(openStarts.values().stream().allMatch(List::isEmpty),
        name + " leaves an entity bracket open");
    final Document document = Document.of(text.toString())
        .with(Layers.SENTENCES, sentences)
        .with(Layers.TOKENS, tokens)
        .with(Layers.POS_TAGS, tags);
    return new GoldDocument(name, document, disjoint(name, entities.values()));
  }

  /**
   * Keeps the first entity's claim on a span the conversion filed under two ids, so the
   * key stays a partition; the scorer rejects overlapping entities.
   */
  private static List<Set<Mention>> disjoint(String name,
      Collection<Set<Mention>> entities) {
    final Set<Mention> seen = new HashSet<>();
    final List<Set<Mention>> key = new ArrayList<>();
    int dropped = 0;
    for (final Set<Mention> entity : entities) {
      final Set<Mention> kept = new HashSet<>();
      for (final Mention mention : entity) {
        if (seen.add(mention)) {
          kept.add(mention);
        } else {
          dropped++;
        }
      }
      if (!kept.isEmpty()) {
        key.add(kept);
      }
    }
    if (dropped > 0) {
      LOG.info("{}: dropped {} key mention(s) filed under two entities", name, dropped);
    }
    return key;
  }

  /** Splits a CoNLL-U line on tabs. */
  private static String[] fields(String line) {
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
  private static boolean isPlainId(String id) {
    for (int i = 0; i < id.length(); i++) {
      if (id.charAt(i) < '0' || id.charAt(i) > '9') {
        return false;
      }
    }
    return !id.isEmpty();
  }

  /** Reads one {@code key=value} attribute of a MISC column, or {@code null}. */
  private static String misc(String misc, String key) {
    int start = 0;
    while (start < misc.length()) {
      int end = misc.indexOf('|', start);
      if (end < 0) {
        end = misc.length();
      }
      final String attribute = misc.substring(start, end);
      if (attribute.startsWith(key + "=")) {
        return attribute.substring(key.length() + 1);
      }
      start = end + 1;
    }
    return null;
  }

  /**
   * Applies the bracket notation of one token: {@code (id} opens a mention, {@code id)}
   * closes the innermost open mention of that id, and {@code (id)} does both on the
   * token. Rich GUM ids carry hyphenated attributes after the numeric id.
   */
  private static void brackets(String entity, int start, int end,
      Map<String, List<Integer>> openStarts, Map<String, Set<Mention>> entities) {
    int i = 0;
    while (i < entity.length()) {
      if (entity.charAt(i) == '(') {
        int j = i + 1;
        while (j < entity.length() && entity.charAt(j) != '(' && entity.charAt(j) != ')') {
          j++;
        }
        final String id = canonicalId(entity.substring(i + 1, j));
        if (j < entity.length() && entity.charAt(j) == ')') {
          entities.computeIfAbsent(id, key -> new HashSet<>()).add(new Mention(start, end));
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
        Assertions.assertTrue(starts != null && !starts.isEmpty(),
            "close without open for entity " + id);
        entities.computeIfAbsent(id, key -> new HashSet<>())
            .add(new Mention(starts.remove(starts.size() - 1), end));
        i = j + 1;
      }
    }
  }

  /** Strips the attribute tail of a rich GUM entity id. */
  private static String canonicalId(String id) {
    final int hyphen = id.indexOf('-');
    return hyphen < 0 ? id : id.substring(0, hyphen);
  }

  /**
   * Runs the name finders over each sentence and adds the entity layer, keeping the
   * first-found span wherever two finders overlap.
   */
  private static Document withEntities(Document document, List<NameFinderME> finders) {
    final List<Annotation<String>> sentences = document.get(Layers.SENTENCES);
    final List<Annotation<String>> tokens = document.get(Layers.TOKENS);
    final List<Annotation<String>> entities = new ArrayList<>();
    int first = 0;
    for (final Annotation<String> sentence : sentences) {
      int last = first;
      while (last < tokens.size()
          && tokens.get(last).span().getStart() < sentence.span().getEnd()) {
        last++;
      }
      final String[] words = new String[last - first];
      for (int t = first; t < last; t++) {
        words[t - first] = tokens.get(t).value();
      }
      final List<Span> found = new ArrayList<>();
      for (final NameFinderME finder : finders) {
        for (final Span span : finder.find(words)) {
          if (found.stream().noneMatch(span::intersects)) {
            found.add(span);
          }
        }
      }
      found.sort((a, b) -> Integer.compare(a.getStart(), b.getStart()));
      for (final Span span : found) {
        entities.add(new Annotation<>(new Span(
            tokens.get(first + span.getStart()).span().getStart(),
            tokens.get(first + span.getEnd() - 1).span().getEnd()), span.getType()));
      }
      first = last;
    }
    for (final NameFinderME finder : finders) {
      finder.clearAdaptiveData();
    }
    return document.with(Layers.ENTITIES, entities);
  }

  /** Groups the chains layer into entities of two or more mentions. */
  private static List<Set<Mention>> responsePartition(Document output) {
    final Map<Integer, Set<Mention>> chains = new TreeMap<>();
    for (final Annotation<CorefMention> mention : output.get(CorefAnnotator.CHAINS)) {
      chains.computeIfAbsent(mention.value().chain(), key -> new HashSet<>())
          .add(new Mention(mention.span().getStart(), mention.span().getEnd()));
    }
    final List<Set<Mention>> response = new ArrayList<>();
    for (final Collection<Mention> chain : chains.values()) {
      if (chain.size() > 1) {
        response.add(new HashSet<>(chain));
      }
    }
    return response;
  }
}
