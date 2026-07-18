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

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import opennlp.tools.postag.BilstmPOSModel;
import opennlp.tools.postag.BilstmPOSTagger;
import opennlp.tools.postag.BilstmPOSTrainer;
import opennlp.tools.postag.FeedforwardPOSModel;
import opennlp.tools.postag.FeedforwardPOSTagger;
import opennlp.tools.postag.FeedforwardPOSTrainer;
import opennlp.tools.postag.POSEvaluator;
import opennlp.tools.postag.POSSample;
import opennlp.tools.postag.POSTagger;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.wordvector.Glove;
import opennlp.tools.util.wordvector.WordVector;
import opennlp.tools.util.wordvector.WordVectorTable;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Trains POS taggers on a Universal Dependencies treebank and scores them on the
 * treebank's test split, reporting UPOS word accuracy.
 *
 * <p>Runs only when {@code opennlp.postag.ud.dir} names a directory containing
 * {@code train.conllu} and {@code test.conllu} (a UD treebank's splits, renamed or
 * linked). The data is downloaded by the runner and never enters the repository; check
 * the treebank's own license before training models for distribution. Two optional
 * properties widen the feedforward control: {@code opennlp.postag.vectors} names a
 * GloVe-format text file consulted as the pretrained vector source, and
 * {@code opennlp.postag.lexicon} names a one-word-per-line file whose words get stored
 * vectors for tagging-time coverage. Assertions are low regression floors; the logged
 * scores are the measurement.</p>
 */
public class ConlluPOSTaggerEvalTest {

  private static final Logger logger =
      LoggerFactory.getLogger(ConlluPOSTaggerEvalTest.class);

  @Test
  @EnabledIfSystemProperty(named = "opennlp.postag.ud.dir", matches = ".+")
  void testFeedforwardBaseline() throws IOException {
    final Path dir = Path.of(System.getProperty("opennlp.postag.ud.dir"));

    final long trainStart = System.currentTimeMillis();
    final FeedforwardPOSModel model;
    try (ObjectStream<POSSample> train = samples(dir.resolve("train.conllu"))) {
      model = FeedforwardPOSTrainer.train(train, FeedforwardPOSTrainer.Settings.defaults());
    }
    logger.info("feedforward baseline trained in {} ms", System.currentTimeMillis() - trainStart);

    final double accuracy = evaluate(model, dir.resolve("test.conllu"));
    logger.info("feedforward baseline UPOS accuracy {}", accuracy);
    record("ff-baseline", accuracy, "");

    // a regression floor, far below any plausible result; the log line is the measurement
    assertTrue(accuracy > 0.85d, "UPOS accuracy regressed below the floor");
  }

  @Test
  @EnabledIfSystemProperty(named = "opennlp.postag.vectors", matches = ".+")
  void testFeedforwardWithPretrainedVectors() throws IOException {
    final Path dir = Path.of(System.getProperty("opennlp.postag.ud.dir"));
    final Function<CharSequence, float[]> vectors = vectors(
        Path.of(System.getProperty("opennlp.postag.vectors")));

    final List<String> lexicon;
    final String lexiconPath = System.getProperty("opennlp.postag.lexicon");
    if (lexiconPath != null && !lexiconPath.isBlank()) {
      lexicon = Files.readAllLines(Path.of(lexiconPath));
    }
    else {
      lexicon = null;
    }

    final long trainStart = System.currentTimeMillis();
    final FeedforwardPOSModel model;
    try (ObjectStream<POSSample> train = samples(dir.resolve("train.conllu"))) {
      if (lexicon == null) {
        model = FeedforwardPOSTrainer.train(train,
            FeedforwardPOSTrainer.Settings.defaults(), vectors);
      }
      else {
        model = FeedforwardPOSTrainer.train(train,
            FeedforwardPOSTrainer.Settings.defaults(), vectors, lexicon);
      }
    }
    logger.info("feedforward pretrained trained in {} ms",
        System.currentTimeMillis() - trainStart);

    final double accuracy = evaluate(model, dir.resolve("test.conllu"));
    logger.info("feedforward pretrained UPOS accuracy {} (lexicon: {})", accuracy,
        lexicon == null ? "none" : lexicon.size() + " words");
    record("ff-pretrained" + (lexicon == null ? "" : "-lexicon"), accuracy, "");

    // a regression floor, far below any plausible result; the log line is the measurement
    assertTrue(accuracy > 0.85d, "UPOS accuracy regressed below the floor");
  }

  /**
   * Appends one measurement line to {@code target/postag-eval-results.csv} so sweep
   * results survive quiet console output.
   *
   * @param run The run label.
   * @param accuracy The measured value (UPOS accuracy, or tokens/s for throughput rows).
   * @param config The configuration label, empty when not applicable.
   * @throws IOException Thrown if the results file cannot be written.
   */
  private static void record(String run, double accuracy, String config)
      throws IOException {
    final Path results = Path.of("target", "postag-eval-results.csv");
    Files.createDirectories(results.getParent());
    Files.writeString(results,
        run + "," + accuracy + "," + config + "\n", StandardCharsets.UTF_8,
        Files.exists(results) ? StandardOpenOption.APPEND : StandardOpenOption.CREATE);
  }

  /**
   * Builds the BiLSTM training settings, letting sweep runs override any field with
   * an {@code opennlp.postag.bilstm.<field>} system property while defaulting to
   * {@link BilstmPOSTrainer.Settings#defaults()}.
   *
   * @return The effective settings. Never {@code null}.
   */
  private static BilstmPOSTrainer.Settings bilstmSettings() {
    final BilstmPOSTrainer.Settings base = BilstmPOSTrainer.Settings.defaults();
    return new BilstmPOSTrainer.Settings(
        intProperty("wordEmbeddingSize", base.wordEmbeddingSize()),
        intProperty("charEmbeddingSize", base.charEmbeddingSize()),
        intProperty("charHiddenSize", base.charHiddenSize()),
        intProperty("hiddenSize", base.hiddenSize()),
        intProperty("epochs", base.epochs()),
        intProperty("batchSize", base.batchSize()),
        doubleProperty("learningRate", base.learningRate()),
        doubleProperty("clipNorm", base.clipNorm()),
        doubleProperty("dropout", base.dropout()),
        intProperty("wordCutoff", base.wordCutoff()),
        intProperty("maxWordLength", base.maxWordLength()),
        longProperty("seed", base.seed()),
        intProperty("threads", base.threads()));
  }

  private static String bilstmLabel(BilstmPOSTrainer.Settings s) {
    return "h" + s.hiddenSize() + ";e" + s.epochs() + ";b" + s.batchSize() + ";lr"
        + s.learningRate() + ";d" + s.dropout() + ";seed" + s.seed() + ";t" + s.threads()
        + ";we" + s.wordEmbeddingSize() + ";ch" + s.charHiddenSize();
  }

  private static int intProperty(String name, int fallback) {
    final String value = System.getProperty("opennlp.postag.bilstm." + name);
    return value != null ? Integer.parseInt(value) : fallback;
  }

  private static long longProperty(String name, long fallback) {
    final String value = System.getProperty("opennlp.postag.bilstm." + name);
    return value != null ? Long.parseLong(value) : fallback;
  }

  private static double doubleProperty(String name, double fallback) {
    final String value = System.getProperty("opennlp.postag.bilstm." + name);
    return value != null ? Double.parseDouble(value) : fallback;
  }

  @Test
  @EnabledIfSystemProperty(named = "opennlp.postag.ud.dir", matches = ".+")
  void testBilstmBaseline() throws IOException {
    final Path dir = Path.of(System.getProperty("opennlp.postag.ud.dir"));

    final long trainStart = System.currentTimeMillis();
    final BilstmPOSModel model;
    final BilstmPOSTrainer.Settings settings = bilstmSettings();
    try (ObjectStream<POSSample> train = samples(dir.resolve("train.conllu"))) {
      model = BilstmPOSTrainer.train(train, settings);
    }
    logger.info("bilstm baseline trained in {} ms", System.currentTimeMillis() - trainStart);

    final BilstmPOSTagger tagger = new BilstmPOSTagger(model);
    final double accuracy = evaluate(tagger, dir.resolve("test.conllu"));
    final double throughput = measureThroughput(tagger, dir.resolve("test.conllu"));
    logger.info("bilstm baseline UPOS accuracy {}, throughput {} tokens/s", accuracy,
        throughput);
    record("bilstm-baseline", accuracy, bilstmLabel(settings));
    record("bilstm-baseline-tokens-per-s", throughput, bilstmLabel(settings));

    // a regression floor, far below any plausible result; the log line is the measurement
    assertTrue(accuracy > 0.85d, "UPOS accuracy regressed below the floor");
  }

  @Test
  @EnabledIfSystemProperty(named = "opennlp.postag.vectors", matches = ".+")
  void testBilstmWithPretrainedVectors() throws IOException {
    final Path dir = Path.of(System.getProperty("opennlp.postag.ud.dir"));
    final Function<CharSequence, float[]> vectors = vectors(
        Path.of(System.getProperty("opennlp.postag.vectors")));

    final List<String> lexicon;
    final String lexiconPath = System.getProperty("opennlp.postag.lexicon");
    if (lexiconPath != null && !lexiconPath.isBlank()) {
      lexicon = Files.readAllLines(Path.of(lexiconPath));
    }
    else {
      lexicon = null;
    }

    final long trainStart = System.currentTimeMillis();
    final BilstmPOSModel model;
    final BilstmPOSTrainer.Settings settings = bilstmSettings();
    try (ObjectStream<POSSample> train = samples(dir.resolve("train.conllu"))) {
      if (lexicon == null) {
        model = BilstmPOSTrainer.train(train, settings, vectors);
      }
      else {
        model = BilstmPOSTrainer.train(train, settings, vectors, lexicon);
      }
    }
    logger.info("bilstm pretrained trained in {} ms", System.currentTimeMillis() - trainStart);

    final BilstmPOSTagger tagger = new BilstmPOSTagger(model);
    final double accuracy = evaluate(tagger, dir.resolve("test.conllu"));
    final double throughput = measureThroughput(tagger, dir.resolve("test.conllu"));
    logger.info("bilstm pretrained UPOS accuracy {} (lexicon: {}), throughput {} tokens/s",
        accuracy, lexicon == null ? "none" : lexicon.size() + " words", throughput);
    record("bilstm-pretrained" + (lexicon == null ? "" : "-lexicon"), accuracy,
        bilstmLabel(settings));
    record("bilstm-pretrained" + (lexicon == null ? "" : "-lexicon") + "-tokens-per-s",
        throughput, bilstmLabel(settings));

    // a regression floor, far below any plausible result; the log line is the measurement
    assertTrue(accuracy > 0.85d, "UPOS accuracy regressed below the floor");
  }

  private static ObjectStream<POSSample> samples(Path conllu) throws IOException {
    final InputStreamFactory in = new MarkableFileInputStreamFactory(conllu.toFile());
    return new ConlluPOSSampleStream(new ConlluStream(in), ConlluTagset.U);
  }

  private static double evaluate(FeedforwardPOSModel model, Path testConllu)
      throws IOException {
    return evaluate(new FeedforwardPOSTagger(model), testConllu);
  }

  private static double evaluate(POSTagger tagger, Path testConllu)
      throws IOException {
    final POSEvaluator evaluator = new POSEvaluator(tagger);
    try (ObjectStream<POSSample> test = samples(testConllu)) {
      evaluator.evaluate(test);
    }
    return evaluator.getWordAccuracy();
  }

  /**
   * Times the tagger over the whole test split in one pass, warm representation
   * cache included, and returns tokens per second.
   */
  private static double measureThroughput(POSTagger tagger, Path testConllu)
      throws IOException {
    final List<String[]> sentences = new ArrayList<>();
    int tokens = 0;
    try (ObjectStream<POSSample> test = samples(testConllu)) {
      POSSample sample;
      while ((sample = test.read()) != null) {
        sentences.add(sample.getSentence());
        tokens += sample.getSentence().length;
      }
    }
    final long start = System.nanoTime();
    for (final String[] sentence : sentences) {
      tagger.tag(sentence);
    }
    return tokens / ((System.nanoTime() - start) / 1e9d);
  }

  /**
   * Builds the word-vector source over a GloVe-format table of raw (unnormalized)
   * rows, mirroring {@code StaticEmbeddingModel.embed}: a whole-word hit is the raw
   * row, a miss falls back to greedy longest-match subword segmentation (the table's
   * {@code ##}-prefixed continuation pieces) with mean pooling, and the pooled result
   * is L2-normalized. Words that cannot be segmented at all yield {@code null}.
   *
   * @param gloveFile The vector text file.
   * @return The vector source. Never {@code null}.
   * @throws IOException Thrown if the file cannot be read.
   */
  private static Function<CharSequence, float[]> vectors(Path gloveFile) throws IOException {
    final WordVectorTable table;
    try (InputStream in = Files.newInputStream(gloveFile)) {
      table = Glove.parse(in);
    }
    logger.info("loaded {} vectors of dimension {} from {}", table.size(),
        table.dimension(), gloveFile);
    return word -> {
      final String form = word.toString();
      final float[] pooled;
      final WordVector hit = table.get(form);
      if (hit != null) {
        pooled = new float[hit.dimension()];
        for (int i = 0; i < pooled.length; i++) {
          pooled[i] = hit.getAsFloat(i);
        }
      }
      else {
        pooled = subwordPool(table, form);
        if (pooled == null) {
          return null;
        }
      }
      double sumOfSquares = 0.0d;
      for (final float value : pooled) {
        sumOfSquares += (double) value * value;
      }
      final double norm = Math.max(Math.sqrt(sumOfSquares), 1e-12d);
      for (int i = 0; i < pooled.length; i++) {
        pooled[i] /= (float) norm;
      }
      return pooled;
    };
  }

  private static float[] subwordPool(WordVectorTable table, String word) {
    final float[] sum = new float[table.dimension()];
    int count = 0;
    int start = 0;
    while (start < word.length()) {
      String match = null;
      final int longest = Math.min(word.length(), start + 60);
      for (int end = longest; end > start; end--) {
        final String piece =
            (start == 0 ? "" : "##") + word.substring(start, end);
        if (table.get(piece) != null) {
          match = piece;
          break;
        }
      }
      if (match == null) {
        return null;
      }
      final WordVector vector = table.get(match);
      for (int i = 0; i < sum.length; i++) {
        sum[i] += vector.getAsFloat(i);
      }
      count++;
      start += match.length() - (start == 0 ? 0 : 2);
    }
    for (int i = 0; i < sum.length; i++) {
      sum[i] /= count;
    }
    return sum;
  }
}
