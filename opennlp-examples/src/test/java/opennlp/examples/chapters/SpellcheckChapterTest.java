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

package opennlp.examples.chapters;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import opennlp.examples.DocExampleSupport;
import opennlp.spellcheck.SpellChecker;
import opennlp.spellcheck.SuggestItem;
import opennlp.spellcheck.Verbosity;
import opennlp.spellcheck.dictionary.FrequencyDictionaryLoader;
import opennlp.spellcheck.dictionary.SymSpellModel;
import opennlp.spellcheck.dictionary.SymSpellModels;
import opennlp.spellcheck.distance.DamerauOSADistance;
import opennlp.spellcheck.distance.LevenshteinDistance;
import opennlp.spellcheck.normalizer.SpellCheckingCharSequenceNormalizer;
import opennlp.spellcheck.stream.SpellCorrectingObjectStream;
import opennlp.spellcheck.symspell.SymSpell;
import opennlp.spellcheck.symspell.SymSpellConfig;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.normalizer.CharSequenceNormalizer;

/**
 * The samples printed in the SpellChecker chapter of the manual.
 * <p>
 * The dictionaries are staged as {@code en-unigrams.txt} and {@code en-bigrams.txt} from the
 * samples the chapter prints. The interface and record declarations are summaries rather than
 * code to run, and the model resolver sample needs an {@code opennlp-models-spellcheck-en}
 * artifact that is not published yet, so those three listings are not verified here.
 */
class SpellcheckChapterTest extends DocExampleSupport {

  private static final String UNIGRAMS = "en-unigrams.txt";

  private static final String BIGRAMS = "en-bigrams.txt";

  private static Path pipelineModel;

  @BeforeAll
  static void stageDictionaries() throws IOException {
    stageFile(UNIGRAMS, "the\t23135851162", "of\t13151942776", "and\t12997637966",
        "quick\t1885979", "brown\t1019234", "fox\t271289");
    stageFile(BIGRAMS, "quick brown\t12046", "brown fox\t5634", "the quick\t98765");
    final SymSpellModel model = SymSpellModels.buildModel("en", SymSpellConfig.defaultConfig(),
        StandardCharsets.UTF_8,
        new MarkableFileInputStreamFactory(workFile(UNIGRAMS).toFile()),
        new MarkableFileInputStreamFactory(workFile(BIGRAMS).toFile()));
    pipelineModel = workFile("en-spellcheck-pipeline.bin");
    try (OutputStream out = Files.newOutputStream(pipelineModel)) {
      SymSpellModels.serialize(model, out);
    }
  }

  @Test
  void configuresTheEngine() {
    // docs:begin example.spellcheck.api.engine
    SymSpellConfig config = SymSpellConfig.builder()
        .maxDictionaryEditDistance(2)   // largest precomputed edit distance (default 2)
        .prefixLength(7)                // leading symbols used for deletes (default 7)
        .countThreshold(1)              // minimum count to index a term (default 1)
        .editDistance(DamerauOSADistance.INSTANCE) // verification metric (default)
        .build();

    SymSpell symSpell = new SymSpell(config);   // or: new SymSpell() for the defaults
    symSpell.add("quick", 1_000);
    symSpell.add("brown", 800);
    symSpell.addBigram("quick", "brown", 120);
    // docs:end
    Assertions.assertEquals("quick", symSpell.lookup("qick").get(0).term());
  }

  @Test
  void looksUpATerm() {
    SymSpell symSpell = engine();
    // docs:begin example.spellcheck.api.lookup
    List<SuggestItem> suggestions = symSpell.lookup("qick", Verbosity.CLOSEST, 2);
    for (SuggestItem item : suggestions) {
      System.out.printf("%s (distance=%d, frequency=%d)%n",
          item.term(), item.editDistance(), item.frequency());
    }
    // docs:end
    Assertions.assertEquals("quick", suggestions.get(0).term());
    Assertions.assertEquals(1, suggestions.get(0).editDistance());
  }

  @Test
  void correctsACompound() {
    SymSpell symSpell = engine();
    // docs:begin example.spellcheck.api.compound
    String corrected = symSpell.lookupCompound("thequick brwn fox", 2).get(0).term();
    // -> "the quick brown fox"
    // docs:end
    Assertions.assertEquals("the quick brown fox", corrected);
  }

  @Test
  void choosesTheDistance() {
    // docs:begin example.spellcheck.api.distance
    SymSpellConfig config = SymSpellConfig.builder()
        .editDistance(LevenshteinDistance.INSTANCE)
        .build();
    // docs:end
    Assertions.assertSame(LevenshteinDistance.INSTANCE, config.editDistance());
  }

  @Test
  void buildsAndReloadsAModel() throws IOException {
    // docs:begin example.spellcheck.dictionary.model
    InputStreamFactory unigrams = new MarkableFileInputStreamFactory(new File("en-unigrams.txt"));
    InputStreamFactory bigrams = new MarkableFileInputStreamFactory(new File("en-bigrams.txt"));

    SymSpellModel model = SymSpellModels.buildModel(
        "en", SymSpellConfig.defaultConfig(), StandardCharsets.UTF_8, unigrams, bigrams);

    // Persist the binary model.
    try (OutputStream out = Files.newOutputStream(Path.of("en-spellcheck.bin"))) {
      SymSpellModels.serialize(model, out);
    }

    // Reload it later and query the engine.
    try (InputStream in = Files.newInputStream(Path.of("en-spellcheck.bin"))) {
      SymSpellModel loaded = SymSpellModels.deserialize(in);
      SpellChecker checker = loaded.getSymSpell();
      List<SuggestItem> suggestions = checker.lookup("qick");
    }
    // docs:end
    try (InputStream in = Files.newInputStream(workFile("en-spellcheck.bin"))) {
      Assertions.assertEquals("quick",
          SymSpellModels.deserialize(in).getSymSpell().lookup("qick").get(0).term());
    }
  }

  @Test
  void loadsDictionaries() throws IOException {
    InputStreamFactory unigrams = new MarkableFileInputStreamFactory(workFile(UNIGRAMS).toFile());
    InputStreamFactory bigrams = new MarkableFileInputStreamFactory(workFile(BIGRAMS).toFile());
    // docs:begin example.spellcheck.dictionary.loader
    SymSpell engine = new SymSpell(SymSpellConfig.defaultConfig());
    FrequencyDictionaryLoader loader = new FrequencyDictionaryLoader(StandardCharsets.UTF_8);
    loader.loadUnigrams(engine, unigrams);
    loader.loadBigrams(engine, bigrams);
    // docs:end
    Assertions.assertEquals("fox", engine.lookup("fx").get(0).term());
  }

  @Test
  void normalizesText() throws IOException {
    try (InputStream in = Files.newInputStream(pipelineModel)) {
      // docs:begin example.spellcheck.pipeline.normalizer
      SymSpellModel model = SymSpellModels.deserialize(in);

      CharSequenceNormalizer normalizer = SpellCheckingCharSequenceNormalizer.builder(model)
          .mode(SpellCheckingCharSequenceNormalizer.Mode.PER_TOKEN)
          .maxEditDistance(2)
          .build();

      CharSequence clean = normalizer.normalize("the qick brwn fox"); // -> "the quick brown fox"
      // docs:end
      Assertions.assertEquals("the quick brown fox", clean.toString());
    }
  }

  @Test
  void keepsSupplementaryCharacters() {
    // docs:begin example.spellcheck.pipeline.supplementary
    SymSpell engine = new SymSpell();
    engine.add("\uD801\uDC28word", 1000);
    CharSequenceNormalizer normalizer = new SpellCheckingCharSequenceNormalizer(engine);

    String corrected = normalizer.normalize("\uD801\uDC28WROLD").toString();
    // corrected is "\uD801\uDC28word"
    // docs:end
    Assertions.assertEquals("\uD801\uDC28word", corrected);
  }

  @Test
  void correctsAStream() throws IOException {
    SymSpellModel model;
    try (InputStream in = Files.newInputStream(pipelineModel)) {
      model = SymSpellModels.deserialize(in);
    }
    stageFile("noisy.txt", "the qick brwn fox");
    InputStreamFactory inputStreamFactory =
        new MarkableFileInputStreamFactory(workFile("noisy.txt").toFile());
    // docs:begin example.spellcheck.pipeline.stream
    ObjectStream<String> lines =
        new PlainTextByLineStream(inputStreamFactory, StandardCharsets.UTF_8);
    ObjectStream<String> corrected = new SpellCorrectingObjectStream(lines, model);

    String line;
    while ((line = corrected.read()) != null) {
      // ... consume corrected lines ...
    }
    // docs:end
    corrected.close();
    try (ObjectStream<String> again = new SpellCorrectingObjectStream(new PlainTextByLineStream(
        inputStreamFactory, StandardCharsets.UTF_8), model)) {
      Assertions.assertEquals("the quick brown fox", again.read());
    }
  }

  /* The engine of the engine sample, with the rest of the unigrams the chapter prints. */
  private static SymSpell engine() {
    SymSpell symSpell = new SymSpell();
    symSpell.add("the", 23_135_851_162L);
    symSpell.add("quick", 1_000);
    symSpell.add("brown", 800);
    symSpell.add("fox", 271_289);
    symSpell.addBigram("quick", "brown", 120);
    symSpell.addBigram("the", "quick", 98_765);
    symSpell.addBigram("brown", "fox", 5_634);
    return symSpell;
  }
}
