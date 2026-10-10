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
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import opennlp.examples.DocExampleSupport;
import opennlp.tools.lemmatizer.DictionaryLemmatizer;
import opennlp.tools.lemmatizer.LemmaSampleStream;
import opennlp.tools.lemmatizer.LemmatizerFactory;
import opennlp.tools.lemmatizer.LemmatizerME;
import opennlp.tools.lemmatizer.LemmatizerModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;

/**
 * The samples printed in the Lemmatizer chapter of the manual.
 * <p>
 * The statistical samples run against the published English lemmatizer model. The dictionary
 * samples read a small dictionary in the format the chapter prints, and the training samples a
 * small fixture in the training format the chapter prints.
 */
class LemmatizerChapterTest extends DocExampleSupport {

  private static final String DICTIONARY_FILE = "english-dict-lemmatizer.txt";

  private static final String TRAINING_FILE = "en-custom-lemmatizer.train";

  private static LemmatizerModel published;

  private static POSModel posModel;

  @BeforeAll
  static void loadModels() throws IOException {
    try (InputStream in = Files.newInputStream(stageModel("opennlp-models-lemmatizer-*.jar",
        "opennlp-en-ud-ewt-lemmas-1.3-2.5.4.bin"))) {
      published = new LemmatizerModel(in);
    }
    try (InputStream in = Files.newInputStream(stageModel("opennlp-models-pos-*.jar",
        "opennlp-en-ud-ewt-pos-1.3-2.5.4.bin"))) {
      posModel = new POSModel(in);
    }
    stageFile(DICTIONARY_FILE,
        "show\tNOUN\tshow",
        "shows\tNOUN\tshow",
        "cities\tNOUN\tcity",
        "newspapers\tNOUN\tnewspaper",
        "had\tVERB\thave",
        "had\tAUX\thave");
    stageTrainingData();
  }

  @Test
  void loadsTheModel() throws IOException {
    // docs:begin example.lemmatizer.api.model
    LemmatizerModel model = null;
    try (InputStream modelIn = new FileInputStream("opennlp-en-ud-ewt-lemmas-1.3-2.5.4.bin")) {
      model = new LemmatizerModel(modelIn);
    }
    // docs:end
    Assertions.assertNotNull(model);
  }

  @Test
  void createsTheLemmatizer() {
    LemmatizerModel model = published;
    // docs:begin example.lemmatizer.api.lemmatizer
    LemmatizerME lemmatizer = new LemmatizerME(model);
    // docs:end
    Assertions.assertNotNull(lemmatizer);
  }

  @Test
  void lemmatizesASentence() {
    LemmatizerME lemmatizer = new LemmatizerME(published);
    // docs:begin example.lemmatizer.api.lemmatize
    String[] tokens = new String[] { "Rockwell", "International", "Corp.", "'s",
        "Tulsa", "unit", "said", "it", "signed", "a", "tentative", "agreement",
        "extending", "its", "contract", "with", "Boeing", "Co.", "to",
        "provide", "structural", "parts", "for", "Boeing", "'s", "747",
        "jetliners", "." };

    String[] postags = new String[] { "PROPN", "ADJ", "NOUN", "PUNCT", "PROPN", "NOUN",
        "VERB", "PRON", "VERB", "DET", "NOUN", "NOUN", "VERB", "PRON", "NOUN", "ADP",
        "PROPN", "NOUN", "PART", "VERB", "ADJ", "NOUN", "ADP", "PROPN", "PUNCT", "NUM", "NOUN",
        "PUNCT" };

    String[] lemmas = lemmatizer.lemmatize(tokens, postags);
    // docs:end
    Assertions.assertEquals(tokens.length, lemmas.length);
    Assertions.assertEquals("say", lemmas[6]);
    Assertions.assertEquals("part", lemmas[21]);
  }

  @Test
  void loadsADictionary() throws IOException {
    // docs:begin example.lemmatizer.dictionary.load
    DictionaryLemmatizer lemmatizer;
    try (InputStream dictLemmatizer = new FileInputStream("english-dict-lemmatizer.txt")) {
      lemmatizer = new DictionaryLemmatizer(dictLemmatizer);
    }
    // docs:end
    Assertions.assertArrayEquals(new String[] {"show"},
        lemmatizer.lemmatize(new String[] {"shows"}, new String[] {"NOUN"}));
  }

  @Test
  void lemmatizesWithADictionary() throws IOException {
    DictionaryLemmatizer lemmatizer = new DictionaryLemmatizer(
        workFile(DICTIONARY_FILE).toFile());
    POSTaggerME tagger = new POSTaggerME(posModel);
    // docs:begin example.lemmatizer.dictionary.lemmatize
    String[] tokens = new String[]{"Most", "large", "cities", "in", "the", "US", "had",
                                   "morning", "and", "afternoon", "newspapers", "."};
    String[] tags = tagger.tag(tokens);
    String[] lemmas = lemmatizer.lemmatize(tokens, tags);
    // docs:end
    Assertions.assertEquals(tokens.length, lemmas.length);
    Assertions.assertEquals("city", lemmas[2]);
    Assertions.assertEquals("newspaper", lemmas[10]);
  }

  @Test
  void createsTrainingParameters() {
    // docs:begin example.lemmatizer.training.api.params
    TrainingParameters mlParams = TrainingParameters.defaultParams();
    // docs:end
    Assertions.assertNotNull(mlParams);
  }

  @Test
  void opensTrainingData() throws IOException {
    // docs:begin example.lemmatizer.training.api.data
    InputStreamFactory inputStreamFactory =
        new MarkableFileInputStreamFactory(new File("en-custom-lemmatizer.train"));
    ObjectStream<String> lineStream =
        new PlainTextByLineStream(inputStreamFactory, StandardCharsets.UTF_8);
    LemmaSampleStream lemmaStream = new LemmaSampleStream(lineStream);
    // docs:end
    try (lemmaStream) {
      Assertions.assertNotNull(lemmaStream.read());
    }
  }

  @Test
  void trainsAModel() throws IOException {
    TrainingParameters mlParams = TrainingParameters.defaultParams();
    LemmaSampleStream lemmaStream = new LemmaSampleStream(new PlainTextByLineStream(
        new MarkableFileInputStreamFactory(workFile(TRAINING_FILE).toFile()),
        StandardCharsets.UTF_8));
    // docs:begin example.lemmatizer.training.api.train
    LemmatizerModel model;
    try (lemmaStream) {
      model = LemmatizerME.train("eng", lemmaStream, mlParams, new LemmatizerFactory());
    }
    // docs:end
    Assertions.assertArrayEquals(new String[] {"reckon"},
        new LemmatizerME(model).lemmatize(new String[] {"reckons"}, new String[] {"VERB"}));
  }

  /* The sample sentence the chapter prints, one tab separated token per line. */
  private static void stageTrainingData() throws IOException {
    final String[][] rows = {{"He", "PRON", "he"}, {"reckons", "VERB", "reckon"},
        {"the", "DET", "the"}, {"current", "ADJ", "current"}, {"accounts", "NOUN", "account"},
        {"deficit", "NOUN", "deficit"}, {"will", "AUX", "will"}, {"narrow", "VERB", "narrow"},
        {"to", "PART", "to"}, {"only", "ADV", "only"}, {"#", "#", "#"},
        {"1.8", "NUM", "1.8"}, {"millions", "NOUN", "million"}, {"in", "ADP", "in"},
        {"September", "PROPN", "september"}, {".", "PUNCT", "O"}};
    final List<String> lines = new ArrayList<>();
    for (String[] row : rows) {
      lines.add(String.join("\t", row));
    }
    lines.add("");
    stageFile(TRAINING_FILE, 10, lines.toArray(new String[0]));
  }
}
