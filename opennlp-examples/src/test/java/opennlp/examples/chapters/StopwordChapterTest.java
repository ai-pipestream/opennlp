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

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Set;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.examples.DocExampleSupport;
import opennlp.tools.stopword.DictionaryStopwordFilter;
import opennlp.tools.stopword.StopwordFilter;
import opennlp.tools.stopword.StopwordFilteringTokenizer;
import opennlp.tools.stopword.StopwordLists;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.tokenize.Tokenizer;

/**
 * The samples printed in the Stopword Filtering chapter of the manual.
 * <p>
 * Each sample states its result in a trailing comment; the assertions after the markers check
 * those comments. The streaming sample uses {@code ...} as a placeholder and is not verified.
 */
class StopwordChapterTest extends DocExampleSupport {

  @Test
  void filtersWithABundledList() {
    // docs:begin example.stopword.bundled
    StopwordFilter filter = StopwordLists.forLanguage("en");

    boolean isStop = filter.isStopword("the");        // true
    String[] kept  = filter.filter(new String[] {
        "the", "quick", "brown", "fox"
    });
    // kept => { "quick", "brown", "fox" }
    // docs:end
    Assertions.assertTrue(isStop);
    Assertions.assertArrayEquals(new String[] {"quick", "brown", "fox"}, kept);
  }

  @Test
  void listsSupportedLanguages() {
    // docs:begin example.stopword.bundled.languages
    Set<String> supported = StopwordLists.supportedLanguages();
    // docs:end
    Assertions.assertTrue(supported.contains("en"));
  }

  @Test
  void loadsACustomList() throws IOException {
    stageFile("my-stopwords.txt", "# project noise terms", "foo", "", "Bar", "of the");
    // docs:begin example.stopword.custom
    try (InputStream in = new FileInputStream("my-stopwords.txt")) {
      StopwordFilter filter = StopwordLists.load(
          in, StandardCharsets.UTF_8, false /* case-insensitive */);
    }
    // docs:end
    try (InputStream in = new FileInputStream(workFile("my-stopwords.txt").toFile())) {
      final StopwordFilter filter = StopwordLists.load(in, StandardCharsets.UTF_8, false);
      Assertions.assertTrue(filter.isStopword("bar"));
      Assertions.assertTrue(filter.isStopword("of", "the"));
      Assertions.assertFalse(filter.isStopword("#"));
    }
  }

  @Test
  void extendsABundledList() throws IOException {
    // docs:begin example.stopword.extending
    StopwordFilter filter;
    try (InputStream bundled =
        StopwordLists.class.getResourceAsStream("/opennlp/tools/stopword/en.txt")) {
      filter = DictionaryStopwordFilter.builder()
          .load(bundled, StandardCharsets.UTF_8)
          .add("foo")            // mark a new word as a stopword
          .remove("the")         // keep "the" in the output
          .build();
    }
    // docs:end
    Assertions.assertTrue(filter.isStopword("foo"));
    Assertions.assertFalse(filter.isStopword("the"));
    Assertions.assertTrue(filter.isStopword("a"));
  }

  @Test
  void matchesMultiWordEntries() {
    // docs:begin example.stopword.multiword
    StopwordFilter filter = DictionaryStopwordFilter.builder()
        .add("of", "the")
        .add("in", "spite", "of")
        .build();

    filter.isStopword("of", "the");        // true
    filter.isStopword("in", "spite", "of"); // true
    filter.isStopword("of");                // false
    // docs:end
    Assertions.assertTrue(filter.isStopword("of", "the"));
    Assertions.assertTrue(filter.isStopword("in", "spite", "of"));
    Assertions.assertFalse(filter.isStopword("of"));
  }

  @Test
  void filtersMultiWordEntries() {
    // docs:begin example.stopword.multiword.filter
    StopwordFilter filter = DictionaryStopwordFilter.builder()
        .add("the")               // 1-gram
        .add("of", "the")         // 2-gram
        .add("in", "spite", "of") // 3-gram
        .build();

    String[] kept = filter.filter(new String[] {
        "the", "king", "of", "the", "hill",
        "won", "in", "spite", "of", "rain"
    });
    // kept => { "king", "hill", "won", "rain" }
    // docs:end
    Assertions.assertArrayEquals(new String[] {"king", "hill", "won", "rain"}, kept);
  }

  @Test
  void filtersWhileTokenizing() {
    // docs:begin example.stopword.tokenizer
    StopwordFilter filter = StopwordLists.forLanguage("en");
    Tokenizer tokenizer = new StopwordFilteringTokenizer(
        SimpleTokenizer.INSTANCE, filter);

    String[] tokens = tokenizer.tokenize("The quick brown fox jumps over the lazy dog");
    // tokens => { "quick", "brown", "fox", "jumps", "lazy", "dog" }
    // docs:end
    Assertions.assertArrayEquals(
        new String[] {"quick", "brown", "fox", "jumps", "lazy", "dog"}, tokens);
  }
}
