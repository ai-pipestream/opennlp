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

import java.util.List;

import org.junit.jupiter.api.Test;

import opennlp.tools.wordnet.LexicalKnowledgeBase;
import opennlp.tools.wordnet.WordNetPOS;
import opennlp.wordnet.LexicalExpander.Expansion;
import opennlp.wordnet.LexicalExpander.Kind;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end expansion over the miniature lexicon fixtures: the WN-LMF and WNDB readers each
 * feed the expander, and the Morphy lemmatizer bridges inflected input, exercising the whole
 * stack the way a consumer wires it.
 */
class LexicalExpanderLexiconTest {

  private static Expansion find(List<Expansion> expansions, String term) {
    return expansions.stream().filter(e -> e.term().equals(term)).findFirst().orElse(null);
  }

  @Test
  void testExpansionOverTheWnLmfLexicon() {
    final LexicalExpander expander =
        LexicalExpander.builder(WnLmfReaderTest.fixture()).build();

    final List<Expansion> expansions = expander.expand("dog", WordNetPOS.NOUN);
    final Expansion domesticDog = find(expansions, "domestic dog");
    assertEquals(Kind.SYNONYM, domesticDog.kind());
    assertEquals(1.0, domesticDog.weight());
    final Expansion canid = find(expansions, "canid");
    assertEquals(Kind.HYPERNYM, canid.kind());
    assertEquals(0.5, canid.weight());
  }

  @Test
  void testExpansionOverTheWndbLexicon() {
    final LexicalExpander expander =
        LexicalExpander.builder(WndbReaderTest.fixture()).build();

    final List<Expansion> expansions = expander.expand("dog", WordNetPOS.NOUN);
    assertTrue(find(expansions, "domestic dog") != null, "got " + expansions);
    final Expansion canid = find(expansions, "canid");
    assertEquals(Kind.HYPERNYM, canid.kind());
  }

  @Test
  void testReadersAgreeOnExpansions() {
    final List<Expansion> lmf = LexicalExpander.builder(WnLmfReaderTest.fixture())
        .hypernymDepth(2).build().expand("mouse", WordNetPOS.NOUN);
    final List<Expansion> wndb = LexicalExpander.builder(WndbReaderTest.fixture())
        .hypernymDepth(2).build().expand("mouse", WordNetPOS.NOUN);

    assertEquals(
        lmf.stream().map(e -> e.term() + "|" + e.kind() + "|" + e.weight()).toList(),
        wndb.stream().map(e -> e.term() + "|" + e.kind() + "|" + e.weight()).toList());
    assertTrue(find(lmf, "rodent") != null, "got " + lmf);
  }

  @Test
  void testMorphyBridgesInflectedInput() {
    final LexicalKnowledgeBase lexicon = WnLmfReaderTest.fixture();
    final LexicalExpander expander = LexicalExpander.builder(lexicon)
        .lemmatizer(new MorphyLemmatizer(lexicon, MorphyExceptionsTest.fixture()))
        .build();

    // A regular inflection resolves by rule, an irregular one by the exception list.
    final List<Expansion> dogs = expander.expand("dogs", WordNetPOS.NOUN);
    assertEquals(Kind.SYNONYM, find(dogs, "dog").kind());
    assertTrue(find(dogs, "canid") != null, "got " + dogs);

    final List<Expansion> mice = expander.expand("mice", WordNetPOS.NOUN);
    assertEquals(Kind.SYNONYM, find(mice, "mouse").kind());
    assertTrue(find(mice, "rodent") != null, "got " + mice);
  }
}
