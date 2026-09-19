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

package opennlp.tools.parser;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ParseTest {

  @AfterEach
  void resetFunctionTags() {
    Parse.useFunctionTags(false);
  }

  @Test
  void testParsesTreebankTreeAcrossWhitespace() {
    Parse parse = Parse.parseParse("(TOP\n  (S\t(NP-SBJ-1 (NNP José))\n(VP (VBZ runs)) (. .)))");

    assertEquals("José runs . ", parse.getText());
    Parse sentence = parse.getChildren()[0];
    assertEquals("S", sentence.getType());
    assertEquals("NP", sentence.getChildren()[0].getType());
    assertEquals("José", sentence.getChildren()[0].getCoveredText());
  }

  @Test
  void testRetainsFirstFunctionTagWhenEnabled() {
    Parse.useFunctionTags(true);

    Parse parse = Parse.parseParse("(TOP (S (NP-SBJ-1 (NNP Renée))))");

    assertEquals("NP-SBJ", parse.getChildren()[0].getChildren()[0].getType());
  }

  @ParameterizedTest
  @CsvSource({"NP-SBJ-TMP-1, NP-SBJ-TMP", "NP-1, NP", "NP-SBJ=2, NP-SBJ",
      "NP-SBJ-TMP=2, NP-SBJ-TMP"})
  void testFunctionTagsExcludeNumericCoindices(String label, String expected) {
    Parse.useFunctionTags(true);
    Parse parse = Parse.parseParse("(TOP (S (" + label + " (NN word))))");
    assertEquals(expected, parse.getChildren()[0].getChildren()[0].getType());
  }

  @Test
  void testUnicodeWhitespaceSeparatesLabelsAndTokens() {
    Parse parse = Parse.parseParse("(TOP\u00a0(NN\u2003word))");
    assertEquals("word ", parse.getText());
    assertEquals("NN", parse.getChildren()[0].getType());
  }

  @Test
  void testDecodesTreebankBracketTokens() {
    Parse parse = Parse.parseParse("(TOP (S (-LRB- -LRB-) (NN 🐈) (-RRB- -RRB-)))");

    assertEquals("( 🐈 ) ", parse.getText());
  }

  @Test
  void testParsesTreebankFileWrapper() {
    Parse parse = Parse.parseParse("((S (NP-SBJ (PRP They)) (VP (VBP agree)) (. .)))");

    assertEquals("They agree . ", parse.getText());
    assertEquals("S", parse.getChildren()[0].getType());
  }

  @ParameterizedTest
  @NullAndEmptySource
  @ValueSource(strings = {
      "   ",
      "TOP",
      ")",
      "(TOP (NN token)",
      "(TOP () (NN token))",
      "(TOP (NN two tokens))",
      "(S (NN one)) (S (NN two))",
      "(TOP (NN token)) trailing"
  })
  void testRejectsMalformedTreebankTrees(String tree) {
    assertThrows(IllegalArgumentException.class, () -> Parse.parseParse(tree));
  }
}
