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
package opennlp.tools.util.normalizer;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class SymbolJoinerCharSequenceNormalizerTest {

  @Test
  void testWholeTokenAmpersandSpellsOut() {
    assertEquals("and",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("&").toString());
  }

  @Test
  void testTheJoinerAndReferenceMarksSpellOut() {
    assertEquals("plus",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("+").toString());
    assertEquals("at",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("@").toString());
    assertEquals("percent",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("%").toString());
    // The legal reference marks: "§ 1983" meets a query typing "section 1983".
    assertEquals("section",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("§").toString());
    assertEquals("paragraph",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("¶").toString());
    assertEquals("degree",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("°").toString());
    assertEquals("copyright",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("©").toString());
    assertEquals("registered",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("®").toString());
    assertEquals("trademark",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("™").toString());
  }

  @Test
  void testEmbeddedAmpersandsAreLeftAlone() {
    // Expanding inside a token would invent a word that appears in neither
    // the document nor the query.
    assertEquals("R&D",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("R&D").toString());
    assertEquals("AT&T",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("AT&T").toString());
    assertEquals("TSR®",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("TSR®").toString());
  }

  @Test
  void testOrdinaryWordsPassThrough() {
    assertEquals("and",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("and").toString());
    assertEquals("court",
        SymbolJoinerCharSequenceNormalizer.getInstance().normalize("court").toString());
  }

  @Test
  void testNullIsRejected() {
    assertThrows(IllegalArgumentException.class,
        () -> SymbolJoinerCharSequenceNormalizer.getInstance().normalize(null));
  }
}
