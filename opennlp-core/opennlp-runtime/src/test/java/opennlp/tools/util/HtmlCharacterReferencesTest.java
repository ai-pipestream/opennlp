/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package opennlp.tools.util;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class HtmlCharacterReferencesTest {

  @Test
  void decodesNamedReferencesWithAttributeRules() {
    assertEquals("& © ∉", HtmlCharacterReferences.decodeAttribute("&amp; &copy; &notin;"));
    assertEquals("Æ Æ", HtmlCharacterReferences.decodeAttribute("&AElig &AElig;"));
    assertEquals("&notit; &amp=", HtmlCharacterReferences.decodeAttribute("&notit; &amp="));
    assertEquals("&not=", HtmlCharacterReferences.decodeAttribute("&not="));
    assertEquals("¬ ", HtmlCharacterReferences.decodeAttribute("&not "));
    assertEquals("⪢̸", HtmlCharacterReferences.decodeAttribute("&NotNestedGreaterGreater;"));
    assertEquals("𝔄", HtmlCharacterReferences.decodeAttribute("&Afr;"));
  }

  @Test
  void decodesNumericReferencesUsingHtmlReplacementRules() {
    assertEquals("AA😀", HtmlCharacterReferences.decodeAttribute("&#65;&#x41;&#x1F600;"));
    assertEquals("���", HtmlCharacterReferences.decodeAttribute("&#0;&#xD800;&#x110000;"));
    assertEquals("€‚Ÿ", HtmlCharacterReferences.decodeAttribute("&#x80;&#130;&#159;"));
    assertEquals("&#x; &#١;", HtmlCharacterReferences.decodeAttribute("&#x; &#١;"));
  }

  @Test
  void decodesOnceAndPreservesUnchangedIdentity() {
    assertEquals("&amp;", HtmlCharacterReferences.decodeAttribute("&amp;amp;"));
    String unchanged = "plain &unknown; text";
    assertSame(unchanged, HtmlCharacterReferences.decodeAttribute(unchanged));
    String plain = new String("plain");
    assertSame(plain, HtmlCharacterReferences.decodeAttribute(plain));
    String longUnknown = '&' + "a".repeat(100_000);
    assertSame(longUnknown, HtmlCharacterReferences.decodeAttribute(longUnknown));
    assertThrows(IllegalArgumentException.class, () -> HtmlCharacterReferences.decodeAttribute(null));
  }

  @Test
  void generatedTableMatchesEveryOfficialEntityVector() throws Exception {
    try (BufferedReader reader = new BufferedReader(new InputStreamReader(
        getClass().getResourceAsStream("/opennlp/tools/util/html-character-references.tsv"),
        StandardCharsets.UTF_8))) {
      int count = 0;
      String line;
      while ((line = reader.readLine()) != null) {
        int tab = line.indexOf('\t');
        String name = line.substring(0, tab);
        StringBuilder expected = new StringBuilder(2);
        int start = tab + 1;
        while (start < line.length()) {
          int comma = line.indexOf(',', start);
          expected.appendCodePoint(Integer.parseInt(line.substring(start, comma), 16));
          start = comma + 1;
        }
        String suffix = name.endsWith(";") ? "" : " ";
        assertEquals(expected + suffix, HtmlCharacterReferences.decodeAttribute('&' + name + suffix), name);
        count++;
      }
      assertEquals(2231, count);
    }
  }
}
