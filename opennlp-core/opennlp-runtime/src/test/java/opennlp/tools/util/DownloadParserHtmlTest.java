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

import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import opennlp.tools.models.ModelType;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DownloadParserHtmlTest {

  @TempDir
  Path tempDir;

  @Test
  void testReadsHtml32AnchorAttributeForms() throws Exception {
    String html = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">"
        + "<A data-x='1' HREF = \"opennlp-en-ud-ewt-tokens-double.bin\">one</A>"
        + "<a class=download href='opennlp-fr-ud-gsd-sentence-single.bin'>two</a>"
        + "<a title='<style>' href=opennlp-de-ud-gsd-pos-unquoted.bin disabled>three</a>";

    Map<String, Map<ModelType, URL>> models = parse(html);

    assertUrlEndsWith(models, "en", ModelType.TOKENIZER,
        "opennlp-en-ud-ewt-tokens-double.bin");
    assertUrlEndsWith(models, "fr", ModelType.SENTENCE_DETECTOR,
        "opennlp-fr-ud-gsd-sentence-single.bin");
    assertUrlEndsWith(models, "de", ModelType.POS,
        "opennlp-de-ud-gsd-pos-unquoted.bin");
  }

  @Test
  void testDecodesHtml32AndNumericCharacterReferences() throws Exception {
    String html = "<a href='opennlp-en-ud-ewt-tokens-a&amp;b&copy;&#233;&#x1F600;.bin'>model</a>";

    Map<String, Map<ModelType, URL>> models = parse(html);

    assertUrlEndsWith(models, "en", ModelType.TOKENIZER,
        "opennlp-en-ud-ewt-tokens-a&b©é😀.bin");
  }

  @Test
  void testIgnoresAnchorsInCommentsAndTextElements() throws Exception {
    String fake = "<a href='opennlp-fr-ud-gsd-pos-fake.bin'>fake</a>";
    String html = "<!-- " + fake + " -->"
        + "<script>" + fake + "</script><style>" + fake + "</style>"
        + "<textarea>" + fake + "</textarea><title>" + fake + "</title>"
        + "<xmp>" + fake + "</xmp>"
        + "<a href='opennlp-en-ud-ewt-tokens-real.bin'>real</a>"
        + "<plaintext>" + fake
        + "<a href='opennlp-de-ud-gsd-pos-after-plaintext.bin'>also text</a>";

    Map<String, Map<ModelType, URL>> models = parse(html);
    assertUrlEndsWith(models, "en", ModelType.TOKENIZER,
        "opennlp-en-ud-ewt-tokens-real.bin");
    assertEquals(1, models.size());
  }

  @Test
  void testIgnoresAnchorsInBogusDeclarations() throws Exception {
    String html = "<!bogus <a href='opennlp-fr-ud-gsd-pos-fake.bin'>>"
        + "<![CDATA[<a href='opennlp-de-ud-gsd-pos-fake.bin'>]]>"
        + "<a href='opennlp-en-ud-ewt-tokens-real.bin'>real</a>";

    Map<String, Map<ModelType, URL>> models = parse(html);
    assertUrlEndsWith(models, "en", ModelType.TOKENIZER,
        "opennlp-en-ud-ewt-tokens-real.bin");
    assertEquals(1, models.size());
  }

  @Test
  void testMalformedMarkupTerminatesWithoutInventingModels() throws Exception {
    String html = "<a href='unterminated" + "x".repeat(200_000)
        + "<!-- <a href='opennlp-en-ud-ewt-tokens-comment.bin'>"
        + "<script><a href='opennlp-fr-ud-gsd-pos-script.bin'>";

    assertTrue(parse(html).isEmpty());
  }

  @Test
  void testTextElementNameRequiresExactBoundary() throws Exception {
    String html = "<style123>not a style element</style123>"
        + "<a href='opennlp-en-ud-ewt-tokens-real.bin'>real</a>";
    assertUrlEndsWith(parse(html), "en", ModelType.TOKENIZER,
        "opennlp-en-ud-ewt-tokens-real.bin");
  }

  @Test
  void testHtml5CharacterReferenceInAttribute() throws Exception {
    String html = "<a href='opennlp-en-ud-ewt-tokens-&NotEqualTilde;.bin'>model</a>";
    assertUrlEndsWith(parse(html), "en", ModelType.TOKENIZER,
        "opennlp-en-ud-ewt-tokens-\u2242\u0338.bin");
  }

  private Map<String, Map<ModelType, URL>> parse(String html) throws Exception {
    Path index = tempDir.resolve("index.html");
    Files.writeString(index, html, StandardCharsets.UTF_8);
    return new DownloadUtil.DownloadParser(index.toUri().toURL()).getAvailableModels();
  }

  private static void assertUrlEndsWith(Map<String, Map<ModelType, URL>> models,
      String language, ModelType type, String expectedFilename) {
    URL url = models.get(language).get(type);
    assertEquals(expectedFilename, url.toString().substring(url.toString().lastIndexOf('/') + 1));
  }
}
