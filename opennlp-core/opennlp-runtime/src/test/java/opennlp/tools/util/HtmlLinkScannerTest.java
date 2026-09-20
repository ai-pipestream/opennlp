/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
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
package opennlp.tools.util;

import java.util.List;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class HtmlLinkScannerTest {

  @Test
  void testReadsRealisticDirectoryIndexInSourceOrder() {
    String html = "<!doctype html><html><head><title>Models</title></head><body>"
        + "<h1>Index of /models</h1><pre>"
        + "<a href='../'>Parent Directory</a>"
        + "<a href=opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin>English tokenizer</a>"
        + "<a class=download HREF = \"opennlp-de-ud-gsd-pos-1.3-2.5.4.bin\""
        + " data-size='12345'>German POS</a>"
        + "<a href='checksums.txt'>checksums</a>"
        + "</pre></body></html>";

    Assertions.assertEquals(List.of("../", "opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin",
        "opennlp-de-ud-gsd-pos-1.3-2.5.4.bin", "checksums.txt"),
        HtmlLinkScanner.links(html));
  }

  @Test
  void testUsesFirstHrefAndDecodesRetainedAttributeValue() {
    String html = "<A HREF='first&amp;&NotEqualTilde;.bin' href='second.bin'>one</A>"
        + "<a href>empty</a><a href='a\r\nb\0c'>preprocessed</a>";

    Assertions.assertEquals(List.of("first&\u2242\u0338.bin", "", "a\nb\ufffdc"),
        HtmlLinkScanner.links(html));
  }

  @Test
  void testRecognizesOnlyAsciiTagSyntaxAndWhitespace() {
    String html = "<A\tHREF\n=\f'one'>one</A>"
        + "<a\u00a0href='not-an-attribute'>ignored</a>"
        + "<\u00e5 href='not-an-ascii-tag'>ignored</\u00e5>"
        + "<a h\u0157ef='not-href' href='two'>two</a>";

    Assertions.assertEquals(List.of("one", "two"), HtmlLinkScanner.links(html));
  }

  @Test
  void testIgnoresMarkupInsideCommentsRawTextAndPlaintext() {
    String fake = "<a href='fake'>fake</a>";
    String html = "<!-- " + fake + " -->"
        + "<script>" + fake + "</script><style>" + fake + "</style>"
        + "<textarea>" + fake + "</textarea><title>" + fake + "</title>"
        + "<xmp>" + fake + "</xmp><iframe>" + fake + "</iframe>"
        + "<noembed>" + fake + "</noembed><noframes>" + fake + "</noframes>"
        + "<a href='real'>real</a><plaintext>" + fake;

    Assertions.assertEquals(List.of("real"), HtmlLinkScanner.links(html));
  }

  @Test
  void testHandlesEscapedAndDoubleEscapedScriptText() {
    String fake = "<a href='fake'>fake</a>";
    String html = "<script><!--<script>" + fake + "</script>" + fake + "--></script>"
        + "<a href='real'>real</a>";

    Assertions.assertEquals(List.of("real"), HtmlLinkScanner.links(html));
  }

  @Test
  void testScriptEscapeCanCloseImmediatelyAfterOpeningDashes() {
    for (String escape : List.of("<!-->", "<!--->", "<!-- -->")) {
      Assertions.assertEquals(List.of("real"), HtmlLinkScanner.links(
          "<script>" + escape + "<script></script><a href=real>"), escape);
    }
  }

  @Test
  void testAbruptAndEofCommentsDoNotInventLinks() {
    Assertions.assertAll(
        () -> Assertions.assertEquals(List.of("one"),
            HtmlLinkScanner.links("<!--><a href=one>")),
        () -> Assertions.assertEquals(List.of("one"),
            HtmlLinkScanner.links("<!---><a href=one>")),
        () -> Assertions.assertEquals(List.of("one"),
            HtmlLinkScanner.links("<!-- --!><a href=one>")),
        () -> Assertions.assertEquals(List.of(),
            HtmlLinkScanner.links("<!-- <a href=fake>")));
  }

  @Test
  void testMalformedEofDoesNotEmitIncompleteStartTag() {
    Assertions.assertAll(
        () -> Assertions.assertEquals(List.of(), HtmlLinkScanner.links("<a")),
        () -> Assertions.assertEquals(List.of(), HtmlLinkScanner.links("<a href")),
        () -> Assertions.assertEquals(List.of(), HtmlLinkScanner.links("<a href=")),
        () -> Assertions.assertEquals(List.of(), HtmlLinkScanner.links("<a href='value")),
        () -> Assertions.assertEquals(List.of("complete"),
            HtmlLinkScanner.links("<a href=complete><a href='unfinished")));
  }

  @Test
  void testForeignContentSuppressesAnchorsExceptAtHtmlIntegrationPoints() {
    String html = "<svg><a href='svg-fake'/><![CDATA[<a href='cdata-fake'>]]>"
        + "<desc><a href='svg-desc-real'>real</a></desc>"
        + "<foreignObject><a href='foreign-object-real'>real</a></foreignObject></svg>"
        + "<math><a href='math-fake'/><mtext><a href='math-text-real'>real</a></mtext>"
        + "<annotation-xml encoding='application/xhtml+xml'>"
        + "<a href='annotation-real'>real</a></annotation-xml></math>";

    Assertions.assertEquals(List.of("svg-desc-real", "foreign-object-real", "math-text-real",
        "annotation-real"), HtmlLinkScanner.links(html));
  }

  @Test
  void testForeignContentBreakoutReturnsToHtmlTokenization() {
    String html = "<svg><g><p><a href='after-breakout'>real</a>"
        + "<svg><font color=red><a href='after-font-breakout'>real</a>";

    Assertions.assertEquals(List.of("after-breakout", "after-font-breakout"),
        HtmlLinkScanner.links(html));
  }

  @Test
  void testLexicalExtractionDoesNotApplyTreeBuilderRepair() {
    String html = "<select><a href='ignored-by-select-tree-building'>source link</a></select>"
        + "<table><a href='foster-parented-by-tree-building'>source link</a></table>"
        + "<b><i><a href='adoption-agency-input'>source link</a></b></i>";

    Assertions.assertEquals(List.of("ignored-by-select-tree-building",
        "foster-parented-by-tree-building", "adoption-agency-input"),
        HtmlLinkScanner.links(html));
  }

  @Test
  void testBogusDeclarationsAndHtmlCdataTextDoNotExposeNestedMarkup() {
    String html = "<!bogus <a href='declaration-fake'>>"
        + "<![CDATA[<a href='html-cdata-fake'>]]>"
        + "<?processing <a href='processing-fake'>>"
        + "<a href='real'>real</a>";

    Assertions.assertEquals(List.of("real"), HtmlLinkScanner.links(html));
  }

  @Test
  void testDeepUnmatchedCloseInputRemainsIterative() {
    String html = "<svg>" + "<g>".repeat(20_000) + "</unknown>".repeat(100_000)
        + "</g>".repeat(20_000) + "</svg><a href='last'>last</a>";

    Assertions.assertEquals(List.of("last"), HtmlLinkScanner.links(html));
  }

  @Test
  void testRawTextNameBoundariesAndSelfClosingSyntax() {
    Assertions.assertEquals(List.of("first", "last"), HtmlLinkScanner.links(
        "<style123><a href=first></style123><style/>"
            + "</style123><a href=fake></STYLE ><a href=last>"));
    Assertions.assertEquals(List.of("path/", "empty"),
        HtmlLinkScanner.links("<a href=path/><a href='empty'/>"));
  }

  @Test
  void testRejectsNull() {
    Assertions.assertThrows(IllegalArgumentException.class, () -> HtmlLinkScanner.links(null));
  }
}
