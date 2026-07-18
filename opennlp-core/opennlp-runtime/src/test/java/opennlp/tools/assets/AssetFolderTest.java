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

package opennlp.tools.assets;

import java.io.ByteArrayOutputStream;
import java.util.Base64;
import java.util.List;

import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.util.Span;
import opennlp.tools.util.normalizer.AlignedText;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Pins the fold: replacement text stands in for each asset, the surrounding text is
 * unchanged, and the alignment maps every folded span back to the exact original
 * offsets.
 */
public class AssetFolderTest {

  /**
   * Builds a document whose text embeds one real PNG payload between two sentences,
   * annotated by the real detector.
   *
   * @return The annotated document.
   */
  private static Document annotated() {
    final ByteArrayOutputStream png = new ByteArrayOutputStream();
    png.writeBytes(new byte[] {(byte) 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A});
    png.writeBytes(new byte[] {0, 0, 0, 13});
    png.writeBytes(new byte[] {'I', 'H', 'D', 'R'});
    png.writeBytes(new byte[] {0, 0, 0, 5, 0, 0, 0, 7});
    png.writeBytes(new byte[] {8, 6, 0, 0, 0});
    png.writeBytes(new byte[16]);
    final String encoded = Base64.getEncoder().encodeToString(png.toByteArray());
    return new AssetAnnotator().annotate(
        Document.of("Before the image. " + encoded + " After the image."));
  }

  @Test
  void testCaptionFoldKeepsSurroundingTextAndMapsOffsets() {
    final Document document = annotated();
    final AlignedText folded = AssetFolder.fold(document, AssetFolder.caption());
    final String result = folded.normalizedString();
    assertEquals("Before the image. [png 5x7, 45 bytes] After the image.", result);

    // The folded caption maps back to the asset's exact original span.
    final Annotation<EmbeddedAsset> asset = document.get(AssetAnnotator.ASSETS).get(0);
    final int captionStart = result.indexOf('[');
    final int captionEnd = result.indexOf(']') + 1;
    final Span original = folded.toOriginalSpan(captionStart, captionEnd);
    assertEquals(asset.span().getStart(), original.getStart());
    assertEquals(asset.span().getEnd(), original.getEnd());

    // Text after the fold maps back to itself, shifted by the length difference.
    final int afterStart = result.indexOf("After");
    final Span after = folded.toOriginalSpan(afterStart, afterStart + 5);
    assertEquals("After", after.getCoveredText(document.text()).toString());
  }

  @Test
  void testDescriberComposesThroughTheReplacementFunction() {
    final Document document = annotated();
    final BinaryContentDescriber describer = (content, mediaType) -> {
      assertEquals("image/png", mediaType);
      return "a " + content.length + " byte " + mediaType + " asset";
    };
    final AlignedText folded = AssetFolder.fold(document, annotation ->
        describer.describe(annotation.value().decode(document.text()),
            annotation.value().mediaType()));
    assertEquals("Before the image. a 45 byte image/png asset After the image.",
        folded.normalizedString());
  }

  @Test
  void testDocumentWithoutTheLayerFailsLoud() {
    assertThrows(IllegalArgumentException.class,
        () -> AssetFolder.fold(Document.of("no layer here"), AssetFolder.caption()));
  }

  @Test
  void testEmptyLayerFoldsToTheIdenticalText() {
    final Document document = new AssetAnnotator().annotate(Document.of("clean text"));
    final AlignedText folded = AssetFolder.fold(document, AssetFolder.caption());
    assertEquals("clean text", folded.normalizedString());
    assertEquals(List.of(), document.get(AssetAnnotator.ASSETS));
  }

  @Test
  void testRejectsContractViolations() {
    final Document document = annotated();
    assertThrows(IllegalArgumentException.class,
        () -> AssetFolder.fold(null, AssetFolder.caption()));
    assertThrows(IllegalArgumentException.class,
        () -> AssetFolder.fold(document, null));
    assertThrows(IllegalArgumentException.class,
        () -> AssetFolder.fold(document, annotation -> null));
  }
}
