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

import org.junit.jupiter.api.Test;

import opennlp.tools.document.Document;
import opennlp.tools.util.Span;
import opennlp.tools.util.normalizer.AlignedText;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Runs the manual's asset examples (docbkx {@code assets.xml}) verbatim: every value
 * the chapter states is asserted here, so a change breaking this test breaks the
 * manual.
 */
public class AssetsManualExampleTest {

  /** The chapter's example text, with its printed base64 literal. */
  private static final String TEXT = "An image data:image/png;base64,"
      + "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAHCAYAAAAAAAAAAAAAAAAAAAAAAAAA"
      + " follows.";

  @Test
  void testDetectionExampleStatesTheAssetExactly() {
    final EmbeddedAsset asset = new CursorAssetDetector().detect(TEXT).get(0);
    assertEquals("png", asset.format());
    assertEquals("image/png", asset.mediaType());
    assertEquals(5, asset.width());
    assertEquals(7, asset.height());
    assertEquals(45, asset.decodedLength());
    assertEquals(45, asset.decode(TEXT).length);
    assertEquals((byte) 0x89, asset.decode(TEXT)[0]);
  }

  @Test
  void testFoldExampleProducesTheStatedTextAndMapsBack() {
    final Document document = new AssetAnnotator().annotate(Document.of(TEXT));
    final AlignedText folded = AssetFolder.fold(document, AssetFolder.caption());
    assertEquals("An image [png 5x7, 45 bytes] follows.", folded.normalizedString());

    final Span original = folded.toOriginalSpan(9, 28);
    assertEquals(document.get(AssetAnnotator.ASSETS).get(0).span().getStart(),
        original.getStart());
    assertEquals(document.get(AssetAnnotator.ASSETS).get(0).span().getEnd(),
        original.getEnd());
  }

  @Test
  void testDescriberExampleComposes() {
    final Document document = new AssetAnnotator().annotate(Document.of(TEXT));
    final BinaryContentDescriber describer =
        (content, mediaType) -> "described " + mediaType;
    final AlignedText described = AssetFolder.fold(document, annotation ->
        describer.describe(annotation.value().decode(TEXT),
            annotation.value().mediaType()));
    assertEquals("An image described image/png follows.", described.normalizedString());
  }
}
