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

package opennlp.tools.artifacts;

import java.util.List;

import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Runs the manual's artifact examples (docbkx {@code artifacts.xml}) verbatim: every
 * value the chapter states is asserted here, so a change breaking this test breaks the
 * manual. The example strings are built from code points; they equal the literals the
 * chapter prints.
 */
public class ArtifactsManualExampleTest {

  /** The chapter's example text: {@code cafÃ© costs �8} built from code points. */
  private static final String TEXT =
      "caf" + new String(new int[] {0x00C3, 0x00A9}, 0, 2)
          + " costs " + new String(new int[] {0xFFFD}, 0, 1) + "8";

  /** The detection example: two findings with the printed types and offsets. */
  @Test
  void testDetectionExamplePrintsTheStatedLines() {
    final ArtifactDetector detector = new CursorArtifactDetector();
    final List<TextArtifact> artifacts = detector.detect(TEXT);
    assertEquals(2, artifacts.size());
    assertEquals("mojibake [3..5)",
        artifacts.get(0).type() + " " + artifacts.get(0).span());
    assertEquals("replacement [12..13)",
        artifacts.get(1).type() + " " + artifacts.get(1).span());
  }

  /** The chapter's contrast: an ordinarily accented text yields no finding at all. */
  @Test
  void testAccentedTextYieldsNothing() {
    final String dejaVu = "d" + new String(new int[] {0x00E9}, 0, 1) + "j"
        + new String(new int[] {0x00E0}, 0, 1) + " vu";
    assertEquals(List.of(), new CursorArtifactDetector().detect(dejaVu));
  }

  /** The layer example: two annotations, the first covering the damaged word part. */
  @Test
  void testLayerExampleCarriesTheStatedAnnotations() {
    final Document document = new ArtifactAnnotator().annotate(Document.of(TEXT));
    final List<Annotation<TextArtifact>> artifacts =
        document.get(ArtifactAnnotator.ARTIFACTS);
    assertEquals(2, artifacts.size());
    assertEquals(new String(new int[] {0x00C3, 0x00A9}, 0, 2),
        artifacts.get(0).span().getCoveredText(TEXT).toString());
    assertEquals(TextArtifact.TYPE_MOJIBAKE, artifacts.get(0).value().type());
  }
}
