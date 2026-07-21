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
import java.util.Set;

import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Pins the document adapter around the built-in detector. */
public class ArtifactAnnotatorTest {

  @Test
  void testProvidesTheArtifactsLayer() {
    final ArtifactAnnotator annotator = new ArtifactAnnotator();
    assertEquals(Set.of(ArtifactAnnotator.ARTIFACTS), annotator.provides());
    assertEquals("opennlp:artifacts", ArtifactAnnotator.ARTIFACTS.id());
  }

  @Test
  void testAnnotatesFindingsWithExactSpans() {
    final String mojibake = new String(new int[] {0x00C3, 0x00A9}, 0, 2);
    final Document document = new ArtifactAnnotator()
        .annotate(Document.of("caf" + mojibake + " time"));
    final List<Annotation<TextArtifact>> artifacts =
        document.get(ArtifactAnnotator.ARTIFACTS);
    assertEquals(1, artifacts.size());
    assertEquals(TextArtifact.TYPE_MOJIBAKE, artifacts.get(0).value().type());
    assertEquals(mojibake,
        artifacts.get(0).span().getCoveredText(document.text()).toString());
  }

  @Test
  void testCleanDocumentGetsAnEmptyLayer() {
    final Document document = new ArtifactAnnotator().annotate(Document.of("all clean"));
    assertTrue(document.get(ArtifactAnnotator.ARTIFACTS).isEmpty());
  }

  @Test
  void testRejectsContractViolations() {
    assertThrows(IllegalArgumentException.class, () -> new ArtifactAnnotator(null));
    assertThrows(IllegalArgumentException.class,
        () -> new ArtifactAnnotator().annotate(null));
  }
}
