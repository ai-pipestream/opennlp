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

package opennlp.tools.noise;

import java.io.ByteArrayOutputStream;
import java.util.Base64;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Test;

import opennlp.tools.assets.AssetAnnotator;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Pins the document adapter and the hand-off between the asset detector and the noise
 * scorer: an embedded binary the asset layer explains is never reported again as
 * noise.
 */
public class NoiseAnnotatorTest {

  @Test
  void testModesDeclareTheirRequirements() {
    assertEquals(Set.of(AssetAnnotator.ASSETS), new NoiseAnnotator().requires());
    assertEquals(Set.of(),
        new NoiseAnnotator(new StructuralNoiseScorer(), false).requires());
    assertEquals("opennlp:noise", NoiseAnnotator.NOISE.id());
  }

  @Test
  void testDetectedAssetsAreNotReportedAsNoise() {
    final ByteArrayOutputStream png = new ByteArrayOutputStream();
    png.writeBytes(new byte[] {(byte) 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A});
    png.writeBytes(new byte[] {0, 0, 0, 13});
    png.writeBytes(new byte[] {'I', 'H', 'D', 'R'});
    png.writeBytes(new byte[24]);
    final String encoded = Base64.getEncoder().encodeToString(png.toByteArray());
    final String text = "report zxkcvbnmsdfg here " + encoded;

    Document document = new AssetAnnotator().annotate(Document.of(text));
    document = new NoiseAnnotator().annotate(document);

    assertEquals(1, document.get(AssetAnnotator.ASSETS).size(),
        "the payload is explained by the asset layer");
    final List<Annotation<NoiseSpan>> noise = document.get(NoiseAnnotator.NOISE);
    assertEquals(1, noise.size(), "the asset span must not be double-reported");
    assertEquals(NoiseSpan.SEVERITY_GIBBERISH, noise.get(0).value().severity());
    assertEquals("zxkcvbnmsdfg",
        noise.get(0).span().getCoveredText(text).toString());
  }

  @Test
  void testStandaloneModeScoresTheWholeText() {
    final String text = "QWxhZGRpbjF2cGVuNHNlc2FtZQ here";
    final Document document = new NoiseAnnotator(new StructuralNoiseScorer(), false)
        .annotate(Document.of(text));
    final List<Annotation<NoiseSpan>> noise = document.get(NoiseAnnotator.NOISE);
    assertEquals(1, noise.size());
    assertEquals(NoiseSpan.SEVERITY_BINARYISH, noise.get(0).value().severity());
  }

  @Test
  void testRejectsContractViolations() {
    assertThrows(IllegalArgumentException.class, () -> new NoiseAnnotator(null, true));
    assertThrows(IllegalArgumentException.class,
        () -> new NoiseAnnotator().annotate(null));
  }
}
