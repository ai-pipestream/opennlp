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

package opennlp.tools.chunker;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.Map;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.ml.maxent.GISModel;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.parser.ParserChunkerFactory;

/**
 * This is the test class for {@link ChunkerModel}.
 */
public class ChunkerModelTest {

  @Test
  void testInvalidFactorySignature() throws Exception {

    ChunkerModel model = null;
    try {
      model = new ChunkerModel(this.getClass().getResourceAsStream("chunker170custom.bin"));
    } catch (IllegalArgumentException e) {
      Assertions.assertTrue(
          e.getMessage().contains("ChunkerFactory"), "Exception must state ChunkerFactory");
      Assertions.assertTrue(
          e.getMessage().contains("opennlp.tools.chunker.DummyChunkerFactory"),
          "Exception must mention DummyChunkerFactory");
    }
    Assertions.assertNull(model);
  }

  @Test
  void test170DefaultFactory() throws Exception {

    // This is an OpenNLP 1.x model. It should load with OpenNLP 2.x.
    Assertions.assertNotNull(
        new ChunkerModel(this.getClass().getResourceAsStream("chunker170default.bin")));

  }

  @Test
  void test180CustomFactory() throws Exception {

    // This is an OpenNLP 1.x model. It should load with OpenNLP 2.x.
    Assertions.assertNotNull(
        new ChunkerModel(this.getClass().getResourceAsStream("chunker180custom.bin")));

  }

  /**
   * The preview line stamps models with its own 0.x version. A legacy parser model's chunker is
   * rebuilt at load time with that stamp and {@link ParserChunkerFactory}, so the pre-1.8
   * factory rule must apply to major 1 only, while a 1.x model with a custom factory still fails.
   */
  @Test
  void testPreviewVersionWithCustomFactoryLoads() throws Exception {
    final MaxentModel maxent = new GISModel(
        new Context[] {new Context(new int[] {0, 1}, new double[] {0.5, 0.5})},
        new String[] {"p"}, new String[] {"B-NP", "O"});

    final ChunkerModel preview = new ChunkerModel("en", maxent,
        Map.of("OpenNLP-Version", "0.1.0"), new ParserChunkerFactory());
    Assertions.assertEquals(0, preview.getVersion().getMajor());
    final ByteArrayOutputStream out = new ByteArrayOutputStream();
    preview.serialize(out);
    Assertions.assertNotNull(new ChunkerModel(new ByteArrayInputStream(out.toByteArray())));

    final IllegalArgumentException e = Assertions.assertThrows(IllegalArgumentException.class,
        () -> new ChunkerModel("en", maxent, Map.of("OpenNLP-Version", "1.5.0"),
            new ParserChunkerFactory()));
    Assertions.assertTrue(e.getMessage().contains("no longer compatible"), e.getMessage());
  }
}
