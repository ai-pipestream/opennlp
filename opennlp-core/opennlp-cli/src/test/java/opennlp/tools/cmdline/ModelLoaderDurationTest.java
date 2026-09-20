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
package opennlp.tools.cmdline;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import nl.altindag.log.LogCaptor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelLoaderDurationTest {

  @Test
  void loadingDurationUsesStableDecimalNotation(@TempDir Path directory) throws IOException {
    Path input = Files.writeString(directory.resolve("test.model"), "content");
    ModelLoader<String> loader = new ModelLoader<>("test") {
      @Override
      protected String loadModel(InputStream stream) throws IOException {
        return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
      }
    };
    Locale previous = Locale.getDefault(Locale.Category.FORMAT);
    try (LogCaptor logs = LogCaptor.forClass(ModelLoader.class)) {
      logs.setLogLevelToInfo();
      Locale.setDefault(Locale.Category.FORMAT, Locale.GERMANY);
      assertEquals("content", loader.load(input.toFile()));
      assertTrue(logs.getInfoLogs().stream().anyMatch(
          message -> message.matches("done \\([0-9]+\\.[0-9]{3}s\\)\\n")),
          () -> "Expected fixed decimal seconds, got " + logs.getInfoLogs());
    } finally {
      Locale.setDefault(Locale.Category.FORMAT, previous);
    }
  }
}
