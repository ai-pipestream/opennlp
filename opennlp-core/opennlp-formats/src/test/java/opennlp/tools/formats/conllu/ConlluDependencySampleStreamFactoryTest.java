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

package opennlp.tools.formats.conllu;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.tools.cmdline.TerminateToolException;
import opennlp.tools.depparse.DependencySample;
import opennlp.tools.util.ObjectStream;

import static opennlp.tools.formats.conllu.ConlluTestLines.line;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** Tests the UTF-8 requirement of the dependency sample stream factory. */
public class ConlluDependencySampleStreamFactoryTest {

  private static final String TOKEN = "café";

  @TempDir
  private Path directory;

  private Path data;

  private final ConlluDependencySampleStreamFactory factory =
      new ConlluDependencySampleStreamFactory(ConlluDependencySampleStreamFactory.Parameters.class);

  @BeforeEach
  void writeData() throws IOException {
    data = directory.resolve("sample.conllu");
    Files.writeString(data, line("1", TOKEN, TOKEN, "NOUN", "NN", "_", "0", "root", "_", "_") + "\n");
  }

  @Test
  void testDefaultEncodingReadsUtf8() throws IOException {
    try (ObjectStream<DependencySample> samples = factory.create(new String[] {"-data", data.toString()})) {
      assertArrayEquals(new String[] {TOKEN}, samples.read().getTokens());
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"UTF-8", "UTF8", "utf-8"})
  void testUtf8AliasesAreAccepted(String encoding) throws IOException {
    try (ObjectStream<DependencySample> samples = factory.create(new String[] {
        "-data", data.toString(), "-encoding", encoding})) {
      assertArrayEquals(new String[] {TOKEN}, samples.read().getTokens());
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"ISO-8859-1", "UTF-16", "US-ASCII"})
  void testOtherEncodingsAreRejected(String encoding) {
    final TerminateToolException error = assertThrows(TerminateToolException.class, () -> {
      try (ObjectStream<DependencySample> ignored = factory.create(new String[] {
          "-data", data.toString(), "-encoding", encoding})) {
        // Close the stream if an unsupported encoding is incorrectly accepted.
      }
    });
    assertEquals("CoNLL-U data must use UTF-8", error.getMessage());
  }
}
