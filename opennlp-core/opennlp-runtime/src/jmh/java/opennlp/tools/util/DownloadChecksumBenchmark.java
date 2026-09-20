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

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

import opennlp.tools.chunker.ChunkerModel;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@State(Scope.Benchmark)
public class DownloadChecksumBenchmark {

  private static final String DOWNLOAD_HOME_PROPERTY = "OPENNLP_DOWNLOAD_HOME";
  private Path temporaryDirectory;
  private URL modelUrl;
  private String previousDownloadHome;

  @Setup(Level.Trial)
  public void setUp() throws IOException {
    previousDownloadHome = System.getProperty(DOWNLOAD_HOME_PROPERTY);
    temporaryDirectory = Files.createTempDirectory("opennlp-checksum-benchmark");
    System.setProperty(DOWNLOAD_HOME_PROPERTY, temporaryDirectory.toString());
    Path remote = Files.createDirectories(temporaryDirectory.resolve("remote"));
    Path model = remote.resolve("opennlp-test-chunker.bin");
    try (InputStream input = DownloadChecksumBenchmark.class
        .getResourceAsStream("/opennlp/tools/chunker/chunker170default.bin")) {
      if (input == null) {
        throw new IOException("Missing benchmark model resource");
      }
      Files.copy(input, model);
    }
    Files.writeString(model.resolveSibling(model.getFileName() + ".sha512"),
        sha512(model) + "  " + model.getFileName(), StandardCharsets.UTF_8);
    modelUrl = model.toUri().toURL();
    DownloadUtil.downloadModel(modelUrl, ChunkerModel.class);
  }

  @Benchmark
  public ChunkerModel validateCachedModel() throws IOException {
    return DownloadUtil.downloadModel(modelUrl, ChunkerModel.class);
  }

  @TearDown(Level.Trial)
  public void tearDown() throws IOException {
    if (previousDownloadHome == null) {
      System.clearProperty(DOWNLOAD_HOME_PROPERTY);
    } else {
      System.setProperty(DOWNLOAD_HOME_PROPERTY, previousDownloadHome);
    }
    try (var paths = Files.walk(temporaryDirectory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
        Files.deleteIfExists(path);
      }
    }
  }

  private static String sha512(Path file) throws IOException {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-512");
      return HexFormat.of().formatHex(digest.digest(Files.readAllBytes(file)));
    } catch (NoSuchAlgorithmException e) {
      throw new IOException("SHA-512 algorithm not found", e);
    }
  }
}
