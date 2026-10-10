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

package opennlp.examples.chapters;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.HexFormat;
import java.util.Set;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.examples.DocExampleSupport;
import opennlp.tools.models.ClassPathModel;
import opennlp.tools.models.ClassPathModelEntry;
import opennlp.tools.models.ClassPathModelLoader;
import opennlp.tools.models.ClassPathModelProvider;
import opennlp.tools.models.DefaultClassPathModelProvider;
import opennlp.tools.models.classgraph.ClassgraphModelFinder;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.util.ResourceInstaller;

/**
 * The samples printed in the Model Loading chapter of the manual.
 * <p>
 * The published sentence and part-of-speech model jars are on the test classpath, so the finder
 * samples find real models. No jar named {@code custom-opennlp-model.jar} exists, so the bundling
 * sample runs and finds nothing. The installer samples install the small gzip-compressed tar
 * archive {@code corpus.tar.gz} from the test resources.
 */
class ModelLoadingChapterTest extends DocExampleSupport {

  @Test
  void findsAndLoadsModels() throws IOException {
    // docs:begin example.modelloading.load
    final ClassgraphModelFinder finder = new ClassgraphModelFinder();
    // or use: new SimpleClassPathModelFinder()
    final ClassPathModelLoader loader = new ClassPathModelLoader();
    final Set<ClassPathModelEntry> models = finder.findModels(false);
    for (ClassPathModelEntry entry : models) {
      final ClassPathModel model = loader.load(entry);
      if (model != null) {
        System.out.println(model.getModelName());
        System.out.println(model.getModelSHA256());
        System.out.println(model.getModelVersion());
        System.out.println(model.getModelLanguage());
        // do something with the model by consuming the byte array
      }
    }
    // docs:end
    Assertions.assertFalse(models.isEmpty());
  }

  @Test
  void loadsThroughAProvider() throws IOException {
    final ClassgraphModelFinder finder = new ClassgraphModelFinder();
    final ClassPathModelLoader loader = new ClassPathModelLoader();
    // docs:begin example.modelloading.load.provider
    final ClassPathModelProvider provider = new DefaultClassPathModelProvider(finder, loader);
    // Here: SentenceModel, other model types accordingly
    final SentenceModel sm = provider.load("en",
        opennlp.tools.models.ModelType.SENTENCE_DETECTOR, SentenceModel.class);
    if (sm != null) {
      // do something with the (sentence) model
    }
    // docs:end
    Assertions.assertNotNull(sm);
  }

  @Test
  void findsABundledModel() throws IOException {
    // docs:begin example.modelloading.bundle
    final ClassgraphModelFinder finder = new ClassgraphModelFinder("custom-opennlp-model.jar");
    // or use: new SimpleClassPathModelFinder("custom-opennlp-model.jar")
    final ClassPathModelLoader loader = new ClassPathModelLoader();
    final Set<ClassPathModelEntry> models = finder.findModels(false);
    for (ClassPathModelEntry entry : models) {
      final ClassPathModel model = loader.load(entry);
      if (model != null) {
        System.out.println(model.getModelName());
        System.out.println(model.getModelSHA256());
        System.out.println(model.getModelVersion());
        System.out.println(model.getModelLanguage());
        // do something with the model by consuming the byte array
      }
    }
    // docs:end
    Assertions.assertTrue(models.isEmpty());
  }

  @Test
  void installsAnArchive() throws IOException {
    Path corpusTarGz = corpus("corpus.tar.gz");
    Path targetDirectory = corpusTarGz.resolveSibling("installed");
    String expectedSha256 = sha256(corpusTarGz);
    // docs:begin example.modelloading.resourceinstaller
    Path result = ResourceInstaller.install(
        corpusTarGz.toUri(), targetDirectory, expectedSha256);
    // result is targetDirectory; the archived files appear under their archive paths
    // docs:end
    Assertions.assertEquals(targetDirectory, result);
    Assertions.assertEquals("hello corpus",
        Files.readString(targetDirectory.resolve("corpus/doc.txt"), StandardCharsets.UTF_8));
  }

  @Test
  void installsWithinLimits() throws IOException {
    Path corpusTarGz = corpus("corpus-limited.tar.gz");
    Path targetDirectory = corpusTarGz.resolveSibling("installed");
    String expectedSha256 = sha256(corpusTarGz);
    // docs:begin example.modelloading.resourceinstaller.limits
    ResourceInstaller.Limits limits = ResourceInstaller.Limits.builder()
        .readTimeout(Duration.ofSeconds(10))
        .maxDownloadBytes(1024L * 1024)   // download limit in bytes
        .maxExpandedBytes(1024L * 1024)   // expansion limit in bytes
        .maxExpansionRatio(500)           // expanded bytes per compressed byte
        .build();
    Path result = ResourceInstaller.install(
        corpusTarGz.toUri(), targetDirectory, expectedSha256, limits);
    // docs:end
    Assertions.assertTrue(Files.isRegularFile(result.resolve("corpus/doc.txt")));
  }

  /* Copies the checked-in archive holding corpus/doc.txt into a fresh directory. */
  private static Path corpus(String fileName) throws IOException {
    final Path dir = Files.createTempDirectory(workFile(""), "corpus");
    try (InputStream in = ModelLoadingChapterTest.class.getResourceAsStream(
        "/opennlp/examples/corpus.tar.gz")) {
      Files.copy(in, dir.resolve(fileName));
    }
    return dir.resolve(fileName);
  }

  private static String sha256(Path file) throws IOException {
    try {
      return HexFormat.of().formatHex(
          MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file)));
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException(e);
    }
  }
}
