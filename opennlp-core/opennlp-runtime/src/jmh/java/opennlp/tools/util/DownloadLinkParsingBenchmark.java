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

import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

import opennlp.tools.models.ModelType;

@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
public class DownloadLinkParsingBenchmark {

  @Param({"normal", "markupHeavy", "longIndex"})
  private String workload;

  private Path index;
  private URL indexUrl;

  @Setup(Level.Trial)
  public void setUp() throws Exception {
    index = Files.createTempFile("opennlp-model-index", ".html");
    Files.writeString(index, switch (workload) {
      case "normal" -> normalIndex(32);
      case "markupHeavy" -> markupHeavyIndex(32);
      case "longIndex" -> normalIndex(2048);
      default -> throw new IllegalArgumentException("Unknown workload: " + workload);
    }, StandardCharsets.UTF_8);
    indexUrl = index.toUri().toURL();
    Map<String, Map<ModelType, URL>> parsed =
        new DownloadUtil.DownloadParser(indexUrl).getAvailableModels();
    validateParsedModels(parsed);
    System.out.println("DownloadLinkParsingBenchmark fingerprint " + workload + "="
        + fingerprint(parsed));
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @Warmup(iterations = 3, time = 1)
  @Measurement(iterations = 5, time = 1)
  public Map<String, Map<ModelType, URL>> parseIndex() throws Exception {
    return new DownloadUtil.DownloadParser(indexUrl).getAvailableModels();
  }

  @Benchmark
  @BenchmarkMode(Mode.SingleShotTime)
  @Warmup(iterations = 0)
  @Measurement(iterations = 1)
  public Map<String, Map<ModelType, URL>> coldStart() throws Exception {
    return new DownloadUtil.DownloadParser(indexUrl).getAvailableModels();
  }

  @TearDown(Level.Trial)
  public void tearDown() throws Exception {
    Files.deleteIfExists(index);
  }

  private static String normalIndex(int links) {
    StringBuilder html = new StringBuilder(links * 100);
    html.append("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\"><html><body>");
    for (int i = 0; i < links; i++) {
      html.append("<a href=\"opennlp-en-ud-ewt-tokens-").append(i)
          .append(".bin\">model</a>");
    }
    return html.append("</body></html>").toString();
  }

  private static String markupHeavyIndex(int links) {
    StringBuilder html = new StringBuilder(links * 180);
    html.append("<html><head><title>models</title><style>a{color:red}</style></head><body>")
        .append("<!-- <a href='opennlp-fr-ud-gsd-pos-fake.bin'> -->")
        .append("<script>const fake=\"<a href='fake.bin'>\";</script>");
    for (int i = 0; i < links; i++) {
      html.append("<A data-i='").append(i).append("' HREF = 'opennlp-en-ud-ewt-tokens-")
          .append(i).append("&amp;copy.bin'>model</A>");
    }
    return html.append("</body></html>").toString();
  }

  private void validateParsedModels(Map<String, Map<ModelType, URL>> parsed) {
    if (!"markupHeavy".equals(workload)) {
      URL tokenizer = parsed.getOrDefault("en", Map.of()).get(ModelType.TOKENIZER);
      String expectedSuffix = "longIndex".equals(workload)
          ? "opennlp-en-ud-ewt-tokens-2047.bin" : "opennlp-en-ud-ewt-tokens-31.bin";
      if (parsed.size() != 1 || tokenizer == null || !tokenizer.toString().endsWith(expectedSuffix)) {
        throw new IllegalStateException("Unexpected parsed model map: " + parsed);
      }
      return;
    }
    if (parsed.containsKey("fr")) {
      throw new IllegalStateException("Parsed a link from an HTML comment: " + parsed);
    }
    if (!parsed.isEmpty()) {
      URL tokenizer = parsed.getOrDefault("en", Map.of()).get(ModelType.TOKENIZER);
      if (parsed.size() != 1 || tokenizer == null
          || !tokenizer.toString().contains("tokens-31&amp;copy.bin")) {
        throw new IllegalStateException("Unexpected markup-heavy model map: " + parsed);
      }
    }
  }

  private static String fingerprint(Map<String, Map<ModelType, URL>> parsed) {
    TreeMap<String, String> values = new TreeMap<>();
    parsed.forEach((language, models) -> models.forEach((type, url) ->
        values.put(language + '/' + type.getName(), url.toString())));
    return values.toString();
  }
}
