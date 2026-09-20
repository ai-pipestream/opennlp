/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
package opennlp.tools.util;

import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.TreeMap;

import opennlp.tools.models.ModelType;

final class DownloadLinkBenchmarkFixture implements AutoCloseable {

  private final String workload;
  private final Path index;
  private final URL indexUrl;

  private DownloadLinkBenchmarkFixture(String workload, Path index, URL indexUrl) {
    this.workload = workload;
    this.index = index;
    this.indexUrl = indexUrl;
  }

  static DownloadLinkBenchmarkFixture create(String workload) throws Exception {
    Path index = Files.createTempFile("opennlp-model-index", ".html");
    Files.writeString(index, switch (workload) {
      case "normal" -> normalIndex(32);
      case "markupHeavy" -> markupHeavyIndex(32);
      case "longIndex" -> normalIndex(2048);
      default -> throw new IllegalArgumentException("Unknown workload: " + workload);
    }, StandardCharsets.UTF_8);
    return new DownloadLinkBenchmarkFixture(workload, index, index.toUri().toURL());
  }

  URL indexUrl() {
    return indexUrl;
  }

  void validateParsedModels(Map<String, Map<ModelType, URL>> parsed) {
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
          || !tokenizer.toString().endsWith("opennlp-en-ud-ewt-tokens-31&copy.bin")) {
        throw new IllegalStateException("Unexpected markup-heavy model map: " + parsed);
      }
    }
  }

  static String fingerprint(Map<String, Map<ModelType, URL>> parsed) {
    TreeMap<String, String> values = new TreeMap<>();
    parsed.forEach((language, models) -> models.forEach((type, url) ->
        values.put(language + '/' + type.getName(), url.toString())));
    return values.toString();
  }

  @Override
  public void close() throws Exception {
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
}
