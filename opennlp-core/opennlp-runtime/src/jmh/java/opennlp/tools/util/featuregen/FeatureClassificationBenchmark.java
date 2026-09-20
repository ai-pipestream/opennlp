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
package opennlp.tools.util.featuregen;

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
import org.openjdk.jmh.annotations.Warmup;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@State(Scope.Thread)
public class FeatureClassificationBenchmark {

  @Param({"ascii", "unicode", "mixedLong"})
  private String workload;

  private String token;

  @Setup(Level.Trial)
  public void setUp() {
    token = switch (workload) {
      case "ascii" -> "OpenNLP-2026";
      case "unicode" -> "École東京١٢٣";
      case "mixedLong" -> "Abç-１２/東京.".repeat(1024);
      default -> throw new IllegalArgumentException("Unknown workload: " + workload);
    };
    StringPattern recognized = StringPattern.recognize(token);
    String feature = FeatureGeneratorUtil.tokenFeature(token);
    if (recognized == null || feature == null || feature.isEmpty()) {
      throw new IllegalStateException("Feature classification returned no result");
    }
    System.out.println("FeatureClassificationBenchmark fingerprint " + workload + "="
        + fingerprint(recognized) + '|' + feature);
  }

  private static String fingerprint(StringPattern pattern) {
    return pattern.isAllLetter() + ":" + pattern.isInitialCapitalLetter() + ":"
        + pattern.isAllCapitalLetter() + ":" + pattern.isAllLowerCaseLetter() + ":"
        + pattern.isAllDigit() + ":" + pattern.isAllHiragana() + ":"
        + pattern.isAllKatakana() + ":" + pattern.digits() + ":"
        + pattern.containsPeriod() + ":" + pattern.containsComma() + ":"
        + pattern.containsSlash() + ":" + pattern.containsDigit() + ":"
        + pattern.containsHyphen() + ":" + pattern.containsLetters();
  }

  @Benchmark
  public StringPattern recognize() {
    return StringPattern.recognize(token);
  }

  @Benchmark
  public String tokenFeature() {
    return FeatureGeneratorUtil.tokenFeature(token);
  }
}
