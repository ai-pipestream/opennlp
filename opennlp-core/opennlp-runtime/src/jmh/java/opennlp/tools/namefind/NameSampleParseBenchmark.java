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
package opennlp.tools.namefind;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.infra.Blackhole;

/** Measures the public {@link NameSample#parse(String, boolean)} parser. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class NameSampleParseBenchmark {

  @State(Scope.Benchmark)
  public static class InputState {

    @Param({"normalUnicode", "markupHeavy", "long16k", "long64k"})
    public String workload;

    String input;

    @Setup(Level.Trial)
    public void prepare() {
      input = switch (workload) {
        case "normalUnicode" -> normalUnicode();
        case "markupHeavy" -> repeated(markupHeavySentence(), 80);
        case "long16k" -> toMinimumLength(normalUnicode(), 16 * 1024);
        case "long64k" -> toMinimumLength(normalUnicode(), 64 * 1024);
        default -> throw new IllegalArgumentException("Unknown workload: " + workload);
      };
    }
  }

  @Benchmark
  public void parse(InputState state, Blackhole blackhole) throws IOException {
    blackhole.consume(NameSample.parse(state.input, false));
  }

  private static String normalUnicode() {
    return "Dr. <START:person> Chloé Dubois <END> met <START:organization> "
        + "Acme 東京 Labs <END> in <START:location> München <END> on Tuesday .";
  }

  private static String markupHeavySentence() {
    return "<START:person> Ana María <END> met <START:person> 李 雷 <END> at "
        + "<START:organization> Café Ω <END> in <START:location> Zürich <END> . ";
  }

  private static String toMinimumLength(String sentence, int minimumLength) {
    int repetitions = (minimumLength + sentence.length()) / (sentence.length() + 1);
    return repeated(sentence + " ", repetitions);
  }

  private static String repeated(String value, int repetitions) {
    return value.repeat(repetitions);
  }
}
