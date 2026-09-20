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
package opennlp.tools.formats.conllu;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
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

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@State(Scope.Thread)
public class ConlluStreamBenchmark {

  @Param({"normal", "rangesAndEmptyNodes", "longSentence"})
  private String workload;

  private byte[] input;
  private ConlluStream stream;

  @Setup(Level.Trial)
  public void prepareInput() throws IOException {
    String sentence = switch (workload) {
      case "normal" -> "# text = Café 東京\n"
          + "1\tCafé\tcafé\tNOUN\t_\t_\t2\tnsubj\t_\t_\n"
          + "2\tarrive\tarrive\tVERB\t_\t_\t0\troot\t_\t_\n";
      case "rangesAndEmptyNodes" -> "# text = can't go\n"
          + "1-2\tcan't\t_\t_\t_\t_\t_\t_\t_\t_\n"
          + "1\tca\tcan\tAUX\t_\t_\t3\taux\t_\t_\n"
          + "2\tn't\tnot\tPART\t_\t_\t3\tadvmod\t_\t_\n"
          + "2.1\treally\treally\tADV\t_\t_\t_\t_\t3:advmod\t_\n"
          + "3\tgo\tgo\tVERB\t_\t_\t0\troot\t_\t_\n";
      case "longSentence" -> longSentence(1024);
      default -> throw new IllegalArgumentException("Unknown workload: " + workload);
    };
    input = (sentence + "\n").getBytes(StandardCharsets.UTF_8);
    try (ConlluStream fixture = new ConlluStream(() -> new ByteArrayInputStream(input))) {
      ConlluSentence parsed = fixture.read();
      if (parsed == null || parsed.getWordLines().isEmpty() || fixture.read() != null) {
        throw new IllegalStateException("Benchmark fixture must contain one nonempty sentence");
      }
      StringBuilder output = new StringBuilder(parsed.getTextComment());
      for (ConlluWordLine word : parsed.getWordLines()) {
        output.append('|').append(word.getId()).append(':').append(word.getForm());
      }
      System.out.println("CoNLL-U fixture " + workload + ": lines="
          + parsed.getWordLines().size() + ", outputHash="
          + Integer.toHexString(output.toString().hashCode()));
    }
  }

  @Setup(Level.Invocation)
  public void openStream() throws IOException {
    stream = new ConlluStream(() -> new ByteArrayInputStream(input));
  }

  @Benchmark
  public ConlluSentence readSentence() throws IOException {
    return stream.read();
  }

  @TearDown(Level.Invocation)
  public void closeStream() throws IOException {
    stream.close();
  }

  private static String longSentence(int words) {
    StringBuilder sentence = new StringBuilder(words * 36);
    sentence.append("# text = long scaling sentence\n");
    for (int i = 1; i <= words; i++) {
      sentence.append(i).append("\tw").append(i)
          .append("\tw\tNOUN\t_\t_\t0\tdep\t_\t_\n");
    }
    return sentence.toString();
  }
}
