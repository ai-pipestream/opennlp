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
package opennlp.tools.tokenize;

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

import opennlp.tools.util.normalizer.CodePointSet;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@State(Scope.Thread)
public class TokenizerCharacterPolicyBenchmark {

  @Param({"ascii", "unicodeMarks", "supplementary", "longToken", "earlyReject"})
  private String workload;

  @Param({"custom", "unicode17Latin"})
  private String inventory;

  private TokenizerCharacterPolicy policy;
  private String[] inputs;
  private int index;

  @Setup(Level.Trial)
  public void setUp() {
    CodePointSet letters = CodePointSet.ofRange('A', 'Z')
        .union(CodePointSet.ofRange('a', 'z'))
        .union(CodePointSet.of('é', 0x1DF00));
    CodePointSet digits = CodePointSet.ofRange('0', '9');
    CodePointSet marks = CodePointSet.of(0x301);
    policy = switch (inventory) {
      case "custom" -> TokenizerCharacterPolicy.of(letters, digits, marks);
      case "unicode17Latin" -> TokenizerCharacterPolicy.latinUnicode17();
      default -> throw new IllegalArgumentException("Unknown inventory: " + inventory);
    };
    inputs = switch (workload) {
      case "ascii" -> new String[] {"OpenNLP2026", "ApacheNLP2025"};
      case "unicodeMarks" -> new String[] {"e\u0301cole", "école"};
      case "supplementary" -> new String[] {
          new String(Character.toChars(0x1DF00)) + "9",
          new String(Character.toChars(0x1DF00)) + "8"};
      case "longToken" -> new String[] {"Ab9".repeat(4096), "Ac8".repeat(4096)};
      case "earlyReject" -> new String[] {"!not-a-token", "?not-a-token"};
      default -> throw new IllegalArgumentException("Unknown workload: " + workload);
    };
    boolean expected = !"earlyReject".equals(workload);
    for (String input : inputs) {
      if (policy.test(input) != expected) {
        throw new IllegalStateException("Unexpected eligibility for " + inventory + '/' + workload);
      }
    }
    System.out.println("Policy fixture " + inventory + '/' + workload
        + ": accepted=" + expected + ", letters=" + policy.getLetters().toArray().length);
  }

  @Benchmark
  public boolean test() {
    return policy.test(inputs[index++ & 1]);
  }
}
