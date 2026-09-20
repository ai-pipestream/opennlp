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
package opennlp.tools.parser;

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

/** Measures the public {@link Parse#parseParse(String)} tree-bank parser. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class ParseParseBenchmark {

  @State(Scope.Benchmark)
  public static class InputState {

    @Param({"normalUnicode", "markupHeavy", "long16k", "long64k"})
    public String workload;

    String input;

    @Setup(Level.Trial)
    public void prepare() {
      input = switch (workload) {
        case "normalUnicode" -> normalUnicode();
        case "markupHeavy" -> nestedTree(48);
        case "long16k" -> longTree(16 * 1024);
        case "long64k" -> longTree(64 * 1024);
        default -> throw new IllegalArgumentException("Unknown workload: " + workload);
      };
      Parse parsed = Parse.parseParse(input);
      if (parsed.getTagNodes().length == 0) {
        throw new IllegalStateException("Benchmark fixture did not produce tagged tokens");
      }
      System.out.println("Treebank fixture " + workload + ": tags="
          + parsed.getTagNodes().length + ", outputHash="
          + Integer.toHexString(parsed.toStringPennTreebank().hashCode()));
    }
  }

  @Benchmark
  public void parse(InputState state, Blackhole blackhole) {
    blackhole.consume(Parse.parseParse(state.input));
  }

  private static String normalUnicode() {
    return "(TOP (S (NP-SBJ (NNP Chloé)) (VP (VBD visited) "
        + "(NP (NNP 東京)) (PP (IN with) (NP (NNP José)))) (. .)))";
  }

  private static String nestedTree(int depth) {
    StringBuilder tree = new StringBuilder(depth * 7 + 32);
    tree.append("(TOP ");
    for (int i = 0; i < depth; i++) {
      tree.append("(NEST ");
    }
    tree.append("(NNP Zürich)");
    tree.append(")".repeat(depth + 1));
    return tree.toString();
  }

  private static String longTree(int minimumLength) {
    String constituent = " (NP (JJ multilingual) (NN text))";
    int repetitions = (minimumLength + constituent.length()) / constituent.length();
    return "(TOP (S" + constituent.repeat(repetitions) + "))";
  }
}
