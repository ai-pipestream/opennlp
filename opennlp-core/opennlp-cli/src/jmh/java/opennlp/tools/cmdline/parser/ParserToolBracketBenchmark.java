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
package opennlp.tools.cmdline.parser;

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

import opennlp.tools.parser.Parse;
import opennlp.tools.parser.Parser;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@State(Scope.Thread)
public class ParserToolBracketBenchmark {

  private static final Parser IDENTITY_PARSER = new Parser() {
    @Override
    public Parse[] parse(Parse tokens, int numParses) {
      return new Parse[] {tokens};
    }

    @Override
    public Parse parse(Parse tokens) {
      return tokens;
    }
  };

  @Param({"plain", "unicode", "bracketDense", "longLine"})
  private String workload;

  private String line;

  @Setup(Level.Trial)
  public void setUp() {
    line = switch (workload) {
      case "plain" -> "The quick brown fox";
      case "unicode" -> "Café（東京） and {résumé}";
      case "bracketDense" -> "((alpha)){beta}(gamma){delta}".repeat(64);
      case "longLine" -> "token ".repeat(4096) + "(tail)";
      default -> throw new IllegalArgumentException("Unknown workload: " + workload);
    };
    Parse[] parsed = parseLine();
    if (parsed.length != 1 || parsed[0].getChildren().length == 0) {
      throw new IllegalStateException("Benchmark fixture did not produce tokens");
    }
    StringBuilder output = new StringBuilder(parsed[0].getText());
    for (Parse child : parsed[0].getChildren()) {
      output.append('|').append(child.getType()).append(':').append(child.getSpan());
    }
    System.out.println("Bracket fixture " + workload + ": tokens="
        + parsed[0].getChildren().length + ", outputHash="
        + Integer.toHexString(output.toString().hashCode()));
  }

  @Benchmark
  public Parse[] parseLine() {
    return ParserTool.parseLine(line, IDENTITY_PARSER, 1);
  }
}
