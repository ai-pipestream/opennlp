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
package opennlp.tools.parser.lang.es;

import java.io.IOException;
import java.io.StringReader;
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
import opennlp.tools.util.Span;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@State(Scope.Thread)
public class AncoraHeadRulesBenchmark {

  @Param({"common", "nearMiss"})
  private String workload;

  private AncoraSpanishHeadRules rules;
  private Parse[] constituents;

  @Setup(Level.Trial)
  public void setUp() throws IOException {
    if ("common".equals(workload)) {
      rules = new AncoraSpanishHeadRules(new StringReader("4 TEST 0 NC.* AQA.*\n"));
      constituents = new Parse[] {constituent("OTHER", 0), constituent("AQ0MS0", 1),
          constituent("NCMS000", 2)};
    } else if ("nearMiss".equals(workload)) {
      rules = new AncoraSpanishHeadRules(new StringReader("3 TEST 0 A.*B.*C.*Z\n"));
      constituents = new Parse[] {constituent("A" + "BC".repeat(128) + "Y", 0),
          constituent("OTHER", 1)};
    } else {
      throw new IllegalArgumentException("Unknown workload: " + workload);
    }
    Parse selected = rules.getHead(constituents, "TEST");
    Parse expected = "common".equals(workload) ? constituents[2] : constituents[1].getHead();
    if (selected != expected) {
      throw new IllegalStateException("Unexpected head for " + workload + ": " + selected);
    }
    System.out.println("AncoraHeadRulesBenchmark fingerprint " + workload + "="
        + selected.getType() + ':' + selected.getText());
  }

  @Benchmark
  public Parse selectHead() {
    return rules.getHead(constituents, "TEST");
  }

  private static Parse constituent(String type, int index) {
    String text = "w" + index;
    Parse token = new Parse(text, new Span(0, text.length()), Parser.TOK_NODE, 1d, index);
    return new Parse(text, new Span(0, text.length()), type, 1d, token);
  }
}
