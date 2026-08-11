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

package opennlp.tools.glossary;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import opennlp.tools.stemmer.snowball.SnowballStemmer;
import opennlp.tools.stemmer.snowball.SnowballStemmerFactory;
import opennlp.tools.util.normalizer.GermanUmlautCharSequenceNormalizer;
import opennlp.tools.util.normalizer.TermAnalyzer;

/**
 * JMH throughput for the three glossary matching paths: the exact character
 * automaton, the offset-aware orthographic fold, and the token-normalized
 * {@link TermAnalyzingGlossaryMatcher}. Two inputs per path: a short mixed text
 * and a realistic 4 KiB document, each against a small and a wide glossary.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
public class AhoCorasickGlossaryMatcherBenchmark {

  private static final String SHORT_TEXT =
      ("Die Strasse und die Stra\u00DFe liegen an New York "
      + "City und Manhattan. Machine Learning meets maximum entropy model. ")
      .repeat(20);

  /** A realistic paragraph repeated to roughly 4 KiB of prose. */
  private static final String FOUR_KIB_TEXT = buildFourKibText();

  private static String buildFourKibText() {
    final String paragraph = "The vendors sold hot dogs and soft drinks near the "
        + "hot dog stands along the avenue, while machine learning engineers from "
        + "New York City argued about maximum entropy models over lunch. Down the "
        + "Strasse, a perceptron model classified the crowd, and the training data "
        + "kept growing as more documents arrived from Manhattan. ";
    final StringBuilder text = new StringBuilder(4200);
    while (text.length() < 4096) {
      text.append(paragraph);
    }
    return text.substring(0, 4096);
  }

  private static List<GlossaryEntry> smallGlossary() {
    final List<GlossaryEntry> glossary = new ArrayList<>();
    glossary.add(new GlossaryEntry("ST", "strasse"));
    glossary.add(new GlossaryEntry("NYC", "New York City"));
    glossary.add(new GlossaryEntry("MN", "Manhattan"));
    glossary.add(new GlossaryEntry("ML", "machine learning"));
    glossary.add(new GlossaryEntry("ME", "maximum entropy model"));
    glossary.add(new GlossaryEntry("FOOD", "hot dog"));
    glossary.add(new GlossaryEntry("DRINK", "soft drink"));
    return glossary;
  }

  /** The small glossary plus synthetic entries, wide enough to stress the automaton. */
  private static List<GlossaryEntry> wideGlossary() {
    final List<GlossaryEntry> glossary = smallGlossary();
    for (int i = 0; i < 1000; i++) {
      glossary.add(new GlossaryEntry("SYN-" + i, "synthetic term " + i));
      glossary.add(new GlossaryEntry("PROD-" + i, "product line " + i + " release"));
    }
    return glossary;
  }

  @State(Scope.Benchmark)
  public static class MatcherState {
    AhoCorasickGlossaryMatcher exact;
    AhoCorasickGlossaryMatcher exactWide;
    AhoCorasickGlossaryMatcher offsetAware;
    TermAnalyzingGlossaryMatcher termAnalyzing;
    TermAnalyzingGlossaryMatcher termAnalyzingWide;

    @Setup(Level.Trial)
    public void setUp() {
      final List<GlossaryEntry> small = smallGlossary();
      final List<GlossaryEntry> wide = wideGlossary();
      final TermAnalyzer analyzer = TermAnalyzer.builder()
          .caseFold()
          .stem(new SnowballStemmerFactory(SnowballStemmer.ALGORITHM.ENGLISH))
          .build();
      exact = new AhoCorasickGlossaryMatcher(small, true);
      exactWide = new AhoCorasickGlossaryMatcher(wide, true);
      offsetAware = new AhoCorasickGlossaryMatcher(small, true,
          GermanUmlautCharSequenceNormalizer.getInstance());
      termAnalyzing = new TermAnalyzingGlossaryMatcher(small, analyzer);
      termAnalyzingWide = new TermAnalyzingGlossaryMatcher(wide, analyzer);
    }
  }

  @Benchmark
  public void exactIgnoreCaseShortText(MatcherState state, Blackhole blackhole) {
    blackhole.consume(state.exact.match(SHORT_TEXT));
  }

  @Benchmark
  public void exactIgnoreCaseFourKib(MatcherState state, Blackhole blackhole) {
    blackhole.consume(state.exact.match(FOUR_KIB_TEXT));
  }

  @Benchmark
  public void exactWideGlossaryFourKib(MatcherState state, Blackhole blackhole) {
    blackhole.consume(state.exactWide.match(FOUR_KIB_TEXT));
  }

  @Benchmark
  public void offsetAwareGermanUmlautShortText(MatcherState state, Blackhole blackhole) {
    blackhole.consume(state.offsetAware.match(SHORT_TEXT));
  }

  @Benchmark
  public void offsetAwareGermanUmlautFourKib(MatcherState state, Blackhole blackhole) {
    blackhole.consume(state.offsetAware.match(FOUR_KIB_TEXT));
  }

  @Benchmark
  public void termAnalyzingStemShortText(MatcherState state, Blackhole blackhole) {
    blackhole.consume(state.termAnalyzing.match(SHORT_TEXT));
  }

  @Benchmark
  public void termAnalyzingStemFourKib(MatcherState state, Blackhole blackhole) {
    blackhole.consume(state.termAnalyzing.match(FOUR_KIB_TEXT));
  }

  @Benchmark
  public void termAnalyzingWideGlossaryFourKib(MatcherState state, Blackhole blackhole) {
    blackhole.consume(state.termAnalyzingWide.match(FOUR_KIB_TEXT));
  }
}
