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

package opennlp.tools.pii;

import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

/**
 * JMH benchmark for the PII extractors, in throughput of characters scanned.
 *
 * <p>Redaction usually sits in front of everything else a pipeline does, so the question the
 * benchmark answers is what a scan costs on text that holds nothing worth reporting. Three
 * workloads separate the costs:</p>
 * <ul>
 *   <li>{@code clean}: prose with no PII at all, the case that dominates a real corpus. This
 *       measures the scanners' rejection path, where a candidate is abandoned after a
 *       character or two.</li>
 *   <li>{@code nearMiss}: the same prose seeded with values that are shaped right and fail
 *       their checksum, which forces every scanner to run its validation and then report
 *       nothing. This is the adversarial case and the upper bound on rejection cost.</li>
 *   <li>{@code dense}: the same prose seeded with real values, one every few words, which
 *       adds the cost of normalizing and reporting a mention.</li>
 * </ul>
 *
 * <p>Both the default {@link CursorPiiExtractor} and the widest
 * {@link PiiPacks#allStructured()} configuration run each workload, so the price of turning
 * every detector on is visible rather than assumed.</p>
 *
 * <p>One op scans one 4 kB document; each thread walks the document list from its own
 * cursor.</p>
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 10, time = 2)
@Fork(2)
public class CursorPiiExtractorBenchmark {

  private static final int DOCUMENTS = 256;
  private static final int TARGET_LENGTH = 4096;

  private static final String[] WORDS = {
      "the", "team", "reviewed", "every", "ticket", "before", "the", "release",
      "and", "asked", "whether", "the", "customer", "record", "was", "complete",
      "support", "replied", "that", "the", "account", "had", "been", "verified",
      "with", "the", "usual", "checks", "on", "file", "since", "january",
      "billing", "confirmed", "the", "invoice", "total", "matched", "the", "order",
      "logs", "showed", "no", "retries", "after", "the", "second", "attempt"
  };

  /** Values that are shaped like PII and fail their checksum or their range rules. */
  private static final String[] NEAR_MISSES = {
      "4111111111111112", "DE89370400440532013001", "555-123", "jane@example.invalidtld",
      "256.1.1.1", "00:1b:44:11:3a", "AKIAIOSFODNN7EXAMPL", "021000022",
      "666-45-6789", "1234567890", "abc.def.ghi", "0x52908400098527886E0F7030069857D2E4169EE"
  };

  /** Values every relevant detector accepts. */
  private static final String[] REAL_VALUES = {
      "4111111111111111", "DE89370400440532013000", "(555) 123-4567", "jane@example.com",
      "192.168.1.20", "00:1b:44:11:3a:b7", "AKIAIOSFODNN7EXAMPLE", "021000021",
      "078-05-1120", "943 476 5919", "0x52908400098527886E0F7030069857D2E4169EE7"
  };

  @State(Scope.Benchmark)
  public static class Corpus {

    @Param({"clean", "nearMiss", "dense"})
    String workload;

    String[] documents;

    @Setup(Level.Trial)
    public void build() {
      final String[] seeds = switch (workload) {
        case "nearMiss" -> NEAR_MISSES;
        case "dense" -> REAL_VALUES;
        default -> new String[0];
      };
      final Random random = new Random(42);
      documents = new String[DOCUMENTS];
      for (int i = 0; i < DOCUMENTS; i++) {
        documents[i] = document(random, seeds);
      }
    }

    /**
     * Builds one document of roughly {@link #TARGET_LENGTH} characters, seeding a value
     * every eight words when the workload calls for them.
     *
     * @param random The source of the word and value choices.
     * @param seeds The values to seed, empty for the clean workload.
     * @return The document text.
     */
    private String document(Random random, String[] seeds) {
      final StringBuilder text = new StringBuilder(TARGET_LENGTH + 32);
      int words = 0;
      while (text.length() < TARGET_LENGTH) {
        if (seeds.length > 0 && words % 8 == 7) {
          text.append(seeds[random.nextInt(seeds.length)]);
        } else {
          text.append(WORDS[random.nextInt(WORDS.length)]);
        }
        text.append(words % 12 == 11 ? ". " : " ");
        words++;
      }
      return text.toString();
    }
  }

  @State(Scope.Benchmark)
  public static class Default {
    PiiExtractor extractor;

    @Setup(Level.Trial)
    public void create() {
      extractor = new CursorPiiExtractor();
    }
  }

  @State(Scope.Benchmark)
  public static class Widest {
    PiiExtractor extractor;

    @Setup(Level.Trial)
    public void create() {
      extractor = PiiPacks.allStructured();
    }
  }

  @State(Scope.Thread)
  public static class Cursor {
    int position;

    @Setup(Level.Trial)
    public void randomize() {
      position = new Random().nextInt(DOCUMENTS);
    }
  }

  @Benchmark
  @Threads(Threads.MAX)
  public void defaultExtractor(Corpus corpus, Default state, Cursor cursor, Blackhole bh) {
    final List<PiiMention> mentions = state.extractor.extract(
        corpus.documents[cursor.position++ & (DOCUMENTS - 1)]);
    bh.consume(mentions);
  }

  @Benchmark
  @Threads(Threads.MAX)
  public void widestConfiguration(Corpus corpus, Widest state, Cursor cursor, Blackhole bh) {
    final List<PiiMention> mentions = state.extractor.extract(
        corpus.documents[cursor.position++ & (DOCUMENTS - 1)]);
    bh.consume(mentions);
  }

  /**
   * Quick local iteration only: {@code forks(0)} disables JVM fork isolation
   * (unlike {@code mvn} with the {@code jmh} profile).
   * Use the Maven-invoked configuration for publishable numbers.
   *
   * @param args Ignored.
   * @throws Exception Thrown if the runner fails.
   */
  public static void main(String[] args) throws Exception {
    Options opt = new OptionsBuilder()
        .include(CursorPiiExtractorBenchmark.class.getSimpleName())
        .forks(0)
        .warmupIterations(3)
        .measurementIterations(5)
        .build();
    new Runner(opt).run();
  }
}
