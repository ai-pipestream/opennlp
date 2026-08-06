# Thread Safety Benchmarks

## JMH Benchmarks

JMH benchmarks compare three instance allocation strategies under
concurrent load, with proper fork isolation, warmup, and statistical
variance reporting.

### Benchmark classes

| Class | Component | Parameters |
|-------|-----------|------------|
| `TokenizerMEBenchmark` | TokenizerME | 3 approaches |
| `SentenceDetectorMEBenchmark` | SentenceDetectorME | 3 approaches |
| `POSTaggerMEBenchmark` | POSTaggerME | 3 approaches x 2 cache configs |
| `SnowballStemmerBenchmark` | SnowballStemmer | 3 approaches (incl. plain-field baseline) |
| `CachingStemmerBenchmark` | CachingStemmer | cached vs uncached x 2 workloads |
| `BeamSearchBenchmark` | BeamSearch | 3 sequence lengths x 2 cache configs, 1 vs max threads |

### Approaches measured

| Approach | Description |
|----------|-------------|
| `newInstancePerCall` | Fresh ME per operation (traditional pattern, backward-compat baseline) |
| `instancePerThread` | One ME per thread, reused across operations |
| `sharedInstance` | Single ME shared by all threads |

### Building and running

```bash
# Build with JMH profile
mvn test-compile -Pjmh \
    -pl opennlp-core/opennlp-runtime -am \
    -Dforbiddenapis.skip=true -Dcheckstyle.skip=true

# Materialize the test classpath once (JMH's forked JVMs inherit
# java.class.path, which mvn exec:java does not populate, running
# through exec:java fails with ClassNotFoundException: ForkedMain)
mvn dependency:build-classpath -pl opennlp-core/opennlp-runtime \
    -Pjmh -DincludeScope=test -Dmdep.outputFile=/tmp/cp.txt

CP="opennlp-core/opennlp-runtime/target/classes:opennlp-core/opennlp-runtime/target/test-classes:$(cat /tmp/cp.txt)"

# Run all ME benchmarks
java -cp "$CP" org.openjdk.jmh.Main 'opennlp.tools.*.ME*'

# Run POSTagger only (includes cacheSize param)
java -cp "$CP" org.openjdk.jmh.Main POSTaggerMEBenchmark
```

### Baseline comparison

Run the `newInstancePerCall` benchmark on both `main` and this
branch. The throughput numbers should be within JMH's error margin.

```bash
# On main:
# ... build and run as above, save output

# On this branch:
# ... build and run as above, compare
```

### SnowballStemmer results (Linux, JDK 25, 32 cores, 2 forks x 10 iterations)

`SnowballStemmerBenchmark` compares the thread-safe `SnowballStemmer`
(engine behind `OwnerOrPerThreadState`) against a baseline replica
that keeps the engine in a plain field (not shareable).
One op = stemming 16 English words.

| Strategy | 1 thread | 8 threads | 32 threads |
|----------|---------:|----------:|-----------:|
| `sharedInstance` (one shared stemmer) | 560k ± 3k ops/s | 1.55M ± 0.17M | 3.16M ± 0.34M |
| `instancePerThread` (stemmer per thread) | 509k ± 26k ops/s | 1.60M ± 0.17M | 2.94M ± 0.11M |
| `legacyInstancePerThread` (plain-field baseline, stemmer per thread) | 544k ± 19k ops/s | 1.46M ± 0.16M | 4.77M ± 0.39M |

At 1 and 8 threads the three strategies are within (or nearly within)
each other's error bars: the `OwnerOrPerThreadState` lookup is not
measurable against the cost of stemming itself. Only at full
saturation (32 threads, hyperthreaded) does the legacy plain-field
baseline pull ahead (~1.5x): with every hardware thread busy, the
per-call owner check plus `ThreadLocal` lookup is no longer hidden by
memory-level parallelism. Real pipelines stem as one stage among many,
so the saturated-microbenchmark gap is an upper bound, and the legacy
strategy was not shareable across threads in the first place.

### CachingStemmer results (same environment)

`CachingStemmerBenchmark` compares a `CachingStemmer` (per-thread LRU,
default 1024 entries, wrapping the English Snowball stemmer) against
the uncached shared stemmer. One op = 16 tokens from a 64k-token
stream. The `zipf` workload samples a 512-word vocabulary with 1/rank
weights (real-text repetition; the cache holds the whole vocabulary);
`diverse` samples an 8192-word vocabulary uniformly (8x cache
capacity: mostly misses plus constant eviction).

| Workload | Strategy | 8 threads | 32 threads |
|----------|----------|----------:|-----------:|
| `zipf` | `cachedShared` | 48.5M ± 0.7M ops/s | 95.4M ± 0.9M |
| `zipf` | `uncachedShared` | 1.43M ± 0.13M | 2.75M ± 0.34M |
| `diverse` | `cachedShared` | 1.81M ± 0.51M | 3.45M ± 0.18M |
| `diverse` | `uncachedShared` | 1.08M ± 0.09M | 3.13M ± 0.12M |

On the Zipf workload the cache is a ~34x throughput multiplier (raw
stemming becomes a hash lookup for the dominant vocabulary). On the
cache-hostile workload it still does not lose: the ~12% residual hit
rate pays for the eviction overhead. The cache more than recovers the
`OwnerOrPerThreadState` lookup cost observed in
`SnowballStemmerBenchmark` at full saturation.

The cache is keyed to the physical thread, and these runs use a fixed
platform-thread pool whose threads live for the whole measurement. On
a virtual-thread-per-task executor every task starts with an empty
cache, so the multiplier only applies to repeats within one task;
workloads that stem a handful of words per task should expect
uncached-level throughput there.

### POSTagger cache impact

The `POSTaggerMEBenchmark` uses `@Param({"0", "3"})` for cache
size, producing a matrix of 3 approaches x 2 cache configs = 6
benchmark runs. This quantifies whether the context generator
cache provides measurable benefit.

### BeamSearch results

`BeamSearchBenchmark` decodes synthetic token sequences with a shared
`BeamSearch` (beam size 3, deterministic hash-seeded model, accept-all
validator). One op = one `bestSequence` call; invocations rotate through a
pool of 64 seeded inputs. The `sequenceLength` param (8/64/256) is the key
axis: the O(n^2) per-candidate outcome-list copying that the chain-node
refactor (OPENNLP-1903) eliminated only shows up on long sequences (NER
chunks run 200+ tokens). The `cacheSize` param (0/64) toggles the contexts
cache. To produce the pre-refactor baseline, place an `opennlp-ml-commons`
jar built before OPENNLP-1903 on the classpath ahead of the freshly built
classes and rerun; the benchmark only uses public API that predates the
refactor.

Results (Linux, JDK 25, 24 pinned cores of a 32-core box, 2 forks x 10
iterations; ops/s = decoded sequences per second):

| Method | sequenceLength | cacheSize | 1 thread | 24 threads | scaling |
|--------|---------------:|----------:|---------:|-----------:|--------:|
| pre-refactor  | 8   | 0  | 226,337 ± 4,164  | 1,404,404 ± 67,614 | 6.2x |
| pre-refactor  | 64  | 0  | 17,802 ± 43      | 67,482 ± 817       | 3.8x |
| pre-refactor  | 256 | 0  | 2,236 ± 38       | 5,491 ± 26         | 2.5x |
| pre-refactor  | 8   | 64 | 234,592 ± 705    | 1,382,812 ± 72,577 | 5.9x |
| pre-refactor  | 64  | 64 | 17,968 ± 92      | 61,639 ± 2,726     | 3.4x |
| pre-refactor  | 256 | 64 | 2,224 ± 20       | 5,260 ± 92         | 2.4x |
| post-refactor | 8   | 0  | 450,384 ± 3,306  | 3,162,724 ± 495,301 | 7.0x |
| post-refactor | 64  | 0  | 33,936 ± 482     | 297,001 ± 13,049   | 8.8x |
| post-refactor | 256 | 0  | 4,357 ± 168      | 50,176 ± 307       | 11.5x |
| post-refactor | 8   | 64 | 488,155 ± 8,095  | 3,512,519 ± 7,541  | 7.2x |
| post-refactor | 64  | 64 | 34,818 ± 232     | 347,266 ± 942      | 10.0x |
| post-refactor | 256 | 64 | 4,308 ± 27       | 50,203 ± 230       | 11.7x |

Two readings. First, the refactor's win grows with sequence length:
~2x single-thread at every length, and at 24 threads 2.3x (len 8),
4.4x (len 64), and 9.1x (len 256), the signature of an O(n^2) copy
cost being removed. Second, the pre-refactor scaling collapses as
sequences lengthen (6.2x down to 2.5x on 24 threads: memory-write
saturation from per-candidate list copying), while post-refactor
scaling grows with length (7.0x up to 11.5x).

Note the synthetic model's `eval` is a cheap hash, so this isolates
the sequence-mechanics cost the refactor targets; with a real maxent
model the eval compute is shared by both versions and the relative
single-thread win is smaller (~1.3x on ARM, roughly neutral on x86),
while the concurrent scaling gap is preserved (measured 5.8x -> 13.3x
at saturation on 24 pinned cores with the SourceForge-era 1.5 English
`en-ner-person.bin` model).

## JUnit Correctness Test

`ThreadSafetyBenchmarkIT` is a Failsafe integration test (`*IT.java`). It
verifies that a shared ME instance produces identical results to a
single-threaded baseline for all 7 ME classes under barrier-synchronized
concurrent access. Run it with `mvn verify` (not `mvn test`, which excludes
`*IT.java`).

```bash
mvn verify -pl opennlp-core/opennlp-runtime -am \
    -Dforbiddenapis.skip=true \
    -Dit.test=ThreadSafetyBenchmarkIT
```

# Thread Safety Benchmarks

## JMH Benchmarks

JMH benchmarks compare three instance allocation strategies under
concurrent load, with proper fork isolation, warmup, and statistical
variance reporting.

### Benchmark classes

| Class | Component | Parameters |
|-------|-----------|------------|
| `TokenizerMEBenchmark` | TokenizerME | 3 approaches |
| `SentenceDetectorMEBenchmark` | SentenceDetectorME | 3 approaches |
| `POSTaggerMEBenchmark` | POSTaggerME | 3 approaches x 2 cache configs |
| `SnowballStemmerBenchmark` | SnowballStemmer | 3 approaches (incl. plain-field baseline) |
| `CachingStemmerBenchmark` | CachingStemmer | cached vs uncached x 2 workloads |
| `SentenceDetectorMEAbbreviationBenchmark` | SentenceDetectorME abbreviation veto | 3 veto implementations x 4 document sizes x 2 dictionary sizes |

### Approaches measured

| Approach | Description |
|----------|-------------|
| `newInstancePerCall` | Fresh ME per operation (traditional pattern, backward-compat baseline) |
| `instancePerThread` | One ME per thread, reused across operations |
| `sharedInstance` | Single ME shared by all threads |

### Building and running

```bash
# Build with JMH profile
mvn test-compile -Pjmh \
    -pl opennlp-core/opennlp-runtime -am \
    -Dforbiddenapis.skip=true -Dcheckstyle.skip=true

# Materialize the test classpath once (JMH's forked JVMs inherit
# java.class.path, which mvn exec:java does not populate, running
# through exec:java fails with ClassNotFoundException: ForkedMain)
mvn dependency:build-classpath -pl opennlp-core/opennlp-runtime \
    -Pjmh -DincludeScope=test -Dmdep.outputFile=/tmp/cp.txt

CP="opennlp-core/opennlp-runtime/target/classes:opennlp-core/opennlp-runtime/target/test-classes:$(cat /tmp/cp.txt)"

# Run all ME benchmarks
java -cp "$CP" org.openjdk.jmh.Main 'opennlp.tools.*.ME*'

# Run POSTagger only (includes cacheSize param)
java -cp "$CP" org.openjdk.jmh.Main POSTaggerMEBenchmark
```

### Baseline comparison

Run the `newInstancePerCall` benchmark on both `main` and this
branch. The throughput numbers should be within JMH's error margin.

```bash
# On main:
# ... build and run as above, save output

# On this branch:
# ... build and run as above, compare
```

### SnowballStemmer results (Linux, JDK 25, 32 cores, 2 forks x 10 iterations)

`SnowballStemmerBenchmark` compares the thread-safe `SnowballStemmer`
(engine behind `OwnerOrPerThreadState`) against a baseline replica
that keeps the engine in a plain field (not shareable).
One op = stemming 16 English words.

| Strategy | 1 thread | 8 threads | 32 threads |
|----------|---------:|----------:|-----------:|
| `sharedInstance` (one shared stemmer) | 560k ± 3k ops/s | 1.55M ± 0.17M | 3.16M ± 0.34M |
| `instancePerThread` (stemmer per thread) | 509k ± 26k ops/s | 1.60M ± 0.17M | 2.94M ± 0.11M |
| `legacyInstancePerThread` (plain-field baseline, stemmer per thread) | 544k ± 19k ops/s | 1.46M ± 0.16M | 4.77M ± 0.39M |

At 1 and 8 threads the three strategies are within (or nearly within)
each other's error bars: the `OwnerOrPerThreadState` lookup is not
measurable against the cost of stemming itself. Only at full
saturation (32 threads, hyperthreaded) does the legacy plain-field
baseline pull ahead (~1.5x): with every hardware thread busy, the
per-call owner check plus `ThreadLocal` lookup is no longer hidden by
memory-level parallelism. Real pipelines stem as one stage among many,
so the saturated-microbenchmark gap is an upper bound, and the legacy
strategy was not shareable across threads in the first place.

### CachingStemmer results (same environment)

`CachingStemmerBenchmark` compares a `CachingStemmer` (per-thread LRU,
default 1024 entries, wrapping the English Snowball stemmer) against
the uncached shared stemmer. One op = 16 tokens from a 64k-token
stream. The `zipf` workload samples a 512-word vocabulary with 1/rank
weights (real-text repetition; the cache holds the whole vocabulary);
`diverse` samples an 8192-word vocabulary uniformly (8x cache
capacity: mostly misses plus constant eviction).

| Workload | Strategy | 8 threads | 32 threads |
|----------|----------|----------:|-----------:|
| `zipf` | `cachedShared` | 48.5M ± 0.7M ops/s | 95.4M ± 0.9M |
| `zipf` | `uncachedShared` | 1.43M ± 0.13M | 2.75M ± 0.34M |
| `diverse` | `cachedShared` | 1.81M ± 0.51M | 3.45M ± 0.18M |
| `diverse` | `uncachedShared` | 1.08M ± 0.09M | 3.13M ± 0.12M |

On the Zipf workload the cache is a ~34x throughput multiplier (raw
stemming becomes a hash lookup for the dominant vocabulary). On the
cache-hostile workload it still does not lose: the ~12% residual hit
rate pays for the eviction overhead. The cache more than recovers the
`OwnerOrPerThreadState` lookup cost observed in
`SnowballStemmerBenchmark` at full saturation.

The cache is keyed to the physical thread, and these runs use a fixed
platform-thread pool whose threads live for the whole measurement. On
a virtual-thread-per-task executor every task starts with an empty
cache, so the multiplier only applies to repeats within one task;
workloads that stem a handful of words per task should expect
uncached-level throughput there.

### SentenceDetectorME abbreviation veto results

`SentenceDetectorMEAbbreviationBenchmark` measures the abbreviation veto
of `SentenceDetectorME.isAcceptableBreak`. One op is one `sentPosDetect`
call over a document of `documentChars` characters. Three variants run
over the same input and the same model, so only the veto differs:

| Variant | Description |
|---------|-------------|
| `noDictionary` | No abbreviation dictionary, so the veto is never entered. The floor: scanning plus one maxent evaluation per candidate. |
| `legacyVeto` | The implementation before the bounded window, restored by overriding the method. |
| `boundedWindowVeto` | The current implementation. |

**This path is gated on an abbreviation dictionary being configured, and
no sentence model published by the project carries one.** A stock model
is the `noDictionary` column, which is why a quadratic veto could sit in
the library unnoticed. It is entered the moment a user supplies a
dictionary, which is the normal thing to do for acceptable segmentation
on domain text.

Results (Linux, JDK 25 GraalVM CE, 8 pinned cores of a 32-core box,
2 forks x 10 iterations x 3 s, 3 warmup iterations x 2 s, 1 thread;
lower is better):

| dictionaryEntries | documentChars | `noDictionary` | `legacyVeto` | `boundedWindowVeto` | legacy / bounded |
|---:|---:|---:|---:|---:|---:|
| 10 | 12 500 | 0.120 ± 0.006 | 5.045 ± 0.069 | 0.163 ± 0.001 | 31x |
| 10 | 25 000 | 0.227 ± 0.003 | 18.876 ± 1.148 | 0.341 ± 0.009 | 55x |
| 10 | 50 000 | 0.455 ± 0.011 | 75.119 ± 2.169 | 0.640 ± 0.011 | 117x |
| 10 | 100 000 | 0.902 ± 0.010 | 278.811 ± 3.664 | 1.263 ± 0.009 | 221x |
| 200 | 12 500 | 0.114 ± 0.002 | 10.582 ± 0.415 | 0.212 ± 0.002 | 50x |
| 200 | 25 000 | 0.235 ± 0.012 | 38.968 ± 0.661 | 0.410 ± 0.004 | 95x |
| 200 | 50 000 | 0.453 ± 0.004 | 145.631 ± 1.307 | 0.821 ± 0.016 | 177x |
| 200 | 100 000 | 0.930 ± 0.023 | 546.487 ± 15.571 | 1.611 ± 0.028 | 339x |

All numbers are ms/op.

The growth rate is the point, not the ratio. Per doubling of
`documentChars`:

| Variant | 10 entries | 200 entries |
|---------|------------|-------------|
| `noDictionary` | 1.89x, 2.00x, 1.98x | 2.06x, 1.93x, 2.05x |
| `legacyVeto` | 3.74x, 3.98x, 3.71x | 3.68x, 3.74x, 3.75x |
| `boundedWindowVeto` | 2.09x, 1.88x, 1.97x | 1.93x, 2.00x, 1.96x |

Four times the work for twice the input is the signature of a quadratic
algorithm: the previous implementation searched the whole text once per
dictionary entry for every end-of-sentence candidate, and for a
case-insensitive dictionary (all the shipped ones are) it also
lower-cased the whole text on every one of those calls. The bounded
window tracks the `noDictionary` floor instead, so the ratio in the last
column above is not a constant speedup, it keeps growing with document
size.

The residual cost of the veto over the floor is roughly 40% at 10
entries and 80% at 200 entries, flat in document size. That is the
per-candidate window probing, and it is linear.

Beyond the matrix, at 200 dictionary entries (1 fork x 3 iterations
x 20 s, same box):

| documentChars | `legacyVeto` | `boundedWindowVeto` |
|---:|---:|---:|
| 200 000 | 2 066.219 ± 103.210 | 3.513 ± 0.165 |
| 400 000 | 8 221.748 ± 955.911 | 6.800 ± 0.154 |

The quadratic law holds over the whole measured range: 546 ms, 2 066 ms
and 8 222 ms for 100 000, 200 000 and 400 000 characters is 3.78x then
3.98x per doubling, while the bounded window goes 1.611 ms, 3.513 ms,
6.800 ms, that is 2.18x then 1.94x. Extending the fitted curves one more
decade, a 4 MB document costs the previous implementation about 14
minutes and the current one about 70 ms. Only the 4 MB figures are
extrapolated; everything in the tables above was measured.

```bash
java -cp "$CP" org.openjdk.jmh.Main SentenceDetectorMEAbbreviationBenchmark \
    -f 2 -wi 3 -w 2s -i 10 -r 3s
```

### POSTagger cache impact

The `POSTaggerMEBenchmark` uses `@Param({"0", "3"})` for cache
size, producing a matrix of 3 approaches x 2 cache configs = 6
benchmark runs. This quantifies whether the context generator
cache provides measurable benefit.

## JUnit Correctness Test

`ThreadSafetyBenchmarkIT` is a Failsafe integration test (`*IT.java`). It
verifies that a shared ME instance produces identical results to a
single-threaded baseline for all 7 ME classes under barrier-synchronized
concurrent access. Run it with `mvn verify` (not `mvn test`, which excludes
`*IT.java`).

```bash
mvn verify -pl opennlp-core/opennlp-runtime -am \
    -Dforbiddenapis.skip=true \
    -Dit.test=ThreadSafetyBenchmarkIT
```
