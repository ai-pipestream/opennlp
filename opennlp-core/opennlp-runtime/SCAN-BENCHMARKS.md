# Runtime scan benchmarks

The checksum benchmark uses the public cached-model path, so its score includes sidecar I/O,
hashing, and model construction. `DownloadLinkParsingBenchmark` covers steady parsing for normal,
markup-heavy, and long indexes. `DownloadLinkColdStartBenchmark` separately measures the first HTML
parser call in each fork.

Build and materialize the test classpath:

```bash
./mvnw test-compile -Pjmh -pl opennlp-core/opennlp-runtime -am \
    -Dopennlp.forkCount=1 -Drat.skip=true
./mvnw dependency:build-classpath -Pjmh -pl opennlp-core/opennlp-runtime \
    -am -Dopennlp.forkCount=1 -Drat.skip=true \
    -DincludeScope=test -Dmdep.outputFile=/tmp/opennlp-jmh-cp.txt
CP="opennlp-core/opennlp-runtime/target/classes:opennlp-core/opennlp-runtime/target/test-classes:$(cat /tmp/opennlp-jmh-cp.txt)"
```

Smoke-test the steady-state harness shape:

```bash
java -cp "$CP" org.openjdk.jmh.Main 'FeatureClassificationBenchmark|AncoraHeadRulesBenchmark|DownloadChecksumBenchmark|DownloadLinkParsingBenchmark' \
    -f 1 -wi 0 -i 1 -r 100ms
```

Smoke-test the cold-start source shape separately:

```bash
java -cp "$CP" org.openjdk.jmh.Main DownloadLinkColdStartBenchmark \
    -p workload=normal -f 1 -wi 0 -i 1
```

To measure an implementation checkout without copying benchmark sources into its review diff,
build the same module in that checkout and prepend its `target/classes` directory to `CP`.
Classpath order selects the implementation while retaining the benchmark from this checkout.

For measurements, use at least two forks, warmup, several measurement iterations, and the allocation profiler. Save the raw output with the exact commit and JDK. Do not treat a smoke run as performance evidence.

Run the link benchmarks only after the HTML parser implementation and its dependency classpath have
been finalized. Steady-state measurements use `DownloadLinkParsingBenchmark` with normal warmup and
measurement iterations. Cold-start measurements use `DownloadLinkColdStartBenchmark` with zero
warmup iterations, exactly one single-shot measurement per fork, and multiple independent forks:

```bash
java -cp "$CP" org.openjdk.jmh.Main DownloadLinkColdStartBenchmark \
    -f 10 -wi 0 -i 1
```

Do not add a warmup invocation to the cold command. Its trial setup creates the local fixture but
does not construct or invoke `DownloadParser`. The timed method performs the first parser call in
that fork. Trial teardown validates the result and prints its fingerprint after timing. One fork is
only a harness check, not cold-start evidence.

Before measuring, prepend the checkout's built reactor class directories to the resolved dependency
classpath. This prevents an installed Maven SNAPSHOT from supplying the implementation being timed:

```bash
while IFS= read -r benchmark_pom; do
    benchmark_classes="${benchmark_pom%/pom.xml}/target/classes"
    if [ -d "$benchmark_classes" ]; then
        CP="$benchmark_classes:$CP"
    fi
done < <(git ls-files '**/pom.xml')
```

For a candidate overlay, repeat this step with absolute paths into the candidate checkout, after
adding its external dependencies. Record the actual loaded class origins, exact branch commit,
harness diff, JDK and raw JMH output. Build and measurement must not overlap with other work.

Unicode feature cases and flexible HTML markup exercise corrected behavior that the baseline may not implement. Compare speed only alongside output correctness. Link timing includes local file reads and model-map construction; checksum timing includes hashing and model loading.

Trial setup validates stable behavior for shared workloads and prints a compact output fingerprint.
The markup-heavy link case permits either the baseline's empty result or the corrected parser's
single English tokenizer entry, while rejecting links embedded in comments. Preserve these
fingerprints with each run and do not describe unequal-output rows as equivalent-work speedups.
