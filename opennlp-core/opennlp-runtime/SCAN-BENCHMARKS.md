# Runtime scan benchmarks

The checksum benchmark uses the public cached-model path, so its score includes sidecar I/O,
hashing, and model construction. `DownloadLinkParsingBenchmark` covers steady parsing and a
single-shot cold-start slice for normal, markup-heavy, and long indexes.

Build and materialize the test classpath:

```bash
./mvnw test-compile -Pjmh -pl opennlp-core/opennlp-runtime -am \
    -Dopennlp.forkCount=1 -Drat.skip=true
./mvnw dependency:build-classpath -Pjmh -pl opennlp-core/opennlp-runtime \
    -am -Dopennlp.forkCount=1 -Drat.skip=true \
    -DincludeScope=test -Dmdep.outputFile=/tmp/opennlp-jmh-cp.txt
CP="opennlp-core/opennlp-runtime/target/classes:opennlp-core/opennlp-runtime/target/test-classes:$(cat /tmp/opennlp-jmh-cp.txt)"
```

Smoke-test the harness:

```bash
java -cp "$CP" org.openjdk.jmh.Main 'FeatureClassificationBenchmark|AncoraHeadRulesBenchmark|DownloadChecksumBenchmark' \
    -f 1 -wi 0 -i 1 -r 100ms
```

To measure an implementation checkout without copying benchmark sources into its review diff,
build the same module in that checkout and prepend its `target/classes` directory to `CP`.
Classpath order selects the implementation while retaining the benchmark from this checkout.

For measurements, use at least two forks, warmup, several measurement iterations, and the allocation profiler. Save the raw output with the exact commit and JDK. Do not treat a smoke run as performance evidence.

Run `DownloadLinkParsingBenchmark` only after the HTML parser implementation and its dependency
classpath have been finalized. Its single-shot method needs multiple independent forks for useful
cold-start evidence; one fork is only a harness check.

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
