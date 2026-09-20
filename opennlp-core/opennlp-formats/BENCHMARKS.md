# CoNLL-U parsing benchmark

Each invocation constructs a fresh stream outside the timed method. The timed operation is the public `read()` call.

Trial setup checks that each fixture produces one nonempty sentence and records the line count and
a deterministic hash of the text, identifiers, and forms. Keep those diagnostics with the results;
the range/empty-node workload may reveal intended output corrections between implementations.

Build and materialize the test classpath:

```bash
./mvnw test-compile -Pjmh -pl opennlp-core/opennlp-formats -am \
    -Dopennlp.forkCount=1 -Drat.skip=true
./mvnw dependency:build-classpath -Pjmh -pl opennlp-core/opennlp-formats \
    -am -Dopennlp.forkCount=1 -Drat.skip=true \
    -DincludeScope=test -Dmdep.outputFile=/tmp/opennlp-jmh-cp.txt
CP="opennlp-core/opennlp-formats/target/classes:opennlp-core/opennlp-formats/target/test-classes:$(cat /tmp/opennlp-jmh-cp.txt)"
```

Smoke-test the harness:

```bash
java -cp "$CP" org.openjdk.jmh.Main 'ConlluStreamBenchmark' \
    -f 1 -wi 0 -i 1 -r 100ms
```

To measure an implementation checkout without copying benchmark sources into its review diff,
build the same module in that checkout and prepend its `target/classes` directory to `CP`.
Classpath order selects the implementation while retaining the benchmark from this checkout.

For measurements, use at least two forks, warmup, several measurement iterations, and the allocation profiler. Save the raw output with the exact commit and JDK. Do not treat a smoke run as performance evidence.

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

The GC profiler includes allocation from invocation setup and teardown even though the timing excludes those methods. Report allocation as the complete stream invocation, not just identifier parsing.
