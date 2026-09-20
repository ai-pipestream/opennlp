<!--
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to You under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Spellcheck exclusion benchmark

A deterministic counting spell checker keeps URL exclusion and token scanning in focus. JMH
reports lookup calls as an event counter, so an unchanged normalized string cannot make a broken
exclusion guard look successful. Trial setup checks the expected lookup count and output for URL,
ordinary Unicode, malformed near-miss, and long-input workloads.

The malformed near-miss fixture records three baseline lookups and four scanner-candidate lookups.
This is an intended classification difference, so preserve the printed fingerprint and do not
describe that row as equivalent-work timing.

Build and materialize the test classpath:

```bash
./mvnw test-compile -Pjmh -pl opennlp-extensions/opennlp-spellcheck -am \
    -Dopennlp.forkCount=1 -Drat.skip=true
./mvnw dependency:build-classpath -Pjmh -pl opennlp-extensions/opennlp-spellcheck \
    -am -Dopennlp.forkCount=1 -Drat.skip=true \
    -DincludeScope=test -Dmdep.outputFile=/tmp/opennlp-jmh-cp.txt
CP="opennlp-extensions/opennlp-spellcheck/target/classes:opennlp-extensions/opennlp-spellcheck/target/test-classes:$(cat /tmp/opennlp-jmh-cp.txt)"
```

Smoke-test the harness:

```bash
java -cp "$CP" org.openjdk.jmh.Main 'SpellcheckExclusionBenchmark' \
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
