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

# Tokenizer character-policy benchmark

Policy and input construction occur during trial setup. The timed method calls the public policy test.

The inventory parameter compares a small explicit policy with the real pinned Unicode 17 Latin
preset. Setup checks the expected eligibility for every input, including decomposed Latin and a
supplementary Latin letter. It prints the inventory size and expected outcome with the run.
Scores include selecting one of two inputs; early rejection visits only the first code point.
This harness depends on the Unicode preset branch and makes no comparison with the former regex.

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
java -cp "$CP" org.openjdk.jmh.Main 'TokenizerCharacterPolicyBenchmark' \
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
