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

Compile the module and benchmark with the repository's existing runtime JMH tooling, then run a smoke check:

```bash
dev/run-spellcheck-benchmark.sh -f 1 -wi 0 -i 1 -r 100ms
```

The launcher adds no module dependencies or plugins. It materializes the existing runtime JMH
classpath, compiles this benchmark directly with `javac`, and loads its generated classes from the
module's `target/jmh-classes` directory. Record that class origin with benchmark results.

To run against an implementation checkout without copying benchmark sources into its review diff,
set `BENCH_IMPLEMENTATION_ROOT` to that checkout. The launcher builds its module and prepends all of
its reactor class directories while retaining the benchmark from this checkout:

```bash
BENCH_IMPLEMENTATION_ROOT=/absolute/path/to/implementation \
    dev/run-spellcheck-benchmark.sh -f 1 -wi 0 -i 1 -r 100ms
```

For measurements, use at least two forks, warmup, several measurement iterations, and the allocation profiler. Save the raw output with the exact commit and JDK. Do not treat a smoke run as performance evidence.

Record the actual loaded class origins, exact branch commit,
harness diff, JDK and raw JMH output. Build and measurement must not overlap with other work.
