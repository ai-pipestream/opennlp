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

# Parser CLI bracket benchmark

A minimal parser implementation returns the flat parse, keeping the timed path focused on `ParserTool.parseLine` preprocessing and token assembly.

Trial setup checks the flat parse and records its token count and a deterministic hash of the text,
token types, and spans. Corrected bracket handling may change this output; review those differences
before comparing timings. Long lines include the cost of constructing the public Parse result,
which can dominate bracket separation. These scores do not isolate the bracket scanner.

Compile the module and benchmark with the repository's existing runtime JMH tooling, then run a smoke check:

```bash
dev/run-parser-cli-benchmark.sh -f 1 -wi 0 -i 1 -r 100ms
```

The launcher adds no module dependencies or plugins. It materializes the existing runtime JMH
classpath, compiles this benchmark directly with `javac`, and loads its generated classes from the
module's `target/jmh-classes` directory. Record that class origin with benchmark results.

To run against an implementation checkout without copying benchmark sources into its review diff,
set `BENCH_IMPLEMENTATION_ROOT` to that checkout. The launcher builds its module and prepends all of
its reactor class directories while retaining the benchmark from this checkout:

```bash
BENCH_IMPLEMENTATION_ROOT=/absolute/path/to/implementation \
    dev/run-parser-cli-benchmark.sh -f 1 -wi 0 -i 1 -r 100ms
```

For measurements, use at least two forks, warmup, several measurement iterations, and the allocation profiler. Save the raw output with the exact commit and JDK. Do not treat a smoke run as performance evidence.

Record the actual loaded class origins, exact branch commit,
harness diff, JDK and raw JMH output. Build and measurement must not overlap with other work.
