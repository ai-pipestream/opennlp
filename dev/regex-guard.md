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

# Production regex guard

OpenNLP checks compiled production classes with `forbiddenapis`. The guard rejects references to
the `java.util.regex` engine and the regex-backed `String` methods listed in
`src/main/forbiddenapis/regex.txt`. It runs during `process-classes`, after a module's classes have
been compiled.

`RegexNameFinder`, `RegexNameFinderFactory`, and their nested classes are the only production
exception. They implement the public feature that accepts user-supplied regular expressions. The
separate general `forbiddenapis` execution still checks those classes for deprecated and
non-portable APIs.

Run the executable guard fixtures with:

```shell
./mvnw -N -Pregex-guard-fixtures -Drat.skip=true invoker:run
```

Calling the Invoker goal directly avoids running the incomplete baseline through the production
scan. Each fixture inherits the root plugin configuration and uses Invoker's filtered
`@project.version@` value for its parent, so release-version changes cannot resolve an older parent
from the local repository. Post-build scripts require proof that compilation completed, the regex
guard ran, and every expected diagnostic appeared. The negative fixtures must fail their nested
builds, while the lookalike receiver and exact name finder exception fixtures must pass.

For a clean production check after composing the implementation branches, run:

```shell
./mvnw clean verify -Dopennlp.forkCount=1
```

The guard scans OpenNLP's compiled output, not bytecode in third-party dependencies. ClassGraph is
a known dependency whose implementation invokes `java.util.regex`; resolving that dependency is a
separate decision and is required before making a runtime-wide regex-free claim.
