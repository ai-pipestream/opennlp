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

# Unicode 17 Latin tokenizer policy data

`UnicodeLatinTokenizerPolicyGenerator.java` derives the resolved Latin tokenizer-policy ranges
from these official Unicode Character Database 17.0.0 files:

* `https://www.unicode.org/Public/17.0.0/ucd/UnicodeData.txt`
  (`2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c`)
* `https://www.unicode.org/Public/17.0.0/ucd/Scripts.txt`
  (`9f5e50d3abaee7d6ce09480f325c706f485ae3240912527e651954d2d6b035bf`)
* `https://www.unicode.org/Public/17.0.0/ucd/ScriptExtensions.txt`
  (`ec2107e58825a1586acee8e0911ce18260394ac8b87e535ca325f1ccbeb06bc6`)

The generator verifies every checksum. It also verifies the version headers in `Scripts.txt` and
`ScriptExtensions.txt`; `UnicodeData.txt` has no version header, so its pinned checksum supplies
the version check.

Run the generator from the repository root after placing those files in a source directory:

```shell
mkdir -p target/unicode-policy-generator
javac -d target/unicode-policy-generator dev/UnicodeLatinTokenizerPolicyGenerator.java
java -cp target/unicode-policy-generator UnicodeLatinTokenizerPolicyGenerator \
  /path/to/UnicodeData.txt /path/to/Scripts.txt /path/to/ScriptExtensions.txt \
  opennlp-core/opennlp-runtime/src/main/java/opennlp/tools/tokenize/Unicode17LatinTokenizerData.java
```

The inventory contains Unicode 17 code points whose Script is Latin and whose general category is
a Letter. Its continuation marks are marks whose Script Extensions include Latin, unioned with
the marks reached through recursive canonical decomposition of the included letters. Digits are
the explicit ASCII range U+0030 through U+0039. The generator merges adjacent values into sorted
ranges and emits deterministic Java source.

The source data and derived ranges are covered by Unicode License V3, reproduced in the project
`LICENSE`. Attribution is recorded in `NOTICE`. The Java generator itself is Apache License 2.0.
