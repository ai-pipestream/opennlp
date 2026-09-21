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

# Unicode 17 emoji sequence data

`UnicodeEmojiSequenceGenerator.java` derives the runtime emoji inventory from these official
Unicode 17.0 files:

* `https://www.unicode.org/Public/17.0.0/emoji/emoji-test.txt`
  (`1d8a944f88d7952f7ef7c5167fef3c67995bcae24543949710231b03a201acda`)
* `https://www.unicode.org/Public/17.0.0/ucd/emoji/emoji-data.txt`
  (`2cb2bb9455cda83e8481541ecf5b6dfda66a3bb89efa3fa7c5297eccf607b72b`)

The generator verifies both SHA-256 checksums and both Unicode version headers. It retains exact
`fully-qualified` sequences from `emoji-test.txt` and the `Emoji_Component` ranges used to identify
structurally connected malformed candidates. Component membership alone never classifies text as
emoji.

Run the generator from the repository root:

```shell
mkdir -p target/unicode-emoji-generator
javac -d target/unicode-emoji-generator dev/UnicodeEmojiSequenceGenerator.java
java -cp target/unicode-emoji-generator UnicodeEmojiSequenceGenerator \
  /path/to/emoji-test.txt /path/to/emoji-data.txt \
  opennlp-core/opennlp-runtime/src/main/resources/opennlp/tools/util/normalizer/emoji/EmojiSequences-17.0.txt
```

The generated inventory and its sources are covered by Unicode License V3, reproduced in the
project `LICENSE`. Attribution is recorded in `NOTICE`. The generator is Apache License 2.0.
The runtime module's `src/main/appended-resources/META-INF/LICENSE` and `NOTICE` also carry
the data license and attribution into the standalone runtime JAR.
