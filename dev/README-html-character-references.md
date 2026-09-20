<!--
  Licensed to the Apache Software Foundation (ASF) under one or more
  contributor license agreements. See the NOTICE file distributed with
  this work for additional information regarding copyright ownership.
  The ASF licenses this file to You under the Apache License, Version 2.0
  (the "License"); you may not use this file except in compliance with
  the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# HTML character reference data

`HtmlCharacterReferenceData.java` is generated from the WHATWG HTML named character reference
list. The checked input was downloaded from <https://html.spec.whatwg.org/entities.json> on
2026-09-20 and has SHA-256
`d741d877ac77c4194c4ad526b5b4a19aef8dfe411ab840a466891cdbb9f362e6`.

The WHATWG source is Copyright WHATWG (Apple, Google, Mozilla, Microsoft). The
[HTML Living Standard copyright section](https://html.spec.whatwg.org/#copyright) licenses the
specification under Creative Commons Attribution 4.0 International and portions incorporated into
source code under the BSD 3-Clause License. The generated Java table and test vectors reproduce
entity names and Unicode scalar values from that source.

Regenerate from the repository root with JDK 21 or later:

```shell
javac -d target/html-character-generator dev/HtmlCharacterReferencesGenerator.java
java -cp target/html-character-generator HtmlCharacterReferencesGenerator \
  dev/html-character-references/entities.json \
  opennlp-core/opennlp-runtime/src/main/java/opennlp/tools/util/HtmlCharacterReferenceData.java \
  opennlp-core/opennlp-runtime/src/test/resources/opennlp/tools/util/html-character-references.tsv
```

Verify the source before generation:

```shell
sha256sum -c dev/html-character-references/entities.json.sha256
```

Runtime decoding is limited to character references in HTML attribute values. It does not perform
HTML parsing, URL decoding, or recursive entity expansion.
