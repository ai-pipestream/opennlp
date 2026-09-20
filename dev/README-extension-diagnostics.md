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

# Extension diagnostics

SymSpell binary model errors use `HexFormat` to display the received and expected
32-bit signatures as eight uppercase hexadecimal digits. For example, an input
signature of zero reports `magic was 0x00000000, expected 0x53594D53`.

The UIMA name finder constructs its invalid type-mapping warning directly,
retaining the original mapping text. Neither diagnostic invokes Java's general
formatter, whose Java 21 implementation uses regex. Model decoding and UIMA
mapping rules are unchanged.
