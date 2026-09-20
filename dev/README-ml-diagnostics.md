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

# Training diagnostics

Data-indexer duration messages use exact integer milliseconds, a locale-independent
decimal point and two fractional digits with HALF_UP rounding. For example, 1005 ms
is reported as `Done indexing in 1.01 s.` even when the JVM's default locale is German.
`ElapsedTimeFormatterTest` contains executable rounding and locale examples.

Fixed progress and malformed-event diagnostics are built directly. An event such
as `outcome/context;` still raises `IllegalArgumentException` identifying the
incomplete `name;value` field. This branch builds on OPENNLP-1989's event parser;
it changes message construction, not event identities or parsing rules.
