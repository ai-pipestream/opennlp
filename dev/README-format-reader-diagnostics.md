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

# Corpus reader diagnostics

CoNLL-U, brat and Leipzig reader errors construct their fixed messages directly.
They do not invoke Java's general formatter, which uses regex on Java 21.
Exception types, document identifiers and nested causes are retained.

For example, a CoNLL-U sentence whose text is `café 😀` and whose word line
contains `absent%😀` reports the unmatched token and the original sentence text.
Percent signs and supplementary Unicode characters remain literal diagnostic
content. The `ConlluTokenSampleStreamTest` and `BratAnnotationStreamTest` cases
provide executable examples, including preservation of a brat parser's cause.
