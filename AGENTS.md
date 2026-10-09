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

# Notes for coding agents

This file is for AI coding agents and the people who run them (see https://agents.md).
Human contributors should read [CONTRIBUTING.md](.github/CONTRIBUTING.md) first. Keep this
file short; it will change as the format settles.

## Build

- JDK 21 or newer and Maven 3.9 or newer (use `./mvnw`).
- Full build with tests: `./mvnw clean install`.
- One module and what it needs: `./mvnw install -pl <module> -am`.
- Checkstyle, Apache RAT and forbiddenapis run in the normal build. A change is not
  done until `./mvnw clean install` passes for the modules it touches.
- Some tests download models. Without network access, run with `-DskipITs` and say so
  in the pull request.
- `opennlp-eval-tests` needs external data sets and is not part of a normal check.

## Layout

- `opennlp-api`: public interfaces and shared utilities (`StringUtil`, `ParamChecks`).
- `opennlp-core`: the implementations (`opennlp-runtime`, `opennlp-formats`,
  `opennlp-cli`, `opennlp-model-resolver`, and the machine learning modules under
  `opennlp-ml`, including the ONNX modules `opennlp-dl` and `opennlp-dl-gpu`).
- `opennlp-extensions`: Morfologik, spell checking and UIMA.
- `opennlp-docs`: the DocBook manual. Update it when behavior a user can see changes.

## Code rules reviewers check

- Every new file starts with the Apache license header. Copy it from a file next to it.
- Check method arguments with `opennlp.tools.util.ParamChecks` (it throws
  `IllegalArgumentException`), not `Objects.requireNonNull`.
- Text is Unicode. Use the `StringUtil` helpers such as `isUnicodeBlank`,
  `isUnicodeWhitespace` and `trimUnicodeWhitespace` instead of `String.isBlank`,
  `String.trim` or `Character.isWhitespace`. Walk text by code point when it can hold
  characters outside the Basic Multilingual Plane.
- Do not add regular expressions (`java.util.regex`, `String.split`, `matches`,
  `replaceAll`, `replaceFirst`) to production code. Use `StringUtil.split`,
  `StringUtil.splitNonEmpty` or a small scanner.
- Case conversion takes a locale: `toLowerCase(Locale.ROOT)`.
- Before writing a helper, look in `StringUtil`, `ParamChecks` and the `opennlp.tools.util`
  packages; reuse what is there.
- New utility classes are `final`, have a private constructor and only static methods.
- Keep helpers `private` and non-static unless another class needs them.
- Constants are `static final` with `UPPER_SNAKE_CASE` names. A value used once stays inline;
  a small private method reads better than a constant that only holds a prefix or format.
- No mutable static state. A shared instance such as `WhitespaceTokenizer.INSTANCE` must not
  be changed at runtime. Mark classes that are safe to share with `@ThreadSafe`.
- Do not change public API without a deprecation path, and mention any behavior change
  in the pull request.
- Match the code around you: names, comment density and test style. Do not reformat
  lines the change does not need.

## Tests

- Bug fixes come with a test that fails before the fix.
- JUnit 5. Put tests in the module of the class under test, in the same package.
- Do not disable or skip a test to make a build pass.

## Issues, commits and pull requests

- Every change has a GitHub issue. Start commit subjects with `GH-<n>:`.
- One change per pull request. Keep unrelated cleanups out.
- Rebase on `main` and squash before asking for review, and fill in the pull request
  template.
- Say in the pull request that an AI tool helped, and review every line yourself before
  you ask anyone else to.
