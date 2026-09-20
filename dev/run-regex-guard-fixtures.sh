#!/bin/sh
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work=$root/target/regex-guard-it
version=$(awk '/<artifactId>opennlp<\/artifactId>/{found=1; next}
  found && /<version>/{line=$0; sub(/^.*<version>/, "", line); sub(/<\/version>.*$/, "", line); print line; exit}' "$root/pom.xml")

test -n "$version" || { echo "Could not read project version" >&2; exit 1; }
rm -rf "$work"
mkdir -p "$work"

prepare() {
  name=$1
  rm -rf "$work/$name"
  cp -R "$root/src/it/regex-guard/$name" "$work/$name"
  sed "s/@project.version@/$version/g" "$work/$name/pom.xml" > "$work/$name/pom.xml.tmp"
  mv "$work/$name/pom.xml.tmp" "$work/$name/pom.xml"
}

require_text() {
  text=$1
  file=$2
  grep -Fq "$text" "$file" || { echo "Missing from $file: $text" >&2; return 1; }
}

reject_text() {
  text=$1
  file=$2
  if grep -Fq "$text" "$file"; then
    echo "Unexpected in $file: $text" >&2
    return 1
  fi
}

require_count() {
  expected=$1
  text=$2
  file=$3
  actual=$(grep -Fc "$text" "$file" || true)
  test "$actual" -eq "$expected" || {
    echo "Expected $expected occurrences in $file, found $actual: $text" >&2
    return 1
  }
}

run_fixture() {
  name=$1
  expected=$2
  extra=${3-}
  log=$work/$name/build.log
  prepare "$name"
  set +e
  # shellcheck disable=SC2086
  "$root/mvnw" -f "$work/$name/pom.xml" clean verify -Dopennlp.forkCount=1 $extra > "$log" 2>&1
  status=$?
  set -e
  if { test "$expected" = success && test "$status" -ne 0; } ||
      { test "$expected" = failure && test "$status" -eq 0; }; then
    cat "$log" >&2
    echo "Unexpected $name exit status: $status" >&2
    exit 1
  fi
  require_text "forbiddenapis:3.10:check (regex-free-production)" "$log"
  if test "$expected" = success; then
    require_text "BUILD SUCCESS" "$log"
  else
    require_text "BUILD FAILURE" "$log"
  fi
}

verify_engine() {
  log=$1
  require_text "Compiling 1 source file" "$log"
  for type in Pattern Matcher MatchResult PatternSyntaxException; do
    require_text "Forbidden class/interface use: java.util.regex.$type" "$log"
  done
}

verify_string_methods() {
  log=$1
  require_text "Compiling 1 source file" "$log"
  for api in \
    'java.lang.String#matches(java.lang.String)' \
    'java.lang.String#replaceAll(java.lang.String,java.lang.String)' \
    'java.lang.String#replaceFirst(java.lang.String,java.lang.String)' \
    'java.lang.String#split(java.lang.String)' \
    'java.lang.String#split(java.lang.String,int)' \
    'java.lang.String#splitWithDelimiters(java.lang.String,int)'; do
    require_count 3 "Forbidden method invocation: $api" "$log"
  done
}

verify_jdk_facades() {
  log=$1
  require_text "Compiling 1 source file" "$log"
  for api in \
    'java.nio.file.Files#newDirectoryStream(java.nio.file.Path,java.lang.String)' \
    'java.nio.file.FileSystem#getPathMatcher(java.lang.String)' \
    'java.lang.String#format(java.lang.String,java.lang.Object[])' \
    'java.lang.String#format(java.util.Locale,java.lang.String,java.lang.Object[])' \
    'java.lang.String#formatted(java.lang.Object[])' \
    'java.io.PrintStream#printf(java.lang.String,java.lang.Object[])' \
    'java.io.PrintStream#printf(java.util.Locale,java.lang.String,java.lang.Object[])' \
    'java.io.PrintStream#format(java.lang.String,java.lang.Object[])' \
    'java.io.PrintStream#format(java.util.Locale,java.lang.String,java.lang.Object[])' \
    'java.io.PrintWriter#printf(java.lang.String,java.lang.Object[])' \
    'java.io.PrintWriter#printf(java.util.Locale,java.lang.String,java.lang.Object[])' \
    'java.io.PrintWriter#format(java.lang.String,java.lang.Object[])' \
    'java.io.PrintWriter#format(java.util.Locale,java.lang.String,java.lang.Object[])' \
    'java.io.Console#printf(java.lang.String,java.lang.Object[])' \
    'java.io.Console#format(java.lang.String,java.lang.Object[])'; do
    require_text "Forbidden method invocation: $api" "$log"
  done
  require_text "Forbidden class/interface use: java.util.Scanner" "$log"
  require_text "Forbidden class/interface use: java.util.Formatter" "$log"
}

run_fixture engine failure
verify_engine "$work/engine/build.log"
run_fixture string-methods failure
verify_string_methods "$work/string-methods/build.log"
cp "$work/string-methods/build.log" "$work/string-methods-complete.log"
run_fixture jdk-facades failure
verify_jdk_facades "$work/jdk-facades/build.log"
run_fixture near-exempt failure
require_text "Compiling 1 source file" "$work/near-exempt/build.log"
require_text "Forbidden class/interface use: java.util.regex.Pattern" "$work/near-exempt/build.log"
require_text "in opennlp.tools.namefind.RegexNameFinderHelper" "$work/near-exempt/build.log"
run_fixture exempt success
require_text "No classes found" "$work/exempt/build.log"
reject_text "Forbidden class/interface use:" "$work/exempt/build.log"
reject_text "Forbidden method invocation:" "$work/exempt/build.log"
run_fixture lookalike success
require_text "Scanned 1 class file(s) for forbidden API invocations" "$work/lookalike/build.log"
reject_text "Forbidden method invocation:" "$work/lookalike/build.log"

mutation=$work/regex-missing-split-with-delimiters.txt
grep -Fv 'java.lang.String#splitWithDelimiters(java.lang.String,int)' \
  "$root/src/main/forbiddenapis/regex.txt" > "$mutation"
run_fixture string-methods failure "-Dregex.forbiddenapis.signatures=$mutation"
cp "$work/string-methods/build.log" "$work/string-methods-mutation.log"
if verify_string_methods "$work/string-methods/build.log" >/dev/null 2>&1; then
  echo "Mutation unexpectedly satisfied all diagnostic assertions" >&2
  exit 1
fi
echo "Regex guard fixtures and missing-signature mutation passed"
