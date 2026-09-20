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
implementation_root=$(CDPATH= cd -- "${BENCH_IMPLEMENTATION_ROOT:-$root}" && pwd)
module=opennlp-core/opennlp-cli
source_file=$root/opennlp-core/opennlp-cli/src/jmh/java/opennlp/tools/cmdline/parser/ParserToolBracketBenchmark.java
classes=$root/$module/target/jmh-classes
generated=$root/$module/target/generated-sources/jmh
dependency_cp=$root/target/runtime-jmh-classpath.txt
module_cp=$root/target/module-benchmark-classpath.txt

"$implementation_root/mvnw" process-classes -pl "$module" -am \
  -DskipTests -Dopennlp.forkCount=1 -Drat.skip=true
"$root/mvnw" -q dependency:build-classpath -Pjmh \
  -pl opennlp-core/opennlp-runtime \
  -DincludeScope=test -Dmdep.outputFile="$dependency_cp"
"$implementation_root/mvnw" -q dependency:build-classpath -pl "$module" \
  -DincludeScope=test -Dmdep.outputFile="$module_cp"

rm -rf "$classes" "$generated"
mkdir -p "$classes" "$generated"

classpath=$(cat "$dependency_cp"):$(cat "$module_cp")
for pom in $(git -C "$root" ls-files '*/pom.xml'); do
  reactor_classes=$root/${pom%/pom.xml}/target/classes
  if test -d "$reactor_classes"; then
    classpath=$reactor_classes:$classpath
  fi
done
if test "$implementation_root" != "$root"; then
  for pom in $(git -C "$implementation_root" ls-files '*/pom.xml'); do
    reactor_classes=$implementation_root/${pom%/pom.xml}/target/classes
    if test -d "$reactor_classes"; then
      classpath=$reactor_classes:$classpath
    fi
  done
fi

javac -cp "$classpath" -processorpath "$classpath" \
  -d "$classes" -s "$generated" "$source_file"
exec java -cp "$classes:$classpath" org.openjdk.jmh.Main ParserToolBracketBenchmark "$@"
