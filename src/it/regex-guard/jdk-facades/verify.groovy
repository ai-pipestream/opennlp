/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
def log = new File(basedir, "build.log").text
assert log.contains("Compiling 1 source file")
assert log.contains("forbiddenapis:3.10:check (regex-free-production)")
assert log.contains("BUILD FAILURE")
[
    "java.nio.file.Files#newDirectoryStream(java.nio.file.Path,java.lang.String)",
    "java.nio.file.FileSystem#getPathMatcher(java.lang.String)",
    "java.lang.String#format(java.lang.String,java.lang.Object[])",
    "java.lang.String#format(java.util.Locale,java.lang.String,java.lang.Object[])",
    "java.lang.String#formatted(java.lang.Object[])",
    "java.io.PrintStream#printf(java.lang.String,java.lang.Object[])",
    "java.io.PrintStream#printf(java.util.Locale,java.lang.String,java.lang.Object[])",
    "java.io.PrintStream#format(java.lang.String,java.lang.Object[])",
    "java.io.PrintStream#format(java.util.Locale,java.lang.String,java.lang.Object[])",
    "java.io.PrintWriter#printf(java.lang.String,java.lang.Object[])",
    "java.io.PrintWriter#printf(java.util.Locale,java.lang.String,java.lang.Object[])",
    "java.io.PrintWriter#format(java.lang.String,java.lang.Object[])",
    "java.io.PrintWriter#format(java.util.Locale,java.lang.String,java.lang.Object[])",
    "java.io.Console#printf(java.lang.String,java.lang.Object[])",
    "java.io.Console#format(java.lang.String,java.lang.Object[])"
].each {
  assert log.contains("Forbidden method invocation: ${it}")
}
["java.util.Scanner", "java.util.Formatter"].each {
  assert log.contains("Forbidden class/interface use: ${it}")
}
return true
