/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 */
def log = new File(basedir, "build.log").text
assert log.contains("Compiling 1 source file")
assert log.contains("forbiddenapis:3.10:check (regex-free-production)")
assert log.contains("BUILD FAILURE")
[
    "java.lang.String#matches(java.lang.String)",
    "java.lang.String#replaceAll(java.lang.String,java.lang.String)",
    "java.lang.String#replaceFirst(java.lang.String,java.lang.String)",
    "java.lang.String#split(java.lang.String)",
    "java.lang.String#split(java.lang.String,int)",
    "java.lang.String#splitWithDelimiters(java.lang.String,int)"
].each {
  assert log.count("Forbidden method invocation: ${it}") == 3
}
return true
