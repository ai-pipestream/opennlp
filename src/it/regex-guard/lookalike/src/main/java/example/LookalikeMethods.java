/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package example;

final class LookalikeMethods {
  boolean matches(String value) { return !value.isEmpty(); }
  String[] split(String value) { return new String[] {value}; }
  String replaceAll(String value, String replacement) { return replacement; }
  String replaceFirst(String value, String replacement) { return value; }
  String format(String value, Object... args) { return value; }
  String formatted(Object... args) { return args.length == 0 ? "" : args[0].toString(); }
  String printf(String value, Object... args) { return value; }
  String getPathMatcher(String value) { return value; }
  String newDirectoryStream(String path, String glob) { return path + glob; }

  Object[] use(String value) {
    return new Object[] {matches(value), split(value), replaceAll(value, "x"), replaceFirst(value, "x"),
        format(value, 1), formatted(value), printf(value, 1), getPathMatcher(value),
        newDirectoryStream(value, "*")};
  }
}
