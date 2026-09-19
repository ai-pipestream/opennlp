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

import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;

final class StringMethods {
  Object[] calls(String value, String expression) {
    String[] split = value.split(
        expression);
    String[] limited = value.split(expression, 2);
    String[] delimited = value.splitWithDelimiters(expression, 2);
    boolean matches = value.matches(expression);
    String replaced = value.replaceAll(expression, "x");
    String first = value.replaceFirst(expression, "x");
    return new Object[] {split, limited, delimited, matches, replaced, first};
  }

  Object[] references(String value) {
    Predicate<String> matches = value::matches;
    Function<String, String[]> split = value::split;
    BiFunction<String, Integer, String[]> limited = value::split;
    BiFunction<String, Integer, String[]> delimited = value::splitWithDelimiters;
    BiFunction<String, String, String> replaceAll = value::replaceAll;
    BiFunction<String, String, String> replaceFirst = value::replaceFirst;
    return new Object[] {matches, split, limited, delimited, replaceAll, replaceFirst};
  }

  Object[] unboundReferences() {
    BiFunction<String, String, Boolean> matches = String::matches;
    BiFunction<String, String, String[]> split = String::split;
    StringIntFunction<String[]> limited = String::split;
    StringIntFunction<String[]> delimited = String::splitWithDelimiters;
    StringStringFunction<String> replaceAll = String::replaceAll;
    StringStringFunction<String> replaceFirst = String::replaceFirst;
    return new Object[] {matches, split, limited, delimited, replaceAll, replaceFirst};
  }

  interface StringIntFunction<T> {
    T apply(String value, String expression, int limit);
  }

  interface StringStringFunction<T> {
    T apply(String value, String expression, String replacement);
  }
}
