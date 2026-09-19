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

import java.util.regex.MatchResult;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

final class EngineUse {
  boolean matches(String expression, String value) {
    try {
      Pattern pattern = java.util.regex.Pattern.compile(
          expression);
      Matcher matcher = pattern.matcher(value);
      MatchResult result = matcher.toMatchResult();
      return matcher.matches() && result.start() == 0;
    } catch (PatternSyntaxException exception) {
      throw new PatternSyntaxException(exception.getDescription(), expression, exception.getIndex());
    }
  }
}
