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

package opennlp.tools.cmdline;

final class ReportLayout {

  private ReportLayout() {
  }

  static String left(Object value, int width) {
    return pad(value, width, false);
  }

  static String right(Object value, int width) {
    return pad(value, width, true);
  }

  /**
   * Counts Unicode code points for logical report columns. This does not attempt to model
   * grapheme clusters or terminal display-cell widths.
   */
  static int width(String value) {
    return value.codePointCount(0, value.length());
  }

  private static String pad(Object value, int width, boolean before) {
    String text = String.valueOf(value);
    int padding = width - width(text);
    if (padding <= 0) {
      return text;
    }
    String spaces = " ".repeat(padding);
    return before ? spaces + text : text + spaces;
  }
}
