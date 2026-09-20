/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package opennlp.tools.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Extracts HTML anchor attributes from directory-index source in source order.
 * Tracks lexical text states and foreign-content namespaces without constructing
 * a document tree. This is not a browser tree builder or an HTML sanitizer:
 * ignored elements, adoption agency repair and foster parenting are not modeled.
 */
final class HtmlLinkScanner {

  private HtmlLinkScanner() {}

  static List<String> links(String html) {
    if (html == null) {
      throw new IllegalArgumentException("html must not be null");
    }
    List<String> links = new ArrayList<>();
    Context context = new Context();
    int cursor = 0;
    while ((cursor = html.indexOf('<', cursor)) >= 0 && cursor + 1 < html.length()) {
      if (html.startsWith("<!--", cursor)) {
        cursor = commentEnd(html, cursor + 4);
      } else if (html.startsWith("<![CDATA[", cursor) && context.inForeignContent()) {
        int end = html.indexOf("]]>", cursor + 9);
        cursor = end < 0 ? html.length() : end + 3;
      } else if (html.charAt(cursor + 1) == '!' || html.charAt(cursor + 1) == '?') {
        // DOCTYPE and bogus declarations end at the first '>'; quotes have no special meaning.
        int end = html.indexOf('>', cursor + 2);
        cursor = end < 0 ? html.length() : end + 1;
      } else {
        boolean closing = html.charAt(cursor + 1) == '/';
        int nameStart = cursor + (closing ? 2 : 1);
        if (nameStart >= html.length() || !asciiLetter(html.charAt(nameStart))) {
          if (closing) {
            int end = html.indexOf('>', nameStart);
            cursor = end < 0 ? html.length() : end + 1;
          } else {
            cursor++;
          }
          continue;
        }
        Tag tag = tag(html, nameStart);
        if (tag == null) {
          break; // EOF inside a tag does not emit a start tag.
        }
        cursor = tag.end;
        if (closing) {
          context.close(tag.name);
          continue;
        }
        boolean htmlElement = context.open(tag);
        if (!htmlElement) {
          continue;
        }
        if (tag.name.equals("a") && tag.href != null) {
          links.add(HtmlCharacterReferences.decodeAttribute(tag.href));
        } else if (tag.name.equals("plaintext")) {
          break;
        } else if (tag.name.equals("script") || rawText(tag.name)) {
          cursor = textEnd(html, cursor, tag.name);
          context.close(tag.name);
        }
      }
    }
    return links;
  }

  /** Parses only attributes needed by link extraction or namespace transitions. */
  private static Tag tag(String html, int cursor) {
    int start = cursor;
    while (cursor < html.length() && !tagDelimiter(html.charAt(cursor))) {
      cursor++;
    }
    String name = asciiName(html, start, cursor);
    String href = null;
    String encoding = null;
    boolean fontAttributes = false;
    while (cursor < html.length()) {
      char ch = html.charAt(cursor);
      if (space(ch)) {
        cursor++;
      } else if (ch == '>') {
        return new Tag(name, cursor + 1, href, encoding, fontAttributes, false);
      } else if (ch == '/') {
        cursor++;
        if (cursor < html.length() && html.charAt(cursor) == '>') {
          return new Tag(name, cursor + 1, href, encoding, fontAttributes, true);
        }
      } else {
        start = cursor++;
        // An initial '=' belongs to the attribute name; subsequent '=' starts its value.
        while (cursor < html.length() && !tagDelimiter(html.charAt(cursor))
            && html.charAt(cursor) != '=') {
          cursor++;
        }
        int nameEnd = cursor;
        while (cursor < html.length() && space(html.charAt(cursor))) {
          cursor++;
        }
        int valueStart = cursor;
        int valueEnd = cursor;
        if (cursor < html.length() && html.charAt(cursor) == '=') {
          cursor++;
          while (cursor < html.length() && space(html.charAt(cursor))) {
            cursor++;
          }
          if (cursor == html.length()) {
            return null;
          }
          ch = html.charAt(cursor);
          if (ch == '\'' || ch == '"') {
            valueStart = ++cursor;
            cursor = html.indexOf(ch, cursor);
            if (cursor < 0) {
              return null;
            }
            valueEnd = cursor++;
          } else {
            valueStart = cursor;
            while (cursor < html.length() && !space(html.charAt(cursor))
                && html.charAt(cursor) != '>') {
              cursor++;
            }
            valueEnd = cursor;
          }
        }
        if (name.equals("a") && href == null && equal(html, start, nameEnd, "href")) {
          href = attribute(html, valueStart, valueEnd);
        } else if (name.equals("annotation-xml") && encoding == null
            && equal(html, start, nameEnd, "encoding")) {
          encoding = HtmlCharacterReferences.decodeAttribute(attribute(html, valueStart, valueEnd));
        } else if (name.equals("font") && (equal(html, start, nameEnd, "color")
            || equal(html, start, nameEnd, "face") || equal(html, start, nameEnd, "size"))) {
          fontAttributes = true;
        }
      }
    }
    return null;
  }

  /** Applies HTML input preprocessing only to retained attribute values. */
  private static String attribute(String html, int start, int end) {
    StringBuilder value = null;
    int copied = start;
    for (int cursor = start; cursor < end; cursor++) {
      char ch = html.charAt(cursor);
      if (ch == '\r' || ch == 0) {
        if (value == null) {
          value = new StringBuilder(end - start);
        }
        value.append(html, copied, cursor).append(ch == 0 ? '\ufffd' : '\n');
        if (ch == '\r' && cursor + 1 < end && html.charAt(cursor + 1) == '\n') {
          cursor++;
        }
        copied = cursor + 1;
      }
    }
    return value == null ? html.substring(start, end) : value.append(html, copied, end).toString();
  }

  private static int commentEnd(String html, int cursor) {
    if (cursor < html.length() && html.charAt(cursor) == '>') {
      return cursor + 1;
    }
    if (html.startsWith("->", cursor)) {
      return cursor + 2;
    }
    while ((cursor = html.indexOf('-', cursor)) >= 0) {
      if (html.startsWith("-->", cursor)) {
        return cursor + 3;
      }
      if (html.startsWith("--!>", cursor)) {
        return cursor + 4;
      }
      cursor++;
    }
    return html.length();
  }

  /** Skips raw/RCDATA text and the normal, escaped and double-escaped script states. */
  private static int textEnd(String html, int cursor, String name) {
    boolean script = name.equals("script");
    int escaped = 0;
    while (cursor < html.length()) {
      if (script && escaped != 0 && html.startsWith("-->", cursor)) {
        escaped = 0;
        cursor += 3;
      } else if (html.charAt(cursor) != '<') {
        cursor++;
      } else if (script && escaped == 0 && html.startsWith("<!--", cursor)) {
        escaped = 1;
        // Revisit the opening dashes: an immediate '>' returns to normal script data.
        cursor += 2;
      } else if (script && escaped == 1 && appropriateName(html, cursor + 1, name)) {
        escaped = 2;
        cursor += name.length() + 1;
      } else if (html.startsWith("</", cursor) && appropriateName(html, cursor + 2, name)) {
        if (escaped == 2) {
          escaped = 1;
          cursor += name.length() + 2;
        } else {
          Tag end = tag(html, cursor + 2);
          return end == null ? html.length() : end.end;
        }
      } else {
        cursor++;
      }
    }
    return html.length();
  }

  private static boolean appropriateName(String html, int start, String name) {
    int end = start + name.length();
    return end < html.length() && tagDelimiter(html.charAt(end)) && equal(html, start, end, name);
  }

  private static boolean rawText(String name) {
    return switch (name) {
      case "style", "textarea", "title", "xmp", "iframe", "noembed", "noframes" -> true;
      default -> false;
    };
  }

  private static boolean space(char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f';
  }

  private static boolean tagDelimiter(char ch) {
    return space(ch) || ch == '/' || ch == '>';
  }

  private static boolean asciiLetter(char ch) {
    return ch >= 'A' && ch <= 'Z' || ch >= 'a' && ch <= 'z';
  }

  private static char lower(char ch) {
    return ch >= 'A' && ch <= 'Z' ? (char) (ch + ('a' - 'A')) : ch;
  }

  private static boolean equal(String text, int start, int end, String expected) {
    if (end - start != expected.length()) {
      return false;
    }
    for (int i = 0; i < expected.length(); i++) {
      if (lower(text.charAt(start + i)) != expected.charAt(i)) {
        return false;
      }
    }
    return true;
  }

  private static String asciiName(String html, int start, int end) {
    char[] chars = new char[end - start];
    for (int i = 0; i < chars.length; i++) {
      char ch = html.charAt(start + i);
      chars[i] = ch == 0 ? '\ufffd' : lower(ch);
    }
    return new String(chars);
  }

  private record Tag(String name, int end, String href, String encoding,
                     boolean fontAttributes, boolean selfClosing) {}

  /** Keeps only open elements inside foreign content, never document nodes or text. */
  private static final class Context {
    private enum Namespace { HTML, SVG, MATH }

    private record Frame(String name, Namespace namespace, boolean integration, int previous) {}

    private final List<Frame> stack = new ArrayList<>();
    private final Map<String, Integer> last = new HashMap<>();

    boolean inForeignContent() {
      return !stack.isEmpty() && stack.getLast().namespace != Namespace.HTML;
    }

    boolean open(Tag tag) {
      Frame parent = stack.isEmpty() ? null : stack.getLast();
      Namespace namespace = parent == null ? Namespace.HTML : parent.namespace;
      if (parent != null && parent.integration
          && !(namespace == Namespace.MATH && !parent.name.equals("annotation-xml")
          && (tag.name.equals("mglyph") || tag.name.equals("malignmark")))) {
        namespace = Namespace.HTML;
      }
      if (namespace != Namespace.HTML && breakout(tag)) {
        while (!stack.isEmpty() && stack.getLast().namespace != Namespace.HTML
            && !stack.getLast().integration) {
          pop();
        }
        namespace = Namespace.HTML;
      }
      if (namespace == Namespace.HTML) {
        if (tag.name.equals("svg")) {
          namespace = Namespace.SVG;
        } else if (tag.name.equals("math")) {
          namespace = Namespace.MATH;
        }
      } else if (namespace == Namespace.MATH && parent != null
          && parent.name.equals("annotation-xml") && tag.name.equals("svg")) {
        namespace = Namespace.SVG;
      }
      if (namespace != Namespace.HTML || !stack.isEmpty()) {
        if (!(namespace != Namespace.HTML && tag.selfClosing)
            && !(namespace == Namespace.HTML && voidElement(tag.name))) {
          int previous = last.getOrDefault(tag.name, -1);
          last.put(tag.name, stack.size());
          stack.add(new Frame(tag.name, namespace, integration(tag, namespace), previous));
        }
      }
      return namespace == Namespace.HTML;
    }

    void close(String name) {
      if (name.equals("p") || name.equals("br")) {
        while (inForeignContent() && !stack.getLast().integration) {
          pop();
        }
      }
      Integer index = last.get(name);
      if (index != null) {
        while (stack.size() > index) {
          pop();
        }
      }
    }

    private void pop() {
      Frame frame = stack.removeLast();
      if (frame.previous < 0) {
        last.remove(frame.name);
      } else {
        last.put(frame.name, frame.previous);
      }
    }

    private static boolean integration(Tag tag, Namespace namespace) {
      if (namespace == Namespace.SVG) {
        return tag.name.equals("foreignobject") || tag.name.equals("desc") || tag.name.equals("title");
      }
      if (namespace == Namespace.MATH) {
        return switch (tag.name) {
          case "mi", "mo", "mn", "ms", "mtext" -> true;
          case "annotation-xml" -> tag.encoding != null
              && (equal(tag.encoding, 0, tag.encoding.length(), "text/html")
              || equal(tag.encoding, 0, tag.encoding.length(), "application/xhtml+xml"));
          default -> false;
        };
      }
      return false;
    }

    private static boolean breakout(Tag tag) {
      return switch (tag.name) {
        case "b", "big", "blockquote", "body", "br", "center", "code", "dd", "div", "dl", "dt",
             "em", "embed", "h1", "h2", "h3", "h4", "h5", "h6", "head", "hr", "i", "img", "li",
             "listing", "menu", "meta", "nobr", "ol", "p", "pre", "ruby", "s", "small", "span",
             "strong", "strike", "sub", "sup", "table", "tt", "u", "ul", "var" -> true;
        case "font" -> tag.fontAttributes;
        default -> false;
      };
    }

    private static boolean voidElement(String name) {
      return switch (name) {
        case "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
             "param", "source", "track", "wbr" -> true;
        default -> false;
      };
    }
  }
}
