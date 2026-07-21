/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
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

package opennlp.tools.noise;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.function.Predicate;

import opennlp.tools.util.Span;
import opennlp.tools.util.StringUtil;

/**
 * The built-in {@link NoiseScorer}: structural signals over whitespace-delimited
 * tokens, a single cursor pass with no regular expression and no bundled data.
 *
 * <p>The signals are deliberately conservative, calibrated so the most
 * consonant-heavy legitimate English words stay below every threshold: a token only
 * scores {@link NoiseSpan#SEVERITY_GIBBERISH} when at least two independent signals
 * agree, and one signal alone means {@link NoiseSpan#SEVERITY_DAMAGED}. Structural
 * signals apply to ASCII-letter tokens only, so text in other scripts is never
 * flagged by them. A stretch shaped like an encoded binary rather than language
 * scores {@link NoiseSpan#SEVERITY_BINARYISH}; spans a caller excludes, such as
 * assets another detector already explained, are not scored at all.</p>
 *
 * <p>{@link NoiseSpan#SEVERITY_MISSPELLED} requires a dictionary: a token the
 * dictionary rejects but a single known confusion repair turns into a dictionary word
 * is reported as misspelled, the shape optical recognition damage takes. Without a
 * dictionary that tier never fires. A token the dictionary accepts is never flagged,
 * whatever its structure.</p>
 *
 * <p>Known conservatism: a mash of ordinary syllables with normal vowel spacing
 * passes the structural signals; catching it needs a character-level language model,
 * which is a planned addition behind this same interface. Tokens interleaving letters
 * and digits heavily are reported as damaged, which can flag technical identifiers in
 * running prose.</p>
 *
 * <p>The scorer is stateless beyond its dictionary and safe for concurrent use when
 * the dictionary is.</p>
 *
 * @since 3.0.0
 */
public final class StructuralNoiseScorer implements NoiseScorer {

  /** Tokens at least this long with base64 shape score as binary-ish. */
  private static final int BINARYISH_MIN_LENGTH = 24;

  /**
   * Case flips marking base64 shape in a pure-letter run. A camel-case identifier
   * yields two flips per hump plus one, so even a four-hump name stays at nine;
   * random-case letters flip about every second character.
   */
  private static final int BINARYISH_MIN_CASE_FLIPS = 12;

  /** The longest consonant run legitimate English reaches is six. */
  private static final int CONSONANT_RUN_SIGNAL = 7;

  /** No English word repeats one character four times in a row. */
  private static final int REPEAT_RUN_SIGNAL = 4;

  /** Below this vowel share a long token is vowel-starved; strengths is 0.111. */
  private static final double LOW_VOWEL_RATIO = 0.10;

  /** Letter-digit alternations marking interleaved damage. */
  private static final int INTERLEAVE_SIGNAL = 4;

  /**
   * The confusion pairs of optical recognition damage, as from-to sibling entries.
   * Project-authored from the classic confusions; both directions are listed where
   * both occur in practice.
   */
  private static final String[][] CONFUSIONS = {
      {"rn", "m"}, {"m", "rn"},
      {"vv", "w"}, {"w", "vv"},
      {"cl", "d"},
      {"1", "l"}, {"l", "1"},
      {"0", "o"}, {"o", "0"},
      {"5", "s"},
  };

  private final Predicate<CharSequence> dictionary;

  /** Initializes the scorer without a dictionary; the misspelled tier never fires. */
  public StructuralNoiseScorer() {
    this.dictionary = null;
  }

  /**
   * Initializes the scorer with a dictionary.
   *
   * @param dictionary Accepts the lowercase form of a known word. Must not be
   *                   {@code null}.
   * @throws IllegalArgumentException Thrown if {@code dictionary} is {@code null}.
   */
  public StructuralNoiseScorer(Predicate<CharSequence> dictionary) {
    if (dictionary == null) {
      throw new IllegalArgumentException("dictionary must not be null");
    }
    this.dictionary = dictionary;
  }

  @Override
  public List<NoiseSpan> score(CharSequence text, Collection<Span> exclude) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    if (exclude == null) {
      throw new IllegalArgumentException("exclude must not be null");
    }
    for (final Span span : exclude) {
      if (span == null) {
        throw new IllegalArgumentException("exclude must not contain null");
      }
    }
    final List<NoiseSpan> tokens = new ArrayList<>();
    final int length = text.length();
    int i = 0;
    while (i < length) {
      if (StringUtil.isWhitespace(text.charAt(i))) {
        i++;
        continue;
      }
      final int start = i;
      while (i < length && !StringUtil.isWhitespace(text.charAt(i))) {
        i++;
      }
      if (!overlapsAny(start, i, exclude)) {
        final NoiseSpan scored = scoreToken(text, start, i);
        if (scored != null) {
          tokens.add(scored);
        }
      }
    }
    return merge(tokens, text);
  }

  /**
   * Scores one token.
   *
   * @param text The text.
   * @param start The token start.
   * @param end The token end.
   * @return The finding, or {@code null} for a clean token.
   */
  private NoiseSpan scoreToken(CharSequence text, int start, int end) {
    final String token = text.subSequence(start, end).toString();
    final String core = trimPunctuation(token);
    if (core.length() < 3 || !isAscii(core)) {
      return null;
    }
    final int coreStart = start + token.indexOf(core);
    final Span span = new Span(coreStart, coreStart + core.length());
    if (dictionary != null && dictionary.test(core.toLowerCase(Locale.ROOT))) {
      return null;
    }
    final NoiseSpan binaryish = binaryish(core, span);
    if (binaryish != null) {
      return binaryish;
    }
    final NoiseSpan structural = structural(core, span);
    if (structural != null) {
      return structural;
    }
    return misspelled(core, span);
  }

  /**
   * The binary-shape tier: long, base64-alphabet, and unlike a word or an identifier.
   *
   * @param core The token without surrounding punctuation.
   * @param span The token span.
   * @return The finding, or {@code null}.
   */
  private static NoiseSpan binaryish(String core, Span span) {
    if (core.length() < BINARYISH_MIN_LENGTH) {
      return null;
    }
    boolean digit = false;
    boolean symbol = false;
    int caseFlips = 0;
    boolean lastUpper = false;
    boolean lastLetter = false;
    for (int i = 0; i < core.length(); i++) {
      final char c = core.charAt(i);
      final boolean upper = c >= 'A' && c <= 'Z';
      final boolean lower = c >= 'a' && c <= 'z';
      if (c >= '0' && c <= '9') {
        digit = true;
      } else if (c == '+' || c == '/' || c == '=') {
        symbol = true;
      } else if (!upper && !lower) {
        return null;
      }
      if ((upper || lower) && lastLetter && upper != lastUpper) {
        caseFlips++;
      }
      lastLetter = upper || lower;
      lastUpper = upper;
    }
    if (digit || symbol || caseFlips >= BINARYISH_MIN_CASE_FLIPS) {
      return new NoiseSpan(span, NoiseSpan.SEVERITY_BINARYISH,
          Math.min(1.0, core.length() / 48.0));
    }
    return null;
  }

  /**
   * The structural tiers over an ASCII token: two agreeing signals are gibberish, one
   * is damage.
   *
   * @param core The token without surrounding punctuation.
   * @param span The token span.
   * @return The finding, or {@code null}.
   */
  private static NoiseSpan structural(String core, Span span) {
    int letters = 0;
    int vowels = 0;
    int consonantRun = 0;
    int maxConsonantRun = 0;
    int repeatRun = 1;
    int maxRepeatRun = 1;
    int interleave = 0;
    boolean lastDigit = false;
    boolean lastLetter = false;
    char last = 0;
    for (int i = 0; i < core.length(); i++) {
      final char c = Character.toLowerCase(core.charAt(i));
      final boolean letter = c >= 'a' && c <= 'z';
      final boolean digit = c >= '0' && c <= '9';
      if (letter) {
        letters++;
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y') {
          vowels++;
          consonantRun = 0;
        } else {
          consonantRun++;
          maxConsonantRun = Math.max(maxConsonantRun, consonantRun);
        }
      } else {
        consonantRun = 0;
      }
      if ((letter && lastDigit) || (digit && lastLetter)) {
        interleave++;
      }
      repeatRun = c == last ? repeatRun + 1 : 1;
      maxRepeatRun = Math.max(maxRepeatRun, repeatRun);
      last = c;
      lastDigit = digit;
      lastLetter = letter;
    }
    int signals = 0;
    if (letters >= 5 && vowels == 0) {
      signals++;
    }
    if (letters >= 8 && vowels > 0 && (double) vowels / letters <= LOW_VOWEL_RATIO) {
      signals++;
    }
    if (maxConsonantRun >= CONSONANT_RUN_SIGNAL) {
      signals++;
    }
    if (maxRepeatRun >= REPEAT_RUN_SIGNAL) {
      signals++;
    }
    if (core.length() >= 8 && interleave >= INTERLEAVE_SIGNAL) {
      signals++;
    }
    if (signals >= 2) {
      return new NoiseSpan(span, NoiseSpan.SEVERITY_GIBBERISH,
          Math.min(1.0, signals / 4.0));
    }
    if (signals == 1) {
      return new NoiseSpan(span, NoiseSpan.SEVERITY_DAMAGED, 0.5);
    }
    return null;
  }

  /**
   * The misspelled tier: one confusion repair reaches a dictionary word.
   *
   * @param core The token without surrounding punctuation.
   * @param span The token span.
   * @return The finding, or {@code null} without a dictionary or a repairing
   *         confusion.
   */
  private NoiseSpan misspelled(String core, Span span) {
    if (dictionary == null) {
      return null;
    }
    final String lower = core.toLowerCase(Locale.ROOT);
    for (final String[] confusion : CONFUSIONS) {
      final String from = confusion[0];
      final String to = confusion[1];
      int at = lower.indexOf(from);
      while (at >= 0) {
        final String candidate =
            lower.substring(0, at) + to + lower.substring(at + from.length());
        if (dictionary.test(candidate)) {
          return new NoiseSpan(span, NoiseSpan.SEVERITY_MISSPELLED, 0.9);
        }
        at = lower.indexOf(from, at + 1);
      }
    }
    return null;
  }

  /**
   * Merges adjacent findings separated by nothing but whitespace into one span of the
   * worst severity.
   *
   * @param findings The per-token findings in order.
   * @param text The text, to check that only whitespace separates neighbors.
   * @return The merged findings. Never {@code null}.
   */
  private static List<NoiseSpan> merge(List<NoiseSpan> findings, CharSequence text) {
    final List<NoiseSpan> merged = new ArrayList<>();
    for (final NoiseSpan finding : findings) {
      if (!merged.isEmpty()) {
        final NoiseSpan previous = merged.get(merged.size() - 1);
        if (onlyWhitespaceBetween(text, previous.span().getEnd(),
            finding.span().getStart())) {
          merged.set(merged.size() - 1, new NoiseSpan(
              new Span(previous.span().getStart(), finding.span().getEnd()),
              worse(previous.severity(), finding.severity()),
              Math.max(previous.score(), finding.score())));
          continue;
        }
      }
      merged.add(finding);
    }
    return merged;
  }

  /**
   * Whether a region holds only whitespace.
   *
   * @param text The text.
   * @param from The inclusive start.
   * @param to The exclusive end.
   * @return {@code true} if every character in between is whitespace.
   */
  private static boolean onlyWhitespaceBetween(CharSequence text, int from, int to) {
    for (int i = from; i < to; i++) {
      if (!StringUtil.isWhitespace(text.charAt(i))) {
        return false;
      }
    }
    return true;
  }

  /**
   * Picks the worse of two severities in this scorer's tier order.
   *
   * @param a One severity.
   * @param b The other.
   * @return The worse one.
   */
  private static String worse(String a, String b) {
    return rank(a) >= rank(b) ? a : b;
  }

  /**
   * Orders this scorer's tiers.
   *
   * @param severity The severity.
   * @return Its rank, higher is worse.
   */
  private static int rank(String severity) {
    return switch (severity) {
      case NoiseSpan.SEVERITY_BINARYISH -> 4;
      case NoiseSpan.SEVERITY_GIBBERISH -> 3;
      case NoiseSpan.SEVERITY_DAMAGED -> 2;
      default -> 1;
    };
  }

  /**
   * Strips leading and trailing ASCII punctuation, keeping the word core.
   *
   * @param token The whitespace-delimited token.
   * @return The core, possibly empty.
   */
  private static String trimPunctuation(String token) {
    int start = 0;
    int end = token.length();
    while (start < end && isAsciiPunctuation(token.charAt(start))) {
      start++;
    }
    while (end > start && isAsciiPunctuation(token.charAt(end - 1))) {
      end--;
    }
    return token.substring(start, end);
  }

  /**
   * Whether the character is ASCII punctuation that surrounds words, deliberately
   * excluding the base64 alphabet's {@code +}, {@code /}, and {@code =}.
   *
   * @param c The character.
   * @return {@code true} for surrounding punctuation.
   */
  private static boolean isAsciiPunctuation(char c) {
    return c == '.' || c == ',' || c == ';' || c == ':' || c == '!' || c == '?'
        || c == '"' || c == '\'' || c == '(' || c == ')' || c == '[' || c == ']'
        || c == '{' || c == '}';
  }

  /**
   * Whether every character is ASCII.
   *
   * @param token The token.
   * @return {@code true} when no character exceeds 0x7F.
   */
  private static boolean isAscii(String token) {
    for (int i = 0; i < token.length(); i++) {
      if (token.charAt(i) > 0x7F) {
        return false;
      }
    }
    return true;
  }

  /**
   * Whether a token region overlaps any excluded span.
   *
   * @param start The token start.
   * @param end The token end.
   * @param exclude The excluded spans.
   * @return {@code true} on any overlap.
   */
  private static boolean overlapsAny(int start, int end, Collection<Span> exclude) {
    for (final Span span : exclude) {
      if (start < span.getEnd() && span.getStart() < end) {
        return true;
      }
    }
    return false;
  }
}
