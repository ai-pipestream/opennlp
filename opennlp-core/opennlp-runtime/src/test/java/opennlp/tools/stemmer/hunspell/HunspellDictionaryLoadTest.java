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

package opennlp.tools.stemmer.hunspell;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.tools.stemmer.hunspell.HunspellDictionary.LoadMode;
import opennlp.tools.stemmer.hunspell.HunspellDictionary.UnsupportedDirective;

import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.COMPOUND;
import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.PLURAL;
import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.SET_UTF8;
import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.stream;

/** Tests the loading policy with project-authored affix and dictionary content. */
class HunspellDictionaryLoadTest {

  private static final String WORDS = "1\ndog/A\n";
  private static final String UNKNOWN = "UNKNOWN_DIRECTIVE";
  private static final String UNRECOGNIZED = "UNRECOGNIZED";
  private static final String FLAG_VALUE = " X";
  private static final String AFFIX_STREAM = "affix stream";

  @TempDir
  private Path directory;

  /**
   * Rejects directive names Hunspell does not read, with or without arguments.
   *
   * @param line An unknown directive with its arguments.
   */
  @ParameterizedTest
  @ValueSource(strings = {UNKNOWN + " 1", UNKNOWN + " a b", UNRECOGNIZED, UNRECOGNIZED + " value"})
  void testUnsupportedDirectiveFailsByDefault(String line) {
    final IOException error = Assertions.assertThrows(IOException.class,
        () -> HunspellDictionary.load(stream(SET_UTF8 + line + "\n" + PLURAL),
            stream(WORDS)));
    final int separator = line.indexOf(' ');
    final String directive = separator < 0 ? line : line.substring(0, separator);
    Assertions.assertTrue(error.getMessage().contains(directive));
    Assertions.assertTrue(error.getMessage().contains(AFFIX_STREAM));
    Assertions.assertTrue(error.getMessage().contains("line 2"));
  }

  /**
   * Counts logical lines with each supported line separator.
   *
   * @param separator A supported line separator.
   */
  @ParameterizedTest
  @ValueSource(strings = {"\n", "\r\n", "\r"})
  void testUnsupportedDirectiveLineNumber(String separator) {
    final String affix = "# comment" + separator + separator + "  " + UNKNOWN + " K";
    final IOException error = Assertions.assertThrows(IOException.class,
        () -> HunspellDictionary.load(stream(affix), stream(WORDS)));
    Assertions.assertTrue(error.getMessage().contains(UNKNOWN));
    Assertions.assertTrue(error.getMessage().contains("line 3"));
  }

  /**
   * Reads the affix file and the word list with each supported line separator, so a
   * file with carriage return line endings loads like one with line feeds.
   *
   * @param separator A supported line separator.
   * @throws IOException Thrown if the fixture fails to load.
   */
  @ParameterizedTest
  @ValueSource(strings = {"\n", "\r\n", "\r"})
  void testLineSeparatorsInAffixAndWordList(String separator) throws IOException {
    final String affix = String.join(separator, "SFX A Y 1", "SFX A 0 s .", "");
    final String words = String.join(separator, "1", "dog/A", "");
    final HunspellDictionary dictionary = HunspellDictionary.load(stream(affix), stream(words));
    Assertions.assertEquals("dog", new HunspellStemmer(dictionary).stem("dogs").toString());
  }

  /**
   * Reports the line of a malformed supported directive with each line separator.
   *
   * @param separator A supported line separator.
   */
  @ParameterizedTest
  @ValueSource(strings = {"\n", "\r\n", "\r"})
  void testMalformedDirectiveLineNumber(String separator) {
    final String affix = String.join(separator, "# comment", "", "IGNORE", "");
    final IOException error = Assertions.assertThrows(IOException.class,
        () -> HunspellDictionary.load(stream(affix), stream(WORDS)));
    Assertions.assertTrue(error.getMessage().contains("IGNORE"), error.getMessage());
    Assertions.assertTrue(error.getMessage().contains("line 3"), error.getMessage());
  }

  /** Rejects an unknown directive immediately after a UTF-8 byte-order mark. */
  @Test
  void testByteOrderMarkDoesNotHideUnsupportedDirective() {
    final IOException error = Assertions.assertThrows(IOException.class,
        () -> HunspellDictionary.load(stream("\uFEFF" + UNKNOWN + " 1\n"), stream(WORDS)));
    Assertions.assertTrue(error.getMessage().contains(UNKNOWN));
    Assertions.assertTrue(error.getMessage().contains("line 1"));
  }

  /**
   * Identifies the source file when path-based loading rejects a directive.
   *
   * @throws IOException Thrown if writing a fixture fails.
   */
  @Test
  void testPathErrorIdentifiesAffixFile() throws IOException {
    final Path affix = directory.resolve("sample.aff");
    final Path words = directory.resolve("sample.dic");
    Files.writeString(affix, SET_UTF8 + UNKNOWN + " K\n");
    Files.writeString(words, WORDS);
    final IOException error = Assertions.assertThrows(IOException.class,
        () -> HunspellDictionary.load(affix, words));
    Assertions.assertTrue(error.getMessage().contains(affix.toString()));
    Assertions.assertTrue(error.getMessage().contains("line 2"));
  }

  /**
   * Loads settings outside the stemmer's operations without a diagnostic.
   *
   * @param setting A metadata or suggestion setting.
   * @throws IOException Thrown if the fixture fails to load.
   */
  @ParameterizedTest
  @ValueSource(strings = {
      "NAME Example", "HOME https://example.org", "VERSION 1", "KEY abc|def",
      "TRY abc", "REP 1\nREP ph f", "MAP 1\nMAP aá", "PHONE 1\nPHONE ph f",
      "NOSUGGEST N", "MAXCPDSUGS 0", "MAXNGRAMSUGS 0", "MAXDIFF 5",
      "ONLYMAXDIFF", "NOSPLITSUGS", "SUGSWITHDOTS", "WARN W",
      "SUBSTANDARD S", "WORDCHARS -", "NONGRAMSUGGEST N", "CHECKNUM"
  })
  void testSettingsOutsideStemmingDoNotPreventStrictLoading(String setting)
      throws IOException {
    final HunspellDictionary dictionary = HunspellDictionary.load(
        stream(setting + "\n" + PLURAL), stream(WORDS));
    Assertions.assertEquals("dog", new HunspellStemmer(dictionary).stem("dogs").toString());
    Assertions.assertTrue(dictionary.getUnsupportedDirectives().isEmpty());
  }

  /**
   * Every directive Hunspell's affix and dictionary parsers read, except the affix
   * rules themselves, each with a valid value.
   *
   * @return Affix content declaring one directive.
   */
  private static Stream<String> hunspellDirectives() {
    final Stream<String> flags = Stream.of("CIRCUMFIX", "COMPOUNDBEGIN", "COMPOUNDEND",
        "COMPOUNDFLAG", "COMPOUNDFORBIDFLAG", "COMPOUNDMIDDLE", "COMPOUNDPERMITFLAG",
        "COMPOUNDROOT", "FORBIDDENWORD", "FORCEUCASE", "KEEPCASE", "LEMMA_PRESENT", "NEEDAFFIX",
        "NONGRAMSUGGEST", "NOSUGGEST", "ONLYINCOMPOUND", "PSEUDOROOT", "SUBSTANDARD", "SYLLABLENUM",
        "WARN").map(directive -> directive + FLAG_VALUE);
    final Stream<String> switches = Stream.of("CHECKCOMPOUNDCASE", "CHECKCOMPOUNDDUP",
        "CHECKCOMPOUNDREP", "CHECKCOMPOUNDTRIPLE", "CHECKNUM", "CHECKSHARPS", "COMPLEXPREFIXES",
        "COMPOUNDMORESUFFIXES", "FORBIDWARN", "FULLSTRIP", "NOSPLITSUGS", "ONLYMAXDIFF",
        "SIMPLIFIEDTRIPLE", "SUGSWITHDOTS");
    final Stream<String> values = Stream.of("AF 1\nAF A", "AM 1\nAM po:noun", "BREAK 1\nBREAK -",
        "CHECKCOMPOUNDPATTERN 1\nCHECKCOMPOUNDPATTERN o b", "COMPOUNDMIN 2",
        "COMPOUNDRULE 1\nCOMPOUNDRULE XY", "COMPOUNDSYLLABLE 6 aeiou", "COMPOUNDWORDMAX 2",
        "FLAG UTF-8", "ICONV 1\nICONV a b", "IGNORE q", "KEY qwerty|asdf", "LANG en_US",
        "MAP 1\nMAP aá", "MAXCPDSUGS 1", "MAXDIFF 5", "MAXNGRAMSUGS 1", "OCONV 1\nOCONV a b",
        "PHONE 1\nPHONE ph f", "REP 1\nREP ph f", "SET UTF-8", "TRY abc", "VERSION 1",
        "WORDCHARS -");
    return Stream.of(flags, switches, values).flatMap(directives -> directives);
  }

  /**
   * Loads under strict loading every directive name Hunspell's parsers read, as a
   * stemming setting or as one the stemmer ignores, and reports none of them as
   * unsupported under partial loading.
   *
   * @param directive Affix content declaring the directive with a valid value.
   * @throws IOException Thrown if the fixture fails to load.
   */
  @ParameterizedTest
  @MethodSource("hunspellDirectives")
  void testHunspellDirectivesAreNotUnsupported(String directive) throws IOException {
    final String affix = directive + "\n" + PLURAL;
    Assertions.assertNotNull(HunspellTestDictionaries.load(affix, WORDS));
    Assertions.assertEquals(List.of(), HunspellTestDictionaries.load(affix, WORDS,
        LoadMode.ALLOW_PARTIAL).getUnsupportedDirectives());
  }

  /**
   * Keeps a number sign that is a directive value, which the Hunspell format allows
   * for flags, separators, and affix material.
   *
   * @param affix Affix content in which {@code #} is a value.
   * @param words The word list.
   * @param input The stemmed word.
   * @param expected The stem.
   * @throws IOException Thrown if the fixture fails to load.
   */
  @ParameterizedTest
  @MethodSource("numberSignValues")
  void testNumberSignValuesAreKept(String affix, String words, String input, String expected)
      throws IOException {
    final HunspellDictionary dictionary = HunspellDictionary.load(stream(affix), stream(words));
    Assertions.assertEquals(expected, new HunspellStemmer(dictionary).stem(input).toString());
  }

  /**
   * Directive values consisting of a number sign.
   *
   * @return Affix content, word list, input, and expected stem.
   */
  private static Stream<Arguments> numberSignValues() {
    return Stream.of(
        Arguments.of("BREAK 1\nBREAK #\n" + PLURAL, WORDS, "dogs#dogs", "dog"),
        Arguments.of("NEEDAFFIX #\n" + PLURAL, "1\ndog/#A\n", "dogs", "dog"),
        Arguments.of("FLAG long\nAF 1\nAF #A\nSFX #A Y 1\nSFX #A 0 s .\n", "1\ndog/1\n", "dogs", "dog"),
        Arguments.of("SFX A Y 1\nSFX A 0 # .\n", WORDS, "dog#", "dog"),
        Arguments.of("SFX A Y 1\nSFX A # s [#]\n", "1\ndog#/A\n", "dogs", "dog#"));
  }

  /**
   * Ignores trailing comments after the fields a directive consumes.
   *
   * @param affix Affix content with a trailing comment.
   * @throws IOException Thrown if the fixture fails to load.
   */
  @ParameterizedTest
  @ValueSource(strings = {
      "COMPOUNDMIN 3 # comment\nSFX A Y 1 # comment\nSFX A 0 s . # comment",
      "NEEDAFFIX X # comment\nSFX A Y 1\nSFX A 0 s .",
      "SET UTF-8 # comment\nFLAG UTF-8 # comment\nSFX A Y 1\nSFX A 0 s .",
      "AF 1\nAF A # comment\nSFX A Y 1\nSFX A 0 s ."
  })
  void testTrailingCommentsAreIgnored(String affix) throws IOException {
    final String words = affix.startsWith("AF") ? "1\ndog/1\n" : WORDS;
    final HunspellDictionary dictionary = HunspellDictionary.load(stream(affix), stream(words));
    Assertions.assertEquals("dog", new HunspellStemmer(dictionary).stem("dogs").toString());
  }

  /**
   * Reports the first location for each skipped directive in source order.
   *
   * @param separator A supported line separator.
   * @throws IOException Thrown if the fixture fails to load.
   */
  @ParameterizedTest
  @ValueSource(strings = {"\n", "\r\n", "\r"})
  void testPartialLoadingReportsFirstOccurrences(String separator) throws IOException {
    final String affix = String.join(separator, UNKNOWN + " 1", UNKNOWN + " a b",
        UNRECOGNIZED + " 1", UNRECOGNIZED + " x", PLURAL);
    final HunspellDictionary dictionary = HunspellDictionary.load(
        stream(affix), stream(WORDS), LoadMode.ALLOW_PARTIAL);
    final List<UnsupportedDirective> diagnostics = dictionary.getUnsupportedDirectives();
    Assertions.assertEquals(List.of(
        new UnsupportedDirective(UNKNOWN, AFFIX_STREAM, 1),
        new UnsupportedDirective(UNRECOGNIZED, AFFIX_STREAM, 3)), diagnostics);
    Assertions.assertThrows(UnsupportedOperationException.class, diagnostics::clear);
    Assertions.assertEquals("dog", new HunspellStemmer(dictionary).stem("dogs").toString());
  }

  /**
   * Includes the affix path in partial-loading diagnostics.
   *
   * @throws IOException Thrown if writing or loading the fixture fails.
   */
  @Test
  void testPartialLoadingReportsFilePath() throws IOException {
    final Path affix = directory.resolve("partial.aff");
    final Path words = directory.resolve("partial.dic");
    Files.writeString(affix, SET_UTF8 + UNKNOWN + " K\n" + PLURAL);
    Files.writeString(words, WORDS);
    final HunspellDictionary dictionary = HunspellDictionary.load(
        affix, words, LoadMode.ALLOW_PARTIAL);
    Assertions.assertEquals(List.of(new UnsupportedDirective(
        UNKNOWN, affix.toString(), 2)), dictionary.getUnsupportedDirectives());
    Assertions.assertEquals("dog", new HunspellStemmer(dictionary).stem("dogs").toString());
  }

  /**
   * Rejects supported malformed content under either loading policy.
   *
   * @param malformed Malformed affix content.
   */
  @ParameterizedTest
  @ValueSource(strings = {"AF -1\n", "FLAG num\nSFX 65536 Y 0\n",
      "COMPOUNDMIN -1\n", "SFX A Y 2\nSFX A 0 s .\n", "FLAG short\n",
      "ICONV 1\n", "ICONV -1\n", "ICONV 1\nICONV a b\nICONV c d\n",
      "OCONV 1\nOCONV _ x\n", "AM 1\n", "AM -1\n",
      "COMPOUNDRULE 1\n", "COMPOUNDRULE 1\nCOMPOUNDRULE *A\n",
      "COMPOUNDRULE 1\nCOMPOUNDRULE (\n", "CHECKCOMPOUNDPATTERN 1\n",
      "CHECKCOMPOUNDPATTERN -1\n", "BREAK 1\n", "BREAK -1\n", "BREAK 1\nBREAK ^\n",
      "IGNORE\n", "LANG\n", "COMPOUNDSYLLABLE -1 ae\n", "KEEPCASE\n",
      "SYLLABLENUM\n", "LEMMA_PRESENT\n", "LEMMA_PRESENT AB\n",
      "AM 1 extra\nAM po:noun\n", "CHECKCOMPOUNDPATTERN 0 extra\n"})
  void testPartialLoadingDoesNotIgnoreMalformedRules(String malformed) {
    Assertions.assertThrows(IOException.class, () -> HunspellDictionary.load(
        stream(malformed), stream("1\ndog\n"), LoadMode.STRICT));
    Assertions.assertThrows(IOException.class, () -> HunspellDictionary.load(
        stream(UNKNOWN + " K\n" + malformed), stream("1\ndog\n"), LoadMode.ALLOW_PARTIAL));
  }

  /**
   * Invalid AM references under each loading policy.
   *
   * @return An invalid reference and a loading policy.
   */
  private static Stream<Arguments> invalidMorphologyAliases() {
    return Stream.of("-1", "0", "2", "invalid", "1 1").flatMap(reference ->
        Stream.of(LoadMode.values()).map(mode -> Arguments.of(reference, mode)));
  }

  /**
   * Rejects invalid AM references in entries and affix fields.
   *
   * @param reference The invalid reference.
   * @param mode The loading policy.
   */
  @ParameterizedTest
  @MethodSource("invalidMorphologyAliases")
  void testInvalidMorphologyAliases(String reference, LoadMode mode) {
    final String aliases = "AM 1\nAM po:noun\n";
    Assertions.assertThrows(IOException.class, () -> HunspellDictionary.load(
        stream(aliases), stream("1\ndog\t" + reference + "\n"), mode));
    Assertions.assertThrows(IOException.class, () -> HunspellDictionary.load(
        stream(aliases + "SFX A Y 1\nSFX A 0 s . " + reference + "\n"), stream(WORDS), mode));
  }

  /**
   * Rejects malformed text in partial mode.
   *
   * @param file The file containing malformed UTF-8.
   */
  @ParameterizedTest
  @ValueSource(strings = {"affix", "dictionary"})
  void testPartialLoadingRejectsMalformedText(String file) {
    final byte[] malformed = {(byte) 0xc3};
    final ByteArrayInputStream affix = "affix".equals(file)
        ? new ByteArrayInputStream(concat(SET_UTF8 + UNKNOWN + " K\nSFX A Y 1\nSFX A 0 ",
            malformed)) : stream(UNKNOWN + " K\n");
    final ByteArrayInputStream words = "dictionary".equals(file)
        ? new ByteArrayInputStream(concat("1\n", malformed)) : stream(WORDS);
    final IOException error = Assertions.assertThrows(IOException.class,
        () -> HunspellDictionary.load(affix, words, LoadMode.ALLOW_PARTIAL));
    Assertions.assertEquals(file + " stream is not valid UTF-8", error.getMessage());
  }

  /**
   * Accepts legacy bytes in recognized metadata without decoding them as rules.
   *
   * @throws IOException Thrown if the fixture fails to load.
   */
  @Test
  void testLegacyMetadataBytesAreIgnored() throws IOException {
    final byte[] affix = concat(SET_UTF8 + "NAME ", new byte[] {(byte) 0xc3});
    final HunspellDictionary dictionary = HunspellDictionary.load(
        new ByteArrayInputStream(affix), stream(WORDS));
    Assertions.assertNotNull(dictionary.lookup("dog"));
    Assertions.assertTrue(dictionary.getUnsupportedDirectives().isEmpty());
  }

  /**
   * Preserves raw flag bytes in compound-boundary conditions without changing word text.
   *
   * @param matchingFlag Whether the left entry has the boundary flag.
   * @throws IOException Thrown if the fixture fails to load.
   */
  @ParameterizedTest
  @ValueSource(booleans = {true, false})
  void testCompoundPatternByteFlag(boolean matchingFlag) throws IOException {
    final byte[] flag = {(byte) 0xc3};
    final byte[] affix = concat(SET_UTF8 + COMPOUND
        + "CHECKCOMPOUNDPATTERN 1\nCHECKCOMPOUNDPATTERN er/", flag, " b\n");
    final byte[] words = concat("2\nriver/C", matchingFlag ? flag : new byte[0], "\nboat/C\n");
    final HunspellStemmer stemmer = new HunspellStemmer(HunspellDictionary.load(
        new ByteArrayInputStream(affix), new ByteArrayInputStream(words)));
    Assertions.assertEquals(matchingFlag ? List.of("riverboat") : List.of("river", "boat"),
        stemmer.stemAll("riverboat"));
  }

  /**
   * Loads supported affix rules after a UTF-8 byte-order mark.
   *
   * @throws IOException Thrown if the fixture fails to load.
   */
  @Test
  void testByteOrderMarkDoesNotHideSupportedDirective() throws IOException {
    final HunspellDictionary dictionary = HunspellDictionary.load(
        stream("\uFEFF" + PLURAL), stream(WORDS));
    Assertions.assertEquals("dog", new HunspellStemmer(dictionary).stem("dogs").toString());
  }

  /**
   * Reads the flag declaration on the first line after a UTF-8 byte-order mark, so
   * the declared flag mode, not the raw-byte default, applies to the rules.
   *
   * @param affix The affix content following the byte-order mark.
   * @param words The word list.
   * @throws IOException Thrown if the fixture fails to load.
   */
  @ParameterizedTest
  @MethodSource("flagDeclarationsAfterByteOrderMark")
  void testByteOrderMarkDoesNotHideFlagDeclaration(byte[] affix, byte[] words)
      throws IOException {
    final byte[] marked = new byte[affix.length + 3];
    marked[0] = (byte) 0xef;
    marked[1] = (byte) 0xbb;
    marked[2] = (byte) 0xbf;
    System.arraycopy(affix, 0, marked, 3, affix.length);
    final HunspellDictionary dictionary = HunspellDictionary.load(
        new ByteArrayInputStream(marked), new ByteArrayInputStream(words));
    Assertions.assertEquals("card", new HunspellStemmer(dictionary).stem("cards").toString());
  }

  /** {@return affix content with a flag declaration or a raw byte flag on the first line} */
  private static Stream<Arguments> flagDeclarationsAfterByteOrderMark() {
    return Stream.of(
        Arguments.of("FLAG UTF-8\nSFX \u00e4 Y 1\nSFX \u00e4 0 s .\n".getBytes(StandardCharsets.UTF_8),
            "1\ncard/\u00e4\n".getBytes(StandardCharsets.UTF_8)),
        Arguments.of("FLAG long\nSFX Qz Y 1\nSFX Qz 0 s .\n".getBytes(StandardCharsets.UTF_8),
            "1\ncard/Qz\n".getBytes(StandardCharsets.UTF_8)),
        Arguments.of("FLAG num\nSFX 312 Y 1\nSFX 312 0 s .\n".getBytes(StandardCharsets.UTF_8),
            "1\ncard/312\n".getBytes(StandardCharsets.UTF_8)),
        Arguments.of(new byte[] {'S', 'F', 'X', ' ', (byte) 0xe4, ' ', 'Y', ' ', '1', '\n',
            'S', 'F', 'X', ' ', (byte) 0xe4, ' ', '0', ' ', 's', ' ', '.', '\n'},
            new byte[] {'1', '\n', 'c', 'a', 'r', 'd', '/', (byte) 0xe4, '\n'}));
  }

  /**
   * Rejects null arguments before reading either stream or opening a file.
   *
   * @param mode The loading policy.
   */
  @ParameterizedTest
  @EnumSource(LoadMode.class)
  void testNullArgumentsAreRejected(LoadMode mode) {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> HunspellDictionary.load((Path) null, directory, mode));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> HunspellDictionary.load(directory, null, mode));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> HunspellDictionary.load(null, stream(WORDS), mode));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> HunspellDictionary.load(stream(PLURAL), null, mode));
  }

  /** Rejects a null loading policy through both public entry points. */
  @Test
  void testNullModeIsRejected() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> HunspellDictionary.load(directory, directory, null));
    final ByteArrayInputStream affix = stream(PLURAL);
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> HunspellDictionary.load(affix, stream(WORDS), null));
    Assertions.assertEquals(PLURAL.getBytes(StandardCharsets.UTF_8).length, affix.available());
  }

  /**
   * Preserves ownership of input streams on success and failure.
   *
   * @param mode The loading policy.
   * @throws IOException Thrown if valid content fails to load.
   */
  @ParameterizedTest
  @EnumSource(LoadMode.class)
  void testStreamsAreNotClosed(LoadMode mode) throws IOException {
    final TrackedStream affix = new TrackedStream(PLURAL);
    final TrackedStream words = new TrackedStream(WORDS);
    HunspellDictionary.load(affix, words, mode);
    Assertions.assertFalse(affix.closed);
    Assertions.assertFalse(words.closed);
    final TrackedStream invalid = new TrackedStream("AF -1\n");
    Assertions.assertThrows(IOException.class,
        () -> HunspellDictionary.load(invalid, words, mode));
    Assertions.assertFalse(invalid.closed);
    Assertions.assertFalse(words.closed);
  }

  /** Verifies validation of a diagnostic's public fields. */
  @Test
  void testDiagnosticArgumentsAreValidated() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new UnsupportedDirective(null, "source", 1));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new UnsupportedDirective(" ", "source", 1));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new UnsupportedDirective(UNKNOWN, null, 1));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new UnsupportedDirective(UNKNOWN, " ", 1));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new UnsupportedDirective(UNKNOWN, "source", 0));
  }

  /** Detects close calls while allowing further input operations. */
  private static final class TrackedStream extends ByteArrayInputStream {
    private boolean closed;

    /**
     * Creates an encoded fixture stream.
     *
     * @param content The fixture text.
     */
    private TrackedStream(String content) {
      super(content.getBytes(StandardCharsets.UTF_8));
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
      closed = true;
    }
  }

  /**
   * Appends raw bytes to a UTF-8 fixture prefix.
   *
   * @param prefix The fixture prefix.
   * @param bytes The raw suffix.
   * @return The combined content.
   */
  private static byte[] concat(String prefix, byte[] bytes) {
    return concat(prefix, bytes, "");
  }

  /**
   * Places raw bytes between a UTF-8 fixture prefix and suffix.
   *
   * @param prefix The fixture prefix.
   * @param bytes The raw bytes.
   * @param suffix The fixture suffix.
   * @return The combined content.
   */
  private static byte[] concat(String prefix, byte[] bytes, String suffix) {
    final byte[] start = prefix.getBytes(StandardCharsets.UTF_8);
    final byte[] end = suffix.getBytes(StandardCharsets.UTF_8);
    final byte[] result = Arrays.copyOf(start, start.length + bytes.length + end.length);
    System.arraycopy(bytes, 0, result, start.length, bytes.length);
    System.arraycopy(end, 0, result, start.length + bytes.length, end.length);
    return result;
  }
}
