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

package opennlp.tools.stemmer.hunspell;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.Fixture;
import opennlp.tools.util.StringUtil;

import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.ACCEPTED;
import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.COMPOUND;
import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.NO_STEMS;
import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.PLURAL;
import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.REJECTED;
import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.load;

/**
 * Compound and morphology fixtures written for this test, with the acceptance, stems
 * and analyses recorded from Hunspell as described in
 * {@code dev/README-hunspell-dictionaries.md}.
 */
class HunspellCompletionTest {

  private static final String HUNGARIAN = COMPOUND + "LANG hu\nCOMPOUNDWORDMAX 2\n";
  private static final String THREE_WORDS = "3\nray/C\nme/C\nfa/C";
  private static final String NEEDS_AFFIX_ZERO_RULES = "NEEDAFFIX N\n"
      + "SFX P Y 1\nSFX P 0 0 . is:bare\nSFX Q Y 1\nSFX Q 0 0 . is:plain\n"
      + "SFX R Y 2\nSFX R 0 0/NPQ . dp:none\nSFX R 0 ix/NPQ . dp:ix\n";
  private static final String PLURAL_FIELD = "SFX A Y 1\nSFX A 0 s . is:plural\n";
  private static final String CARD_HOMONYMS = "2\ncard/A po:noun\ncard/A po:verb\n";
  private static final List<String> CARD_ANALYSES =
      List.of("st:card po:noun is:plural", "st:card po:verb is:plural");
  private static final String SHARP_S = "ß";
  private static final String DOUBLE_S = "ss";
  private static final int SHARP_S_POSITIONS = 6;

  /** {@return the compound and morphology fixtures} */
  private static Stream<Fixture> fixtures() {
    return Stream.of(
        new Fixture("legacy-lemma", "LEMMA_PRESENT L\n" + PLURAL,
            "1\nfeet/AL st:foot\n", "feets", List.of("foot"),
            ACCEPTED, List.of("foot")),
        new Fixture("syllable-c-reject", HUNGARIAN + "COMPOUNDSYLLABLE 5 aeiouy\n"
            + "SYLLABLENUM klmc\nSFX c Y 1\nSFX c 0 s .\n",
            THREE_WORDS + "c\n", "raymefas", List.of("raymefas"),
            REJECTED, NO_STEMS),
        new Fixture("syllable-c-accept", HUNGARIAN + "COMPOUNDSYLLABLE 6 aeiouy\n"
            + "SYLLABLENUM klmc\nSFX c Y 1\nSFX c 0 s .\n",
            THREE_WORDS + "c\n", "raymefas", List.of("ray", "me", "fa"),
            ACCEPTED, List.of("raymefa")),
        new Fixture("syllable-j-reject", HUNGARIAN + "COMPOUNDSYLLABLE 4 aeiouy\n"
            + "SYLLABLENUM klmc\nSFX J Y 1\nSFX J 0 s .\n",
            THREE_WORDS + "J\n", "raymefas", List.of("raymefas"),
            REJECTED, NO_STEMS),
        new Fixture("syllable-i-with-j", HUNGARIAN + "COMPOUNDSYLLABLE 4 aeiouy\n"
            + "SYLLABLENUM klmc\nSFX I Y 1\nSFX I 0 s .\n",
            THREE_WORDS + "IJ\n", "raymefas", List.of("raymefas"),
            REJECTED, NO_STEMS),
        new Fixture("syllable-i-without-j", HUNGARIAN + "COMPOUNDSYLLABLE 4 aeiouy\n"
            + "SYLLABLENUM klmc\nSFX I Y 1\nSFX I 0 s .\n",
            THREE_WORDS + "I\n", "raymefas", List.of("ray", "me", "fa"),
            ACCEPTED, List.of("raymefa")),
        new Fixture("syllable-terminal-inflection", HUNGARIAN + "COMPOUNDSYLLABLE 4 aeiouy\n"
            + "SFX A Y 1\nSFX A 0 a .\n", THREE_WORDS + "A\n",
            "raymefaa", List.of("ray", "me", "fa"),
            ACCEPTED, List.of("raymefa")),
        new Fixture("syllable-unaffixed-i", HUNGARIAN + "COMPOUNDSYLLABLE 3 aeiouy\n",
            THREE_WORDS + "I\n", "raymefa", List.of("ray", "me", "fa"),
            ACCEPTED, List.of("rayme")),
        new Fixture("syllable-prefix-word-count", HUNGARIAN + "COMPOUNDSYLLABLE 2 aeiou\n"
            + "PFX A Y 1\nPFX A 0 reco .\n", "2\nme/CA\nfa/C\n",
            "recomefa", List.of("recomefa"),
            REJECTED, NO_STEMS),
        new Fixture("multiple-triple-junctions", COMPOUND + "CHECKCOMPOUNDTRIPLE\nSIMPLIFIEDTRIPLE\n",
            "2\nmill/C\nloom/C\n", "milloommilloom", List.of("mill", "loom"),
            ACCEPTED, NO_STEMS),
        new Fixture("multiple-pattern-junctions", COMPOUND + "CHECKCOMPOUNDPATTERN 1\n"
            + "CHECKCOMPOUNDPATTERN er b X\n", "2\nriver/C\nboat/C\n",
            "rivXoatrivXoat", List.of("rivXoatrivXoat"),
            REJECTED, NO_STEMS),
        new Fixture("compound-cross-product", COMPOUND + "COMPOUNDPERMITFLAG P\n"
            + "PFX A Y 1\nPFX A 0 re .\nSFX B Y 1\nSFX B 0 s/P .\n",
            "2\nriver/CAB\nboat/C\n", "reriversboat", List.of("river", "boat"),
            ACCEPTED, List.of("rerivers")),
        new Fixture("compound-cross-product-no-permit", COMPOUND + "COMPOUNDPERMITFLAG P\n"
            + "PFX A Y 1\nPFX A 0 re .\nSFX B Y 1\nSFX B 0 s .\n",
            "2\nriver/CAB\nboat/C\n", "reriversboat", List.of("reriversboat"),
            REJECTED, NO_STEMS),
        new Fixture("optional-affix-condition", "SFX A Y 1\nSFX A 0 s\n",
            "1\ncard/A\n", "cards", List.of("card"),
            ACCEPTED, List.of("card")),
        new Fixture("optional-condition-with-morphology", "SFX A Y 1\nSFX A 0 s is:plural\n",
            "1\ncard/A\n", "cards", List.of("cards"),
            REJECTED, NO_STEMS),
        new Fixture("obsolete-compound-first", "COMPOUNDFIRST V\n",
            "2\nriver/V\nboat/V\n", "riverboat", List.of("riverboat"),
            REJECTED, NO_STEMS),
        new Fixture("obsolete-compound-last", "COMPOUNDLAST V\n",
            "2\nriver/V\nboat/V\n", "riverboat", List.of("riverboat"),
            REJECTED, NO_STEMS),
        new Fixture("obsolete-only-root", "ONLYROOT V\n" + PLURAL,
            "1\ncard/VA\n", "cards", List.of("card"),
            ACCEPTED, List.of("card")),
        new Fixture("obsolete-hungarian-linking-vowel", "LANG hu\nHU_KOTOHANGZO V\n"
            + PLURAL, "1\ncard/VA\n", "cards", List.of("card"),
            ACCEPTED, List.of("card")),
        new Fixture("generation-option", "GENERATE 1\n" + PLURAL,
            "1\ncard/A\n", "cards", List.of("card"),
            ACCEPTED, List.of("card")),
        new Fixture("compound-word-with-space", COMPOUND,
            "3\nriver/C\nboat/C\nriver boat\n", "riverboat", List.of("riverboat"),
            REJECTED, List.of("river")),
        new Fixture("compound-affixed-word-with-space", COMPOUND + PLURAL,
            "3\nriver/C\nboat/CA\nriver boat/A\n", "riverboats", List.of("riverboats"),
            REJECTED, List.of("riverboat")),
        new Fixture("compound-affixed-duplicate", COMPOUND + "CHECKCOMPOUNDDUP\n"
            + "COMPOUNDPERMITFLAG P\nSFX A Y 1\nSFX A 0 s/P .\n",
            "1\nriver/CA\n", "riversriver", List.of("riversriver"),
            REJECTED, NO_STEMS),
        // Hunspell's spell checker rejects these forms while its analyzer stems them
        new Fixture("turkish-capitalized-name", "LANG tr_TR\n", "1\nİpek\n",
            "İPEK", List.of("İpek"),
            REJECTED, List.of("İpek")),
        new Fixture("azerbaijani-capitalized-name", "LANG az_AZ\n", "1\nİpek\n",
            "İPEK", List.of("İpek"),
            REJECTED, List.of("İpek")),
        new Fixture("compound-pattern-suffix-flag", COMPOUND + "COMPOUNDPERMITFLAG P\n"
            + "CHECKCOMPOUNDPATTERN 1\nCHECKCOMPOUNDPATTERN s/X b\n"
            + "SFX A Y 1\nSFX A 0 s/PX .\n", "2\nriver/CA\nboat/C\n",
            "riversboat", List.of("riversboat"),
            REJECTED, List.of("rivers")),
        new Fixture("compound-pattern-prefix-flag", COMPOUND + "COMPOUNDPERMITFLAG P\n"
            + "CHECKCOMPOUNDPATTERN 1\nCHECKCOMPOUNDPATTERN r r/X\n"
            + "PFX A Y 1\nPFX A 0 re/PX .\n", "2\nriver/C\nboat/CA\n",
            "riverreboat", List.of("riverreboat"),
            REJECTED, NO_STEMS),
        new Fixture("compound-cross-double-suffix", COMPOUND + "COMPOUNDPERMITFLAG P\n"
            + "COMPOUNDMORESUFFIXES\nPFX R Y 1\nPFX R 0 re .\n"
            + "SFX A Y 1\nSFX A 0 er/BP .\nSFX B Y 1\nSFX B 0 s/P .\n",
            "2\nboat/CAR\nriver/C\n", "reboatersriver", List.of("reboatersriver"),
            REJECTED, NO_STEMS),
        new Fixture("compound-cross-double-suffix-final", COMPOUND + "COMPOUNDPERMITFLAG P\n"
            + "COMPOUNDMORESUFFIXES\nPFX R Y 1\nPFX R 0 re/P .\n"
            + "SFX A Y 1\nSFX A 0 er/B .\nSFX B Y 1\nSFX B 0 s .\n",
            "2\nboat/CAR\nriver/C\n", "riverreboaters", List.of("river", "boat"),
            ACCEPTED, List.of("riverboat")),
        new Fixture("compound-complex-prefix", COMPOUND + "COMPLEXPREFIXES\n"
            + "COMPOUNDMORESUFFIXES\nPFX A Y 1\nPFX A 0 re/B .\n"
            + "PFX B Y 1\nPFX B 0 un .\n", "2\nriver/CA\nboat/C\n",
            "unreriverboat", List.of("river", "boat"),
            ACCEPTED, NO_STEMS),
        new Fixture("compound-complex-prefix-suffix", COMPOUND + "COMPLEXPREFIXES\n"
            + "COMPOUNDMORESUFFIXES\nCOMPOUNDPERMITFLAG P\n"
            + "PFX A Y 1\nPFX A 0 re/B .\nPFX B Y 1\nPFX B 0 un .\n"
            + "SFX S Y 1\nSFX S 0 s/P .\n", "2\nriver/CAS\nboat/C\n",
            "unreriversboat", List.of("river", "boat"),
            ACCEPTED, NO_STEMS),
        // deviations listed in the manual: Unicode case mapping, the suggester-based
        // rejection of multi-part compounds, and numeric tokens
        new Fixture("dotted-capital-i", "", "1\ninvite\n", "İnvite", List.of("invite"),
            REJECTED, List.of("invite")),
        new Fixture("dotted-capital-i-all-caps", "", "1\nİnci\n", "İNCİ", List.of("İnci"),
            ACCEPTED, NO_STEMS),
        // KEEPCASE with CHECKSHARPS admits the SS spelling of an all-uppercase form only
        new Fixture("keepcase-sharp-s-double-s", "CHECKSHARPS\nKEEPCASE k\n", "1\nfleiß/k\n",
            "FLEISS", List.of("fleiß"),
            ACCEPTED, NO_STEMS),
        new Fixture("keepcase-capital-sharp-s", "CHECKSHARPS\nKEEPCASE k\n", "1\nfleiß/k\n",
            "FLEIẞ", List.of("FLEIẞ"),
            REJECTED, List.of("fleiß")),
        new Fixture("keepcase-sharp-s-capitalized", "CHECKSHARPS\nKEEPCASE k\n", "1\nfleiß/k\n",
            "Fleiß", List.of("fleiß"),
            ACCEPTED, List.of("fleiß")),
        new Fixture("multi-part-compound-near-listed-word", "TRY abcdefghijklmnopqrstuvwxyz\n"
            + "COMPOUNDFLAG x\n", "5\nsun/x\nset/x\nlamp/x\nbunset\nbunsetlamp\n",
            "sunsetlamp", List.of("sun", "set", "lamp"),
            REJECTED, List.of("sunset")),
        new Fixture("numeric-token", "", "1\nfoo\n", "1.5", List.of("1.5"),
            ACCEPTED, NO_STEMS));
  }

  /** {@return the fixtures whose stems are compared with Hunspell's} */
  private static Stream<Fixture> comparableStemFixtures() {
    return fixtures().filter(Fixture::hasComparableStems);
  }

  /**
   * The fixtures whose recognition deliberately differs from Hunspell's spell checker,
   * each with the manual's reason.
   */
  private static final Map<String, String> RECOGNITION_DEVIATIONS = Map.of(
      "turkish-capitalized-name", "Hunspell's spell checker rejects what its analyzer stems",
      "azerbaijani-capitalized-name", "Hunspell's spell checker rejects what its analyzer stems",
      "dotted-capital-i", "a dotted capital I is lowercased outside the Turkic languages",
      "multi-part-compound-near-listed-word", "Hunspell rejects it through its suggester",
      "numeric-token", "Hunspell accepts numbers before any lookup");

  /**
   * The fixtures with comparable stems whose stems differ from the stems Hunspell
   * returned, each with the reason.
   */
  private static final Map<String, String> STEM_DEVIATIONS = Map.of(
      "dotted-capital-i-all-caps", "Hunspell's analyzer returns no stem",
      "keepcase-sharp-s-double-s", "Hunspell's analyzer does not expand SS to a sharp s",
      "numeric-token", "Hunspell's analyzer returns no stem for a number");

  /**
   * Checks the OpenNLP stems of one fixture.
   *
   * @param fixture The fixture.
   * @throws IOException Thrown if the fixture fails to load.
   */
  @ParameterizedTest(name = "{0}")
  @MethodSource("fixtures")
  void testStemming(Fixture fixture) throws IOException {
    Assertions.assertEquals(fixture.expected(), fixture.stems());
  }

  /**
   * Tests recognition against the acceptance recorded from Hunspell's spell checker.
   * A fixture listed in {@link #RECOGNITION_DEVIATIONS} must differ, so a stale entry
   * fails too.
   *
   * @param fixture The fixture.
   * @throws IOException Thrown if the fixture fails to load.
   */
  @ParameterizedTest(name = "recognition {0}")
  @MethodSource("fixtures")
  void testRecognitionAgainstHunspell(Fixture fixture) throws IOException {
    HunspellTestDictionaries.assertRecognition(fixture, RECOGNITION_DEVIATIONS.get(fixture.name()));
  }

  /**
   * Tests OpenNLP stems against the stems recorded from Hunspell's analyzer. A fixture
   * listed in {@link #STEM_DEVIATIONS} must differ, so a stale entry fails too.
   *
   * @param fixture The fixture.
   * @throws IOException Thrown if the fixture fails to load.
   */
  @ParameterizedTest(name = "stems {0}")
  @MethodSource("comparableStemFixtures")
  void testStemsAgainstHunspell(Fixture fixture) throws IOException {
    HunspellTestDictionaries.assertStems(fixture, STEM_DEVIATIONS.get(fixture.name()));
  }

  /** Checks that every listed deviation names a fixture its comparison runs on. */
  @Test
  void testDeviationsNameComparedFixtures() {
    final Set<String> all = fixtures().map(Fixture::name).collect(Collectors.toSet());
    final Set<String> comparable = comparableStemFixtures().map(Fixture::name)
        .collect(Collectors.toSet());
    Assertions.assertTrue(all.containsAll(RECOGNITION_DEVIATIONS.keySet()));
    Assertions.assertTrue(comparable.containsAll(STEM_DEVIATIONS.keySet()));
  }

  /**
   * Checks the package-private analysis operation, which preserves entry and affix
   * fields and answers an immutable list.
   *
   * @throws IOException Thrown if the fixture fails to load.
   */
  @Test
  void testMorphologicalAnalysis() throws IOException {
    final HunspellStemmer stemmer = new HunspellStemmer(load(PLURAL_FIELD, CARD_HOMONYMS));
    Assertions.assertEquals(CARD_ANALYSES, stemmer.analyze("cards"));
    Assertions.assertEquals(List.of(), stemmer.analyze("unlisted"));
    Assertions.assertThrows(IllegalArgumentException.class, () -> stemmer.analyze(null));
    Assertions.assertEquals(List.of(), stemmer.analyze(""));
    Assertions.assertThrows(UnsupportedOperationException.class, () -> stemmer.analyze("cards").clear());
  }

  /**
   * Checks the documented maximum number of CHECKSHARPS case variants. The word list
   * holds every lowercase spelling of the all-uppercase input, so without the limit
   * each of them would be a stem; with it, the listed uppercase form and some, but
   * not all, of the spellings are found.
   *
   * @throws IOException Thrown if the fixture fails to load.
   */
  @Test
  void testSharpVariantLimit() throws IOException {
    final int spellings = 1 << SHARP_S_POSITIONS;
    final String input = StringUtil.toUpperCase(DOUBLE_S.repeat(SHARP_S_POSITIONS));
    final StringBuilder words = new StringBuilder().append(2 * spellings + 1).append('\n')
        .append(input).append('\n');
    for (int mask = 0; mask < spellings; mask++) {
      final StringBuilder entry = new StringBuilder();
      for (int bit = 0; bit < SHARP_S_POSITIONS; bit++) {
        entry.append((mask & (1 << bit)) == 0 ? DOUBLE_S : SHARP_S);
      }
      words.append(entry).append('\n');
      words.append(StringUtil.toUpperCase(entry.substring(0, 1))).append(entry.substring(1)).append('\n');
    }
    final HunspellStemmer stemmer = new HunspellStemmer(load("CHECKSHARPS\n", words.toString()));
    final List<CharSequence> stems = stemmer.stemAll(input);
    Assertions.assertEquals(input, stems.get(0).toString());
    Assertions.assertTrue(stems.size() > 1, () -> "no sharp-s variant found: " + stems);
    Assertions.assertTrue(stems.size() < spellings, () -> "variants not limited: " + stems.size());
  }

  /**
   * Checks that the flag-set lists a lookup answers are immutable and shared across
   * concurrent stemming.
   *
   * @throws Exception Thrown if the fixture fails to load or a worker fails.
   */
  @Test
  void testSharedMorphologyAndImmutableFlags() throws Exception {
    final HunspellDictionary dictionary = load(PLURAL_FIELD, CARD_HOMONYMS);
    Assertions.assertThrows(UnsupportedOperationException.class, () -> dictionary.lookup("card").clear());
    Assertions.assertThrows(UnsupportedOperationException.class,
        () -> dictionary.lookup("card", true).add(new int[0]));
    final HunspellStemmer stemmer = new HunspellStemmer(dictionary);
    try (var executor = Executors.newFixedThreadPool(4)) {
      final List<Callable<List<String>>> tasks = new ArrayList<>();
      for (int i = 0; i < 64; i++) {
        tasks.add(() -> {
          Assertions.assertEquals(List.of("card"), stemmer.stemAll("cards"));
          Assertions.assertEquals(List.of(), stemmer.analyze("unlisted"));
          return stemmer.analyze("cards");
        });
      }
      for (var result : executor.invokeAll(tasks)) {
        Assertions.assertEquals(CARD_ANALYSES, result.get());
      }
    }
  }

  /**
   * Supplies morphology recorded from Hunspell.
   *
   * @return Affix content, dictionary content, input, and expected analysis.
   */
  private static Stream<String[]> morphology() {
    return Stream.of(
        new String[] {PLURAL, "1\ncard/A\n", "cards", "st:card fl:A"},
        new String[] {PLURAL_FIELD, "1\ncard/A po:noun\n",
            "cards", "st:card po:noun is:plural"},
        new String[] {"AM 2\nAM st:foot ts:present\nAM is:plural\nSFX A Y 1\nSFX A 0 s . 2\n",
            "1\nfeet/A\t1\n", "feets", "st:foot ts:present is:plural"},
        new String[] {"PFX B Y 1\nPFX B 0 re . dp:again\n" + PLURAL_FIELD,
            "1\ngo/AB po:verb\n", "regos", "dp:again st:go po:verb is:plural"},
        new String[] {COMPOUND + PLURAL_FIELD,
            "2\nriver/C po:noun\nboat/CA po:noun\n", "riverboats",
            "pa:river st:river po:noun pa:boats st:boat po:noun is:plural"},
        new String[] {PLURAL_FIELD, "1\ncard/A custom\n",
            "cards", "st:card is:plural"},
        new String[] {"NEEDAFFIX X\nSFX A Y 1\nSFX A 0 0 .\n", "1\nfoo/XA\n", "foo", "st:foo fl:A"},
        new String[] {"PFX C Y 1\nPFX C 0 pre .\n", "1\nfoo/C\n", "prefoo", "pre st:foo fl:C"},
        new String[] {"PFX C Y 1\nPFX C 0 pre .\n", "1\nfoo/C po:noun\n", "prefoo", "pre st:foo po:noun"},
        new String[] {"PFX B Y 1\nPFX B 0 re .\n" + PLURAL_FIELD,
            "1\ngo/AB po:verb\n", "regos", "fl:B st:go po:verb is:plural"},
        new String[] {"PFX B Y 1\nPFX B 0 re . dp:again\n" + PLURAL,
            "1\ngo/AB\n", "regos", "dp:again st:go fl:A"},
        new String[] {"SFX A Y 1\nSFX A 0 er/B .\nSFX B Y 1\nSFX B 0 s .\n",
            "1\nwalk/A po:verb\n", "walkers", "st:walk po:verb fl:A fl:B"},
        new String[] {COMPOUND, "2\nfoo/C id:1\nbar/C\n", "foobar", "pa:foo st:foo id:1 pa:bar"},
        new String[] {COMPOUND, "3\nfoo/C\nbar/C id:2\nbaz/C\n", "foobarbaz",
            "pa:foo st:foo pa:bar st:bar id:2 pa:baz"},
        new String[] {COMPOUND + PLURAL, "2\nfoo/C id:1\nbar/CA\n", "foobars",
            "pa:foo st:foo id:1 pa:bars st:bar fl:A"},
        new String[] {PLURAL, "1\neBook/A po:noun\n", "EBOOKS", "st:Ebook po:noun fl:A"});
  }

  /**
   * Analyses that include a rule adding and removing no material. Hunspell's analyzer
   * returns them in a different order, so they are compared as sets.
   *
   * @return Affix content, word list, input, and every expected analysis.
   */
  private static Stream<Arguments> zeroAffixAnalyses() {
    return Stream.of(
        Arguments.of("SFX A Y 1\nSFX A 0 0 . is:zero\n", "1\nbar/A\n", "bar",
            List.of("st:bar", "st:bar is:zero")),
        Arguments.of("PFX A Y 1\nPFX A 0 0 . dp:zero\n", "1\nbar/A\n", "bar",
            List.of("st:bar", "dp:zero st:bar fl:A")),
        Arguments.of(NEEDS_AFFIX_ZERO_RULES, "1\nlumen/NPQR po:noun\n", "lumen",
            List.of("st:lumen po:noun is:bare", "st:lumen po:noun is:plain",
                "st:lumen po:noun dp:none is:bare", "st:lumen po:noun dp:none is:plain")),
        Arguments.of(NEEDS_AFFIX_ZERO_RULES, "1\nlumen/NPQR po:noun\n", "lumenix",
            List.of("st:lumen po:noun dp:ix is:bare", "st:lumen po:noun dp:ix is:plain")));
  }

  /**
   * Tests that a rule without material is undone on its own and inside a continuation.
   *
   * @param affix The affix content.
   * @param words The word list.
   * @param input The analyzed word.
   * @param expected The distinct analyses, as recorded from Hunspell's analyzer.
   * @throws IOException Thrown if the fixture fails to load.
   */
  @ParameterizedTest
  @MethodSource("zeroAffixAnalyses")
  void testZeroAffixAnalyses(String affix, String words, String input, List<String> expected)
      throws IOException {
    final HunspellStemmer stemmer = new HunspellStemmer(load(affix, words));
    Assertions.assertEquals(new TreeSet<>(expected), new TreeSet<>(stemmer.analyze(input)));
  }

  /**
   * Tests that only the first listed homonym decides whether a spelling is forbidden.
   * Hunspell's spell checker accepts {@code reed} with the valid homonym listed first
   * and rejects it with the forbidden homonym listed first.
   *
   * @throws IOException Thrown if the fixture fails to load.
   */
  @Test
  void testForbiddenFirstHomonym() throws IOException {
    final String affix = "FORBIDDENWORD F\nCOMPOUNDFLAG K\nCOMPOUNDMIN 1\n";
    final HunspellStemmer allowed = new HunspellStemmer(load(affix, "2\nreed/T\nreed/KF\n"));
    Assertions.assertEquals(List.of("st:reed"), allowed.analyze("reed"));
    Assertions.assertEquals(List.of(), allowed.analyze("reedreed"));
    final HunspellStemmer forbidden = new HunspellStemmer(load(affix, "2\nreed/KF\nreed/T\n"));
    Assertions.assertEquals(List.of(), forbidden.analyze("reed"));
  }

  /**
   * Tests analyses against the field text recorded from Hunspell's analyzer, with
   * separator whitespace normalized to single spaces.
   *
   * @param affix The affix content.
   * @param words The word list.
   * @param input The analyzed word.
   * @param expected The recorded analysis.
   * @throws IOException Thrown if the fixture fails to load.
   */
  @ParameterizedTest
  @MethodSource("morphology")
  void testMorphologyFields(String affix, String words, String input, String expected) throws IOException {
    final HunspellStemmer stemmer = new HunspellStemmer(load(affix, words));
    Assertions.assertEquals(List.of(expected), stemmer.analyze(input));
  }
}
