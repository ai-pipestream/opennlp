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
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.Fixture;

import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.ACCEPTED;
import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.COMPOUND;
import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.NO_STEMS;
import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.PLURAL;
import static opennlp.tools.stemmer.hunspell.HunspellTestDictionaries.REJECTED;

/**
 * Dictionary fixtures for Hunspell conversion, casing, compounding, and morphology.
 * The fixtures are written for this test: each declares its own words and flags for
 * the rule it exercises. Each fixture also carries the acceptance and the stems
 * Hunspell returned for its input, recorded as described in
 * {@code dev/README-hunspell-dictionaries.md}.
 */
class HunspellCompatibilityTest {

  private static final String HUNGARIAN_HYPHEN = "LANG hu\nCOMPOUNDFLAG W\nCOMPOUNDFORBIDFLAG X\n";
  private static final String HUNGARIAN_WORDS = "3\nhegy/FX\ntető/W\nlakó\n";
  private static final String COMPOUND_FORBID = "COMPOUNDFLAG K\nCOMPOUNDPERMITFLAG L\n"
      + "COMPOUNDFORBIDFLAG M\nSFX T Y 2\nSFX T 0 wood/LK .\nSFX T 0 ward/LK .\n";
  private static final String COMPOUND_FORBID_WORDS = "3\nfire/T\nhouse/K\nfireward/M\n";
  private static final String ONLY_IN_COMPOUND = COMPOUND + "ONLYINCOMPOUND Q\n"
      + "COMPOUNDPERMITFLAG P\nSFX D Y 1\nSFX D 0 en/QP .\n";
  private static final String ONLY_IN_COMPOUND_WORDS = "2\nlamp/C\nglass/CD\n";
  private static final String FORBIDDEN_AFFIXED = "FORBIDDENWORD F\n" + COMPOUND
      + "SFX S Y 1\nSFX S 0 s .\n";
  private static final String FORBIDDEN_AFFIXED_WORDS = "4\nsun/CS\nlight/C\nroom/CS\nsunlightroom/FS\n";

  /**
   * Supplies the fixtures.
   *
   * @return The fixtures.
   */
  private static Stream<Fixture> fixtures() {
    return Stream.of(
        new Fixture("input-ligature", "ICONV 1\nICONV ﬁ fi\n" + PLURAL,
            "1\nfield/A\n", "ﬁelds", List.of("field"),
            ACCEPTED, List.of("field")),
        new Fixture("longest-input-match", "ICONV 2\nICONV æ ae\nICONV æx ax\n" + PLURAL,
            "1\nax/A\n", "æxs", List.of("ax"),
            ACCEPTED, List.of("ax")),
        new Fixture("input-end-anchor", "ICONV 1\nICONV z_ s\n" + PLURAL,
            "1\nquartz/A\n", "quartzz", List.of("quartz"),
            ACCEPTED, List.of("quartz")),
        new Fixture("output-conversion", "OCONV 1\nOCONV ae ä\n" + PLURAL,
            "1\nbaer/A\n", "baers", List.of("bär"),
            ACCEPTED, List.of("bär")),
        new Fixture("ignored-input-and-dictionary", "IGNORE ’\n" + PLURAL,
            "1\npe’arl/A\n", "pear’ls", List.of("pearl"),
            ACCEPTED, List.of("pearl")),
        new Fixture("ignored-affix", "IGNORE ’\nSFX A Y 1\nSFX A 0 s’ .\n",
            "1\npearl/A\n", "pearls", List.of("pearl"),
            ACCEPTED, List.of("pearl")),
        new Fixture("keepcase-exact", "KEEPCASE K\n" + PLURAL,
            "1\ncard/AK\n", "cards", List.of("card"),
            ACCEPTED, List.of("card")),
        new Fixture("keepcase-title", "KEEPCASE K\n" + PLURAL,
            "1\ncard/AK\n", "Cards", List.of("Cards"),
            REJECTED, List.of("card")),
        new Fixture("keepcase-uppercase", "KEEPCASE K\n" + PLURAL,
            "1\ncard/AK\n", "CARDS", List.of("CARDS"),
            REJECTED, List.of("card")),
        new Fixture("mixed-case-is-not-lowercase", PLURAL,
            "1\ncard/A\n", "cArds", List.of("cArds"),
            REJECTED, NO_STEMS),
        new Fixture("uppercase-proper-name", PLURAL,
            "1\nMaren/A\n", "MARENS", List.of("Maren"),
            ACCEPTED, List.of("Maren")),
        new Fixture("turkish-case", "LANG tr\n" + PLURAL,
            "1\nılık/A\n", "ILIKS", List.of("ılık"),
            ACCEPTED, List.of("ılık")),
        new Fixture("complex-prefixes", "COMPLEXPREFIXES\nPFX B Y 1\n"
            + "PFX B 0 re/C .\nPFX C Y 1\nPFX C 0 un .\n" + PLURAL,
            "1\ndo/AB\n", "unredos", List.of("do"),
            ACCEPTED, NO_STEMS),
        new Fixture("complex-prefix-requires-continuation", "COMPLEXPREFIXES\n"
            + "PFX B Y 1\nPFX B 0 re .\nPFX C Y 1\nPFX C 0 un .\n",
            "1\ndo/BC\n", "unredo", List.of("unredo"),
            REJECTED, NO_STEMS),
        new Fixture("complex-prefix-single-suffix", "COMPLEXPREFIXES\n"
            + "SFX B Y 1\nSFX B 0 er/C .\nSFX C Y 1\nSFX C 0 s .\n",
            "1\nwalk/B\n", "walkers", List.of("walkers"),
            REJECTED, NO_STEMS),
        new Fixture("explicit-stem", PLURAL, "1\nfeet/A st:foot is:plural\n",
            "feet", List.of("foot"),
            ACCEPTED, List.of("foot")),
        new Fixture("affixed-explicit-stem", PLURAL,
            "1\nfeet/A st:foot is:plural\n", "feets", List.of("foot"),
            ACCEPTED, List.of("foot")),
        new Fixture("morphology-alias", "AM 1\nAM st:goose is:plural\n" + PLURAL,
            "1\ngeese/A\t1\n", "geese", List.of("goose"),
            ACCEPTED, List.of("goose")),
        new Fixture("derivational-suffix", "SFX A Y 1\nSFX A 0 ness/B . ds:noun\n"
            + "SFX B Y 1\nSFX B 0 es . is:plural\n",
            "1\nkind/A po:adj\n", "kindnesses", List.of("kindness"),
            ACCEPTED, List.of("kindness")),
        new Fixture("surface-prefix", PLURAL, "1\nroot/A sp:pre st:base\n",
            "roots", List.of("prebase"),
            ACCEPTED, List.of("prebase")),
        new Fixture("sharp-s-uppercase", "CHECKSHARPS\n" + PLURAL,
            "1\nsoße/A\n", "SOSSES", List.of("soße"),
            ACCEPTED, NO_STEMS),
        new Fixture("sharp-s-keepcase", "CHECKSHARPS\nKEEPCASE K\n" + PLURAL,
            "1\nsoße/AK\n", "SOSSES", List.of("soße"),
            ACCEPTED, NO_STEMS),
        new Fixture("forbidden-warning", "WARN W\nFORBIDWARN\n" + PLURAL,
            "1\ncard/AW\n", "cards", List.of("cards"),
            REJECTED, List.of("card")),
        new Fixture("compound-rule", "COMPOUNDMIN 1\nCOMPOUNDRULE 1\nCOMPOUNDRULE RS\n"
            + PLURAL, "2\nriver/R\nboat/AS\n", "riverboats", List.of("river", "boat"),
            ACCEPTED, List.of("river")),
        new Fixture("compound-rule-order", "COMPOUNDMIN 1\nCOMPOUNDRULE 1\nCOMPOUNDRULE RS\n",
            "2\nriver/R\nboat/S\n", "boatriver", List.of("boatriver"),
            REJECTED, NO_STEMS),
        new Fixture("compound-rule-star", "COMPOUNDMIN 1\nCOMPOUNDRULE 1\nCOMPOUNDRULE R*S\n",
            "3\nriver/R\nstone/R\nboat/S\n", "riverstoneboat", List.of("river", "stone", "boat"),
            ACCEPTED, List.of("riverstoneboat")),
        new Fixture("compound-rule-optional", "COMPOUNDMIN 1\nCOMPOUNDRULE 1\nCOMPOUNDRULE R?TS\n",
            "3\nriver/R\nstone/T\nboat/S\n", "stoneboat", List.of("stone", "boat"),
            ACCEPTED, List.of("stoneboat")),
        new Fixture("compound-rule-long", "FLAG long\nCOMPOUNDMIN 1\nCOMPOUNDRULE 1\n"
            + "COMPOUNDRULE (Ra)(Sa)\n", "2\nriver/Ra\nboat/Sa\n",
            "riverboat", List.of("river", "boat"),
            ACCEPTED, List.of("riverboat")),
        new Fixture("compound-rule-numeric", "FLAG num\nCOMPOUNDMIN 1\nCOMPOUNDRULE 1\n"
            + "COMPOUNDRULE (12)(34)\n", "2\nriver/12\nboat/34\n",
            "riverboat", List.of("river", "boat"),
            ACCEPTED, List.of("riverboat")),
        new Fixture("compound-rule-homonyms", "COMPOUNDMIN 1\nCOMPOUNDRULE 1\n"
            + "COMPOUNDRULE RTS\n", "3\nriver/R\nriver/T\nboat/S\n",
            "riverboat", List.of("riverboat"),
            REJECTED, NO_STEMS),
        new Fixture("compound-force-uppercase", COMPOUND + "FORCEUCASE U\n",
            "2\nriver/C\nboat/CU\n", "Riverboat", List.of("river", "boat"),
            ACCEPTED, List.of("river")),
        new Fixture("compound-force-uppercase-reject", COMPOUND + "FORCEUCASE U\n",
            "2\nriver/C\nboat/CU\n", "riverboat", List.of("riverboat"),
            REJECTED, List.of("river")),
        new Fixture("compound-root-count", COMPOUND + "COMPOUNDROOT R\nCOMPOUNDWORDMAX 3\n",
            "3\nrain/CR\ncoat/C\nrack/C\n", "raincoatrack", List.of("raincoatrack"),
            REJECTED, NO_STEMS),
        new Fixture("compound-root-count-accept", COMPOUND + "COMPOUNDROOT R\nCOMPOUNDWORDMAX 4\n",
            "3\nrain/CR\ncoat/C\nrack/C\n", "raincoatrack", List.of("rain", "coat", "rack"),
            ACCEPTED, List.of("raincoat")),
        new Fixture("compound-replacement-check", COMPOUND + "CHECKCOMPOUNDREP\nREP 1\nREP coat boat\n",
            "3\nrain/C\ncoat/C\nrainboat\n", "raincoat", List.of("raincoat"),
            REJECTED, List.of("rain")),
        new Fixture("compound-pattern", COMPOUND + "CHECKCOMPOUNDPATTERN 1\n"
            + "CHECKCOMPOUNDPATTERN er b\n", "2\nriver/C\nboat/C\n",
            "riverboat", List.of("riverboat"),
            REJECTED, NO_STEMS),
        new Fixture("compound-pattern-flags", COMPOUND + "CHECKCOMPOUNDPATTERN 1\n"
            + "CHECKCOMPOUNDPATTERN er/X b/Y\n", "2\nriver/C\nboat/CY\n",
            "riverboat", List.of("river", "boat"),
            ACCEPTED, List.of("river")),
        new Fixture("compound-pattern-replacement", COMPOUND + "CHECKCOMPOUNDPATTERN 1\n"
            + "CHECKCOMPOUNDPATTERN er b X\n", "2\nriver/C\nboat/C\n",
            "rivXoat", List.of("river", "boat"),
            ACCEPTED, NO_STEMS),
        new Fixture("compound-simplified-triple", COMPOUND + "CHECKCOMPOUNDTRIPLE\nSIMPLIFIEDTRIPLE\n",
            "2\nmill/C\nloom/C\n", "milloom", List.of("mill", "loom"),
            ACCEPTED, NO_STEMS),
        new Fixture("compound-more-suffixes", COMPOUND + "COMPOUNDMORESUFFIXES\n"
            + "SFX A Y 1\nSFX A 0 er/B .\nSFX B Y 1\nSFX B 0 s .\n",
            "2\nriver/C\nboat/CA\n", "riverboaters", List.of("river", "boat"),
            ACCEPTED, List.of("riverboat")),
        new Fixture("compound-syllable-limit", COMPOUND + "LANG hu\nCOMPOUNDWORDMAX 2\n"
            + "COMPOUNDSYLLABLE 4 aeiouy\n", "3\nray/C\nme/C\nfa/C\n",
            "raymefa", List.of("ray", "me", "fa"),
            ACCEPTED, List.of("rayme")),
        new Fixture("compound-syllable-limit-reject", COMPOUND + "LANG hu\nCOMPOUNDWORDMAX 2\n"
            + "COMPOUNDSYLLABLE 2 aeiouy\n", "3\nray/C\nme/C\nfa/C\n",
            "raymefa", List.of("raymefa"),
            REJECTED, NO_STEMS),
        new Fixture("break-default", PLURAL, "2\nriver/A\nboat/A\n",
            "rivers-boats", List.of("river", "boat"),
            ACCEPTED, NO_STEMS),
        new Fixture("break-recursive", PLURAL, "2\nriver/A\nboat/A\n",
            "rivers-boats-rivers", List.of("river", "boat"),
            ACCEPTED, NO_STEMS),
        new Fixture("break-start", PLURAL, "1\nriver/A\n", "-rivers", List.of("river"),
            ACCEPTED, NO_STEMS),
        new Fixture("break-end", PLURAL, "1\nriver/A\n", "rivers-", List.of("river"),
            ACCEPTED, NO_STEMS),
        new Fixture("break-custom", "BREAK 1\nBREAK ::\n" + PLURAL,
            "2\nriver/A\nboat/A\n", "rivers::boats", List.of("river", "boat"),
            ACCEPTED, NO_STEMS),
        new Fixture("break-disabled", "BREAK 0\n" + PLURAL,
            "2\nriver/A\nboat/A\n", "rivers-boats", List.of("rivers-boats"),
            REJECTED, NO_STEMS),
        new Fixture("break-unknown-part", PLURAL, "1\nriver/A\n",
            "rivers-absent", List.of("rivers-absent"),
            REJECTED, NO_STEMS),
        new Fixture("break-internal-only", "BREAK 1\nBREAK -\n" + PLURAL,
            "1\nriver/A\n", "-rivers", List.of("-rivers"),
            REJECTED, NO_STEMS),
        new Fixture("replacement-trailing-fields", COMPOUND + "CHECKCOMPOUNDREP\n"
            + "REP 1\nREP coat boat trailing_metadata\n",
            "3\nrain/C\ncoat/C\nrainboat\n", "raincoat", List.of("raincoat"),
            REJECTED, List.of("rain")),
        new Fixture("replacement-morphology", COMPOUND + "CHECKCOMPOUNDREP\n",
            "3\nrain/C\ncoat/C\nrainboat ph:raincoat\n", "raincoat", List.of("raincoat"),
            REJECTED, List.of("rain")),
        new Fixture("replacement-morphology-arrow", COMPOUND + "CHECKCOMPOUNDREP\n",
            "3\nrain/C\ncoat/C\nrainboat ph:coat->boat\n", "raincoat", List.of("raincoat"),
            REJECTED, List.of("rain")),
        new Fixture("replacement-morphology-star-unlisted", COMPOUND + "CHECKCOMPOUNDREP\n" + PLURAL,
            "3\nrain/C\ncoat/C\nrainboats/A ph:raincoats*\n", "raincoat", List.of("rain", "coat"),
            ACCEPTED, List.of("rain")),
        new Fixture("replacement-morphology-star", COMPOUND + "CHECKCOMPOUNDREP\n" + PLURAL,
            "4\nrain/C\ncoat/C\nrainboats/A ph:raincoats*\nrainboat\n",
            "raincoat", List.of("raincoat"),
            REJECTED, List.of("rain")),
        new Fixture("derivation-surface-prefix", "PFX U Y 1\nPFX U 0 mis . dp:pfx_mis sp:mis\n"
            + "SFX A Y 1\nSFX A 0 ment/U . ds:der_ment\n", "1\nmanage/A po:verb\n",
            "mismanagement", List.of("mismanagement"),
            ACCEPTED, List.of("mismanagement")),
        new Fixture("derivation-inflectional-prefix", "PFX P Y 1\nPFX P 0 mis . ip:mis\n"
            + "SFX R Y 1\nSFX R 0 ment/P . ds:DER\n", "1\nmanage/R po:verb\n",
            "mismanagement", List.of("management"),
            ACCEPTED, List.of("management")),
        new Fixture("compound-pattern-substitution-only", COMPOUND + "CHECKCOMPOUNDPATTERN 2\n"
            + "CHECKCOMPOUNDPATTERN a t k\nCHECKCOMPOUNDPATTERN ea ta y\n",
            "2\nsea/C\ntable/C\n", "sekable", List.of("sea", "table"),
            ACCEPTED, NO_STEMS),
        new Fixture("compound-pattern-substitution-second", COMPOUND + "CHECKCOMPOUNDPATTERN 2\n"
            + "CHECKCOMPOUNDPATTERN a t k\nCHECKCOMPOUNDPATTERN ea ta y\n",
            "2\nsea/C\ntable/C\n", "syble", List.of("sea", "table"),
            ACCEPTED, NO_STEMS),
        new Fixture("compound-duplicate-last-parts", COMPOUND + "CHECKCOMPOUNDDUP\n",
            "2\nfoo/C\nbar/C\n", "foofoobar", List.of("foo", "bar"),
            ACCEPTED, List.of("foofoo")),
        new Fixture("compound-duplicate-reject", COMPOUND + "CHECKCOMPOUNDDUP\n",
            "2\nfoo/C\nbar/C\n", "foobarbar", List.of("foobarbar"),
            REJECTED, NO_STEMS),
        new Fixture("compound-forbid-entry", COMPOUND_FORBID, COMPOUND_FORBID_WORDS,
            "firewardhouse", List.of("firewardhouse"),
            REJECTED, NO_STEMS),
        new Fixture("compound-forbid-entry-other-suffix", COMPOUND_FORBID, COMPOUND_FORBID_WORDS,
            "firewoodhouse", List.of("fire", "house"),
            ACCEPTED, List.of("firewood")),
        new Fixture("compound-only-suffix-at-end", ONLY_IN_COMPOUND, ONLY_IN_COMPOUND_WORDS,
            "lampglassen", List.of("lampglassen"),
            REJECTED, NO_STEMS),
        new Fixture("compound-only-suffix-inside", ONLY_IN_COMPOUND, ONLY_IN_COMPOUND_WORDS,
            "glassenlamp", List.of("glass", "lamp"),
            ACCEPTED, NO_STEMS),
        new Fixture("compound-replacement-inner", COMPOUND + "CHECKCOMPOUNDREP\n"
            + "REP 1\nREP wallpaper wall_paper\n",
            "3\nwall/C\npaper/C\nwall paper\n", "paperwallpaper",
            List.of("paperwallpaper"),
            REJECTED, List.of("paperwall")),
        new Fixture("compound-replacement-inner-unaffected", COMPOUND + "CHECKCOMPOUNDREP\n"
            + "REP 1\nREP wallpaper wall_paper\n",
            "3\nwall/C\npaper/C\nwall paper\n", "paperwall",
            List.of("paper", "wall"),
            ACCEPTED, List.of("paper")),
        new Fixture("mixed-case-initial-capital", "PFX b Y 1\nPFX b e ra e\n",
            "1\neMarket/b\n", "RaMarket", List.of("eMarket"),
            ACCEPTED, NO_STEMS),
        new Fixture("mixed-case-initial-capital-entry", "PFX b Y 1\nPFX b e ra e\n",
            "1\neMarket/b\n", "EMarket", List.of("eMarket"),
            ACCEPTED, NO_STEMS),
        new Fixture("forbidden-affixed-blocks-compound", FORBIDDEN_AFFIXED, FORBIDDEN_AFFIXED_WORDS,
            "sunlightrooms", List.of("sunlightrooms"),
            REJECTED, List.of("sunlightroom")),
        new Fixture("forbidden-affixed-other-order", FORBIDDEN_AFFIXED, FORBIDDEN_AFFIXED_WORDS,
            "roomlightsuns", List.of("room", "light", "sun"),
            ACCEPTED, List.of("roomlightsun")),
        new Fixture("turkic-capitalized-entry", "LANG tr\n", "1\nİnci\n",
            "İNCİ", List.of("İnci"),
            REJECTED, List.of("İnci")),
        new Fixture("break-number-sign", "BREAK 1\nBREAK #\n" + PLURAL,
            "2\nriver/A\nboat/A\n", "rivers#boats", List.of("river", "boat"),
            ACCEPTED, NO_STEMS),
        new Fixture("flag-number-sign", "NEEDAFFIX #\n" + PLURAL,
            "2\nfoo/#A\nbar/A\n", "foos", List.of("foo"),
            ACCEPTED, List.of("foo")),
        new Fixture("flag-number-sign-virtual-stem", "NEEDAFFIX #\n" + PLURAL,
            "2\nfoo/#A\nbar/A\n", "foo", List.of("foo"),
            REJECTED, NO_STEMS),
        new Fixture("hidden-capital-mixed-case", PLURAL, "1\neBook/A\n", "EBOOKS", List.of("Ebook"),
            ACCEPTED, List.of("Ebook")),
        new Fixture("hidden-capital-initial-capital", PLURAL, "1\neBook/A\n", "Ebooks", List.of("Ebooks"),
            REJECTED, List.of("Ebook")),
        new Fixture("hidden-capital-all-caps-entry", "SFX M Y 1\nSFX M 0 less .\n",
            "1\nRADAR/M\n", "RADARLESS", List.of("Radar"),
            ACCEPTED, List.of("Radar")),
        new Fixture("hidden-capital-listed-form-wins", PLURAL, "2\neBook/A\nEbook\n",
            "EBOOKS", List.of("EBOOKS"),
            REJECTED, NO_STEMS),
        new Fixture("hidden-capital-unflagged-all-caps", PLURAL, "1\nACME\n", "Acme", List.of("Acme"),
            REJECTED, NO_STEMS),
        new Fixture("hidden-capital-not-in-compound", COMPOUND + PLURAL, "2\neBook/AC\ncover/C\n",
            "EBOOKCOVER", List.of("EBOOKCOVER"),
            REJECTED, NO_STEMS),
        new Fixture("hungarian-hyphen-moving-rule", HUNGARIAN_HYPHEN, HUNGARIAN_WORDS,
            "hegytető-lakó", List.of("hegy", "tető", "lakó"),
            ACCEPTED, NO_STEMS),
        new Fixture("hungarian-hyphen-rule-needs-hyphen", HUNGARIAN_HYPHEN, HUNGARIAN_WORDS,
            "hegytető", List.of("hegytető"),
            REJECTED, NO_STEMS),
        new Fixture("hungarian-hyphen-rule-needs-language", HUNGARIAN_HYPHEN.replace("LANG hu", "LANG de"),
            HUNGARIAN_WORDS, "hegytető-lakó", List.of("hegytető-lakó"),
            REJECTED, NO_STEMS),
        new Fixture("hungarian-hyphen-rule-needs-flag", HUNGARIAN_HYPHEN,
            "3\nhegy/X\ntető/W\nlakó\n", "hegytető-lakó", List.of("hegytető-lakó"),
            REJECTED, NO_STEMS),
        new Fixture("hungarian-hyphen-rule-first-part-only", HUNGARIAN_HYPHEN, HUNGARIAN_WORDS,
            "lakó-hegytető", List.of("lakó-hegytető"),
            REJECTED, NO_STEMS),
        new Fixture("apostrophe-all-caps", "PFX P Y 1\nPFX P 0 d' .\n", "1\nOrient/P\n",
            "D'ORIENT", List.of("Orient"),
            ACCEPTED, NO_STEMS),
        new Fixture("apostrophe-capitalized", "PFX P Y 1\nPFX P 0 d' .\n", "1\nOrient/P\n",
            "D'Orient", List.of("Orient"),
            ACCEPTED, NO_STEMS),
        new Fixture("trailing-period", PLURAL, "2\ntext/A\netc.\n", "texts.", List.of("text"),
            ACCEPTED, List.of("text")),
        new Fixture("trailing-periods", PLURAL, "2\ntext/A\netc.\n", "texts...", List.of("text"),
            ACCEPTED, List.of("text")),
        new Fixture("trailing-period-entry", PLURAL, "2\ntext/A\netc.\n", "etc.", List.of("etc."),
            ACCEPTED, List.of("etc.")),
        new Fixture("trailing-period-not-added", PLURAL, "2\ntext/A\netc.\n", "etc", List.of("etc"),
            REJECTED, NO_STEMS),
        new Fixture("numeric-flag-maximum", "FLAG num\nSFX 65535 Y 1\nSFX 65535 0 s .\n",
            "1\ndog/65535\n", "dogs", List.of("dog"),
            ACCEPTED, List.of("dog")),
        // the manual's compound example: OpenNLP returns the part stems, Hunspell one stem
        new Fixture("compound-part-stems", COMPOUND + PLURAL, "2\nriver/C\nboat/CA\n",
            "riverboats", List.of("river", "boat"),
            ACCEPTED, List.of("riverboat")));
  }

  /**
   * Supplies the fixtures whose stems are compared with Hunspell's.
   *
   * @return The fixtures with comparable stems.
   */
  private static Stream<Fixture> comparableStemFixtures() {
    return fixtures().filter(Fixture::hasComparableStems);
  }

  /**
   * The fixtures whose recognition deliberately differs from Hunspell's spell
   * checker, each with the manual's reason.
   */
  private static final Map<String, String> RECOGNITION_DEVIATIONS = Map.of(
      "turkic-capitalized-entry", "Hunspell's spell checker rejects what its analyzer stems");

  /**
   * The fixtures with comparable stems whose stems differ from the stems Hunspell
   * returned, each with the reason.
   */
  private static final Map<String, String> STEM_DEVIATIONS = Map.ofEntries(
      Map.entry("complex-prefixes", "Hunspell's analyzer returns no stem under COMPLEXPREFIXES"),
      Map.entry("sharp-s-uppercase", "Hunspell's analyzer does not expand SS to a sharp s"),
      Map.entry("sharp-s-keepcase", "Hunspell's analyzer does not expand SS to a sharp s"),
      Map.entry("mixed-case-initial-capital", "Hunspell's analyzer returns no stem"),
      Map.entry("mixed-case-initial-capital-entry", "Hunspell's analyzer returns no stem"),
      Map.entry("apostrophe-all-caps", "Hunspell's analyzer does not undo an elided-article prefix"),
      Map.entry("apostrophe-capitalized", "Hunspell's analyzer does not undo an elided-article prefix"),
      Map.entry("break-start", "Hunspell's analyzer returns no stem for BREAK forms"),
      Map.entry("break-end", "Hunspell's analyzer returns no stem for BREAK forms"));

  /**
   * Checks the OpenNLP stems of one fixture.
   *
   * @param fixture The dictionary and assertion.
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
}
