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

package opennlp.tools.tokenize;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.formats.ResourceAsStreamFactory;
import opennlp.tools.tokenize.DummyTokenizerFactory.DummyContextGenerator;
import opennlp.tools.tokenize.DummyTokenizerFactory.DummyDictionary;
import opennlp.tools.tokenize.lang.Factory;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.model.ArtifactProvider;
import opennlp.tools.util.normalizer.CodePointSet;

/**
 * Tests for the {@link TokenizerFactory} class.
 */
public class TokenizerFactoryTest {

  private static final Locale LOCALE_DUTCH = Locale.of("nl");
  private static final Locale LOCALE_POLISH = Locale.of("pl");
  private static final Locale LOCALE_PORTUGUESE = Locale.of("pt");
  private static final Locale LOCALE_SPANISH = Locale.of("es");

  private static ObjectStream<TokenSample> createSampleStream() throws IOException {
    InputStreamFactory in = new ResourceAsStreamFactory(
        TokenizerFactoryTest.class, "/opennlp/tools/tokenize/token.train");

    return new TokenSampleStream(new PlainTextByLineStream(in, StandardCharsets.UTF_8));
  }

  private static TokenizerModel train(TokenizerFactory factory)
      throws IOException {
    return TokenizerME.train(createSampleStream(), factory, TrainingParameters.defaultParams());
  }

  private static TokenizerCharacterPolicy legacyPolicy(String expression) {
    TokenizerCharacterPolicy policy = new Factory().importLegacyAlphanumeric(expression);
    Assertions.assertNotNull(policy, () -> "Missing legacy fixture: " + expression);
    return policy;
  }

  private static void assertPolicyEquals(TokenizerCharacterPolicy expected,
      TokenizerCharacterPolicy actual) {
    Assertions.assertEquals(expected.getLetters(), actual.getLetters());
    Assertions.assertEquals(expected.getDigits(), actual.getDigits());
    Assertions.assertEquals(expected.getMarks(), actual.getMarks());
  }

  private static TestTokenizerFactory loadedFactory(Map<String, String> manifest) {
    TestTokenizerFactory factory = new TestTokenizerFactory();
    factory.load(new MapArtifactProvider(manifest));
    return factory;
  }

  @Test
  void testExplicitPolicyManifestRoundTripPreservesEverySet() throws InvalidFormatException {
    TokenizerCharacterPolicy expected = TokenizerCharacterPolicy.of(
        CodePointSet.of('a', 0x10400), CodePointSet.of('7'), CodePointSet.of(0x0301));
    TokenizerFactory written = new TokenizerFactory("x-test", null, false, expected);
    Map<String, String> manifest = written.createManifestEntries();

    TestTokenizerFactory loaded = loadedFactory(manifest);
    loaded.validateArtifactMap();

    assertPolicyEquals(expected, loaded.getTokenizerCharacterPolicy());
    Assertions.assertFalse(loaded.isUseAlphaNumericOptimization());
  }

  @Test
  void testInvalidPolicyManifestRejectedEvenWhenOptimizationDisabled() {
    Map<String, String> manifest = new HashMap<>();
    manifest.put("useAlphaNumericOptimization", "false");
    manifest.put("tokenizerCharacterPolicyVersion", "1");
    manifest.put("tokenizerCharacterPolicyLetters", "61,,62");
    manifest.put("tokenizerCharacterPolicyDigits", "30");
    manifest.put("tokenizerCharacterPolicyMarks", "");

    InvalidFormatException error = Assertions.assertThrows(InvalidFormatException.class,
        () -> loadedFactory(manifest).validateArtifactMap());
    Assertions.assertTrue(error.getMessage().contains("character policy"));
  }

  @Test
  void testUnsupportedPolicyVersionRejected() {
    Map<String, String> manifest = new HashMap<>();
    manifest.put("useAlphaNumericOptimization", "false");
    manifest.put("tokenizerCharacterPolicyVersion", "2");

    InvalidFormatException error = Assertions.assertThrows(InvalidFormatException.class,
        () -> loadedFactory(manifest).validateArtifactMap());
    Assertions.assertTrue(error.getCause().getMessage().contains("version"));
  }

  @Test
  void testPolicyEntriesWithoutVersionAreRejected() {
    TestTokenizerFactory factory = loadedFactory(Map.of(
        "useAlphaNumericOptimization", "false",
        "tokenizerCharacterPolicyLetters", "61"));

    InvalidFormatException error = Assertions.assertThrows(InvalidFormatException.class,
        factory::validateArtifactMap);
    Assertions.assertTrue(error.getCause().getMessage().contains("require"));
  }

  @Test
  void testMalformedOptimizationFlagIsRejected() {
    TestTokenizerFactory factory = loadedFactory(Map.of(
        "useAlphaNumericOptimization", "yes"));

    InvalidFormatException error = Assertions.assertThrows(InvalidFormatException.class,
        factory::validateArtifactMap);
    Assertions.assertTrue(error.getCause().getMessage().contains("useAlphaNumericOptimization"));
  }

  @ParameterizedTest
  @ValueSource(strings = {"en", "eng", "es", "spa", "it", "ita", "pt", "por", "ca",
      "cat", "pl", "pol", "de", "deu", "ger", "fr", "fre", "fra", "nl", "nld", "dut"})
  void testSupportedLatinLanguageDefaults(String language) {
    TokenizerCharacterPolicy policy = new TokenizerFactory(language, null, true, null)
        .getTokenizerCharacterPolicy();

    Assertions.assertTrue(policy.test("café"));
    Assertions.assertTrue(policy.test("cafe\u0301"));
    Assertions.assertTrue(policy.test("A7"));
    Assertions.assertFalse(policy.test("中文"));
    Assertions.assertFalse(policy.test("😀"));
    Assertions.assertFalse(policy.test("\u0301a"));
  }

  @Test
  void testUnknownLanguageUsesConservativeAsciiDefault() {
    TokenizerCharacterPolicy policy = new TokenizerFactory("x-test", null, true, null)
        .getTokenizerCharacterPolicy();

    assertPolicyEquals(TokenizerCharacterPolicy.ascii(), policy);
  }

  @Test
  void testConstructorDoesNotDispatchToOverridableInit() {
    Assertions.assertDoesNotThrow(ConstructorDispatchFactory::new);
  }

  @Test
  void testLegacyModelWithoutExpressionUsesAsciiPolicy() throws InvalidFormatException {
    TestTokenizerFactory factory = loadedFactory(
        Map.of("useAlphaNumericOptimization", "true"));
    factory.validateArtifactMap();
    assertPolicyEquals(TokenizerCharacterPolicy.ascii(),
        factory.getTokenizerCharacterPolicy());
  }

  @Test
  void testKnownLegacyExpressionImportsBoundedPolicy() throws InvalidFormatException {
    String expression = "^[A-Za-z0-9äéöüÄÉÖÜß]+$";
    TestTokenizerFactory factory = loadedFactory(Map.of(
        "useAlphaNumericOptimization", "true", "alphaNumericPattern", expression));
    factory.validateArtifactMap();
    assertPolicyEquals(legacyPolicy(expression), factory.getTokenizerCharacterPolicy());
  }

  @Test
  void testUnsupportedLegacyExpressionGivesMigrationGuidance() {
    TestTokenizerFactory factory = loadedFactory(Map.of(
        "useAlphaNumericOptimization", "true", "alphaNumericPattern", "^[a-f]+$"));

    InvalidFormatException error = Assertions.assertThrows(InvalidFormatException.class,
        factory::validateArtifactMap);
    Assertions.assertTrue(error.getCause().getMessage().contains("explicit"));
    Assertions.assertTrue(error.getCause().getMessage().contains("retrain"));
  }

  private static Dictionary loadAbbDictionary(Locale loc) throws IOException {
    final String abbrevDict;
    if (loc.equals(LOCALE_DUTCH)) {
      abbrevDict = "opennlp/tools/lang/abb_NL.xml";
    } else if (loc.equals(Locale.GERMAN)) {
      abbrevDict = "opennlp/tools/lang/abb_DE.xml";
    } else if (loc.equals(Locale.FRENCH)) {
      abbrevDict = "opennlp/tools/lang/abb_FR.xml";
    } else if (loc.equals(Locale.ITALIAN)) {
      abbrevDict = "opennlp/tools/lang/abb_IT.xml";
    } else if (loc.equals(LOCALE_POLISH)) {
      abbrevDict = "opennlp/tools/lang/abb_PL.xml";
    } else if (loc.equals(LOCALE_PORTUGUESE)) {
      abbrevDict = "opennlp/tools/lang/abb_PT.xml";
    } else if (loc.equals(LOCALE_SPANISH)) {
      abbrevDict = "opennlp/tools/lang/abb_ES.xml";
    } else {
      abbrevDict = "opennlp/tools/lang/abb_EN.xml";
    }
    return new Dictionary(TokenizerFactoryTest.class.getClassLoader()
            .getResourceAsStream(abbrevDict));
  }

  @Test
  void testDefault() throws IOException {

    Dictionary dic = loadAbbDictionary(Locale.ENGLISH);
    final String lang = "eng";

    TokenizerCharacterPolicy policy = TokenizerCharacterPolicy.ascii();
    TokenizerModel model = train(new TokenizerFactory(lang, dic, false, policy));

    TokenizerFactory factory = model.getFactory();
    Assertions.assertNotNull(factory.getAbbreviationDictionary());
    Assertions.assertInstanceOf(DefaultTokenContextGenerator.class, factory.getContextGenerator());

    assertPolicyEquals(policy, factory.getTokenizerCharacterPolicy());
    Assertions.assertEquals(lang, factory.getLanguageCode());
    Assertions.assertEquals(lang, model.getLanguage());
    Assertions.assertFalse(factory.isUseAlphaNumericOptimization());

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    model.serialize(out);
    ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());

    TokenizerModel fromSerialized = new TokenizerModel(in);

    factory = fromSerialized.getFactory();
    Assertions.assertNotNull(factory.getAbbreviationDictionary());
    Assertions.assertInstanceOf(DefaultTokenContextGenerator.class, factory.getContextGenerator());

    assertPolicyEquals(policy, factory.getTokenizerCharacterPolicy());
    Assertions.assertEquals(lang, factory.getLanguageCode());
    Assertions.assertEquals(lang, model.getLanguage());
    Assertions.assertFalse(factory.isUseAlphaNumericOptimization());
  }

  @Test
  void testNullDict() throws IOException {

    Dictionary dic = null;
    final String lang = "eng";

    TokenizerCharacterPolicy policy = TokenizerCharacterPolicy.ascii();
    TokenizerModel model = train(new TokenizerFactory(lang, dic, false, policy));

    TokenizerFactory factory = model.getFactory();
    Assertions.assertNull(factory.getAbbreviationDictionary());
    Assertions.assertInstanceOf(DefaultTokenContextGenerator.class, factory.getContextGenerator());

    assertPolicyEquals(policy, factory.getTokenizerCharacterPolicy());
    Assertions.assertEquals(lang, factory.getLanguageCode());
    Assertions.assertEquals(lang, model.getLanguage());
    Assertions.assertFalse(factory.isUseAlphaNumericOptimization());

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    model.serialize(out);
    ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());

    TokenizerModel fromSerialized = new TokenizerModel(in);

    factory = fromSerialized.getFactory();
    Assertions.assertNull(factory.getAbbreviationDictionary());
    Assertions.assertInstanceOf(DefaultTokenContextGenerator.class, factory.getContextGenerator());

    assertPolicyEquals(policy, factory.getTokenizerCharacterPolicy());
    Assertions.assertEquals(lang, factory.getLanguageCode());
    Assertions.assertEquals(lang, model.getLanguage());
    Assertions.assertFalse(factory.isUseAlphaNumericOptimization());
  }

  @Test
  void testCustomPatternAndAlphaOpt() throws IOException {

    Dictionary dic = null;
    final String lang = "spa";
    String pattern = "^[0-9a-záéíóúüýñA-ZÁÉÍÓÚÝÑ]+$";

    TokenizerCharacterPolicy policy = legacyPolicy(pattern);
    TokenizerModel model = train(new TokenizerFactory(lang, dic, true, policy));

    TokenizerFactory factory = model.getFactory();
    Assertions.assertNull(factory.getAbbreviationDictionary());
    Assertions.assertInstanceOf(DefaultTokenContextGenerator.class, factory.getContextGenerator());

    assertPolicyEquals(policy, factory.getTokenizerCharacterPolicy());
    Assertions.assertEquals(lang, factory.getLanguageCode());
    Assertions.assertEquals(lang, model.getLanguage());
    Assertions.assertTrue(factory.isUseAlphaNumericOptimization());

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    model.serialize(out);
    ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());

    TokenizerModel fromSerialized = new TokenizerModel(in);

    factory = fromSerialized.getFactory();
    Assertions.assertNull(factory.getAbbreviationDictionary());
    Assertions.assertInstanceOf(DefaultTokenContextGenerator.class, factory.getContextGenerator());
    assertPolicyEquals(policy, factory.getTokenizerCharacterPolicy());
    Assertions.assertEquals(lang, factory.getLanguageCode());
    Assertions.assertEquals(lang, model.getLanguage());
    Assertions.assertTrue(factory.isUseAlphaNumericOptimization());
  }

  void checkCustomPatternForTokenizerME(String lang, String pattern, String sentence,
      int expectedNumTokens) throws IOException {
    Locale loc = Locale.ENGLISH;
    if ("dut".equals(lang) || "nld".equals(lang)) {
      loc = LOCALE_DUTCH;
    } else if ("deu".equals(lang)) {
      loc = Locale.GERMAN;
    } else if ("fra".equals(lang)) {
      loc = Locale.FRENCH;
    } else if ("ita".equals(lang)) {
      loc = Locale.ITALIAN;
    } else if ("pol".equals(lang)) {
      loc = LOCALE_POLISH;
    } else if ("por".equals(lang)) {
      loc = LOCALE_PORTUGUESE;
    } else if ("spa".equals(lang)) {
      loc = LOCALE_SPANISH;
    }
    TokenizerModel model = train(new TokenizerFactory(lang, loadAbbDictionary(loc), true,
        legacyPolicy(pattern)));

    TokenizerME tokenizer = new TokenizerME(model);
    String[] tokens = tokenizer.tokenize(sentence);

    Assertions.assertEquals(expectedNumTokens, tokens.length);
    String[] sentSplit = sentence
            .replace("'", " '")
            .replace(",", " ,")
            .split(" ");
    for (int i = 0; i < sentSplit.length; i++) {
      String sElement = sentSplit[i];
      if (i == sentSplit.length - 1) {
        sElement = sElement.replace(".", ""); // compensate for sentence ending
      }
      Assertions.assertEquals(sElement, tokens[i]);
    }
  }

  // For language specific patterns see: opennlp.tools.tokenize.lang.Factory

  @Test
  void testCustomPatternForTokenizerMEWithAbbreviationsDeu() throws IOException {
    String lang = "deu";
    String pattern = "^[A-Za-z0-9äéöüÄÉÖÜß]+$";
    String sentence = "Ich wähle den auf S. 183 ff. mitgeteilten " +
            "Traum von der botanischen Monographie.";
    checkCustomPatternForTokenizerME(lang, pattern, sentence, 14);
  }

  @Test // to verify OPENNLP-1555
  void testCustomPatternForTokenizerMEWithMultiDotAbbreviationsDeu() throws IOException {
    String lang = "deu";
    String pattern = "^[A-Za-z0-9äéöüÄÉÖÜß]+$";
    // Adds an extra "z.B.", the compact form of "z. B." (zum Beispiel => for example)
    String sentence = "Ich wähle z.B. den auf S. 183 ff. mitgeteilten " +
            "Traum von der botanischen Monographie.";
    checkCustomPatternForTokenizerME(lang, pattern, sentence, 15);
  }

  @Test
  void testCustomPatternForTokenizerMEWithAbbreviationsDut() throws IOException {
    String lang = "dut";
    String pattern = "^[A-Za-z0-9äöüëèéïĳÄÖÜËÉÈÏĲ]+$";
    String sentence = "Ik kies voor de droom van de botanische monografie die " +
            "op p. 183 en volgende wordt beschreven.";
    checkCustomPatternForTokenizerME(lang, pattern, sentence, 18);
  }

  @Test // to verify OPENNLP-1555
  void testCustomPatternForTokenizerMEWithMultiDotAbbreviationsDut() throws IOException {
    String lang = "dut";
    String pattern = "^[A-Za-z0-9äöüëèéïĳÄÖÜËÉÈÏĲ]+$";
    String sentence = "Ik kies voor de droom van de botanische monografie die " +
            "op p. 183 e.v. wordt beschreven.";
    checkCustomPatternForTokenizerME(lang, pattern, sentence, 17);
  }

  @Test
  void testCustomPatternForTokenizerMEWithAbbreviationsFra() throws IOException {
    String lang = "fra";
    String pattern = "^[a-zA-Z0-9àâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]+$";
    String sentence = "Je choisis le rêve de la monographie botanique communiqué à la p. 205.";
    checkCustomPatternForTokenizerME(lang, pattern, sentence, 14);
  }

  @Test
  void testCustomPatternForTokenizerMEWithAbbreviationsPol() throws IOException {
    String lang = "pol";
    String pattern = "^[A-Za-z0-9żźćńółęąśŻŹĆĄŚĘŁÓŃ]+$";
    String sentence = "W szkicu autobiograficznym pt. moje życie i psychoanaliza Freud pisze, że " +
            "jego przodkowie żyli przez wiele lat w Kolonii.";
    checkCustomPatternForTokenizerME(lang, pattern, sentence, 21);
  }

  @Test
  void testCustomPatternForTokenizerMEWithAbbreviationsPor() throws IOException {
    String lang = "por";
    String pattern = "^[0-9a-záãâàéêíóõôúüçA-ZÁÃÂÀÉÊÍÓÕÔÚÜÇ]+$";
    String sentence = "O povo pernambucano, tradicionalmente inimigo dos imperadores, " +
            "lembrava-se do tempo em que o Sr. D. Pedro de Alcantara dava-se ao luxo " +
            "de visitar o norte.";
    checkCustomPatternForTokenizerME(lang, pattern, sentence, 28);
  }

  @Test
  void testCustomPatternForTokenizerMEWithAbbreviationsSpa() throws IOException {
    String lang = "spa";
    String pattern = "^[0-9a-záéíóúüýñA-ZÁÉÍÓÚÝÑ]+$";
    String sentence = "Elegiremos el de la monografía botánica expuesto antes del " +
            "capítulo V en pág. 448 del presente volumen.";
    checkCustomPatternForTokenizerME(lang, pattern, sentence, 18);
  }

  @Test
  void testCustomPatternForTokenizerMEPor() throws IOException {
    String lang = "por";
    String pattern = "^[0-9a-záãâàéêíóõôúüçA-ZÁÃÂÀÉÊÍÓÕÔÚÜÇ]+$";
    String sentence = "Na floresta mágica a raposa dança com unicórnios felizes.";
    checkCustomPatternForTokenizerME(lang, pattern, sentence, 10);
  }

  @Test
  void testCustomPatternForTokenizerMESpa() throws IOException {
    String lang = "spa";
    String pattern = "^[0-9a-záéíóúüýñA-ZÁÉÍÓÚÝÑ]+$";
    String sentence = "En el verano los niños juegan en el parque y sus risas crean alegría.";
    checkCustomPatternForTokenizerME(lang, pattern, sentence, 15);
  }

  @Test
  void testCustomPatternForTokenizerMECat() throws IOException {
    String lang = "cat";
    String pattern = "^[0-9a-zàèéíïòóúüçA-ZÀÈÉÍÏÒÓÚÜÇ]+$";
    String sentence = "Als xiuxiuejants avets l'os blau neda amb cignes i s'ho passen bé.";
    checkCustomPatternForTokenizerME(lang, pattern, sentence, 15);
  }

  @Test
  void testCustomPatternForTokenizerMEIta() throws IOException {
    String lang = "ita";
    String pattern = "^[0-9a-zàèéìîíòóùüA-ZÀÈÉÌÎÍÒÓÙÜ]+$";
    String sentence = "Cosa fare di domenica per migliorare il tuo lunedì.";
    checkCustomPatternForTokenizerME(lang, pattern, sentence, 10);
  }

  @Test
  void testCustomPatternForTokenizerMEWithAbbreviationsIta() throws IOException {
    String lang = "ita";
    String pattern = "^[0-9a-zàèéìîíòóùüA-ZÀÈÉÌÎÍÒÓÙÜ]+$";
    String sentence = "La chiesa fu costruita fra il 1258 ed il 1308 ca. come chiesa " +
        "del convento degli Agostiniani.";
    checkCustomPatternForTokenizerME(lang, pattern, sentence, 18);
  }

  @Test
  void testContractionsIta() throws IOException {

    Dictionary dic = null;
    String lang = "ita";
    String pattern = "^[0-9a-zàèéìîíòóùüA-ZÀÈÉÌÎÍÒÓÙÜ]+$";

    TokenizerModel model = train(new TokenizerFactory(lang, dic, true,
        legacyPolicy(pattern)));

    TokenizerME tokenizer = new TokenizerME(model);
    String sentence = "La contrazione di \"dove è\" è \"dov'è\".";
    String[] tokens = tokenizer.tokenize(sentence);

    Assertions.assertEquals(11, tokens.length);
    String[] sentSplit = sentence.replaceAll("\\.", " .")
        .replaceAll("'", " '").replaceAll("([^ ])\"", "$1 \"").split(" ");
    for (int i = 0; i < sentSplit.length; i++) {
      Assertions.assertEquals(sentSplit[i], tokens[i]);
    }
  }

  @Test
  void testContractionsEng() throws IOException {

    Dictionary dic = null;
    String lang = "eng";
    String pattern = "^[A-Za-z0-9]+$";

    TokenizerModel model = train(new TokenizerFactory(lang, dic, true,
        legacyPolicy(pattern)));

    TokenizerME tokenizer = new TokenizerME(model);
    String sentence = "The cat wasn't in the house and the dog wasn't either.";
    String[] tokens = tokenizer.tokenize(sentence);

    Assertions.assertEquals(14, tokens.length);
    String[] sentSplit = sentence.replaceAll("\\.", " .")
        .replaceAll("'", " '").split(" ");
    for (int i = 0; i < sentSplit.length; i++) {
      Assertions.assertEquals(sentSplit[i], tokens[i]);
    }
  }

  @Test
  void testDummyFactory() throws IOException {

    Dictionary dic = loadAbbDictionary(Locale.ENGLISH);
    final String lang = "eng";
    String pattern = "^[0-9A-Za-z]+$";

    TokenizerCharacterPolicy policy = TokenizerCharacterPolicy.ascii();
    TokenizerModel model = train(new DummyTokenizerFactory(lang, dic, true, policy));

    TokenizerFactory factory = model.getFactory();
    Assertions.assertInstanceOf(DummyDictionary.class, factory.getAbbreviationDictionary());
    Assertions.assertInstanceOf(DummyContextGenerator.class, factory.getContextGenerator());
    assertPolicyEquals(policy, factory.getTokenizerCharacterPolicy());
    Assertions.assertEquals(lang, factory.getLanguageCode());
    Assertions.assertEquals(lang, model.getLanguage());
    Assertions.assertTrue(factory.isUseAlphaNumericOptimization());

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    model.serialize(out);
    ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());

    TokenizerModel fromSerialized = new TokenizerModel(in);

    factory = fromSerialized.getFactory();
    Assertions.assertInstanceOf(DummyDictionary.class, factory.getAbbreviationDictionary());
    Assertions.assertInstanceOf(DummyContextGenerator.class, factory.getContextGenerator());
    assertPolicyEquals(policy, factory.getTokenizerCharacterPolicy());
    Assertions.assertEquals(lang, factory.getLanguageCode());
    Assertions.assertEquals(lang, model.getLanguage());
    Assertions.assertTrue(factory.isUseAlphaNumericOptimization());
  }

  @Test
  void testCreateDummyFactory() throws IOException {
    Dictionary dic = loadAbbDictionary(Locale.ENGLISH);
    final String lang = "eng";
    String pattern = "^[0-9A-Za-z]+$";

    TokenizerCharacterPolicy policy = TokenizerCharacterPolicy.ascii();
    TokenizerFactory factory = TokenizerFactory.create(
        DummyTokenizerFactory.class.getCanonicalName(), lang, dic, true,
        policy);

    Assertions.assertInstanceOf(DummyDictionary.class, factory.getAbbreviationDictionary());
    Assertions.assertInstanceOf(DummyContextGenerator.class, factory.getContextGenerator());
    assertPolicyEquals(policy, factory.getTokenizerCharacterPolicy());
    Assertions.assertEquals(lang, factory.getLanguageCode());
    Assertions.assertTrue(factory.isUseAlphaNumericOptimization());
  }

  private static final class TestTokenizerFactory extends TokenizerFactory {

    private void load(ArtifactProvider provider) {
      init(provider);
    }
  }

  private static final class ConstructorDispatchFactory extends TokenizerFactory {

    private ConstructorDispatchFactory() {
      super("eng", null, true, null);
    }

    @Override
    protected void init(String languageCode, Dictionary abbreviationDictionary,
        boolean useAlphaNumericOptimization, TokenizerCharacterPolicy alphanumericPolicy) {
      throw new AssertionError("Constructor dispatched to overridable init");
    }
  }

  private record MapArtifactProvider(Map<String, String> manifest) implements ArtifactProvider {

    @Override
    public <T> T getArtifact(String key) {
      return null;
    }

    @Override
    public String getManifestProperty(String key) {
      return manifest.get(key);
    }

    @Override
    public String getLanguage() {
      return "x-test";
    }

    @Override
    public boolean isLoadedFromSerialized() {
      return true;
    }
  }
}
