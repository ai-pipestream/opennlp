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

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.tokenize.lang.Factory;
import opennlp.tools.util.BaseToolFactory;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.ext.ExtensionLoader;
import opennlp.tools.util.normalizer.CodePointSet;

/**
 * The factory that provides {@link Tokenizer} default implementation and
 * resources. Users can extend this class if their application requires
 * overriding the {@link TokenContextGenerator}, {@link Dictionary} etc.
 */
public class TokenizerFactory extends BaseToolFactory {

  private String languageCode;
  private Dictionary abbreviationDictionary;
  private Boolean useAlphaNumericOptimization = false;
  private TokenizerCharacterPolicy alphanumericPolicy;

  private static final String ABBREVIATIONS_ENTRY_NAME = "abbreviations.dictionary";
  private static final String USE_ALPHA_NUMERIC_OPTIMIZATION = "useAlphaNumericOptimization";
  private static final String ALPHA_NUMERIC_PATTERN = "alphaNumericPattern";
  private static final String POLICY_VERSION = "tokenizerCharacterPolicyVersion";
  private static final String POLICY_LETTERS = "tokenizerCharacterPolicyLetters";
  private static final String POLICY_DIGITS = "tokenizerCharacterPolicyDigits";
  private static final String POLICY_MARKS = "tokenizerCharacterPolicyMarks";

  /**
   * Instantiates a {@link TokenizerFactory} that provides the default implementation
   * of the resources.
   */
  public TokenizerFactory() {
  }
  
  /**
   * Instantiates a {@link TokenizerFactory}. Use this constructor to
   * programmatically create a factory.
   *
   * @param languageCode The ISO language code to be used for this factory.
   * @param abbreviationDictionary The {@link Dictionary} which holds abbreviations.
   * @param useAlphaNumericOptimization Whether alphanumerics are skipped, or not.
   * @param alphanumericPolicy The explicit character policy, or {@code null} to resolve the
   *                           built-in policy for {@code languageCode}.
   */
  public TokenizerFactory(String languageCode, Dictionary abbreviationDictionary,
                          boolean useAlphaNumericOptimization,
                          TokenizerCharacterPolicy alphanumericPolicy) {
    configure(languageCode, abbreviationDictionary,
        useAlphaNumericOptimization, alphanumericPolicy);
  }

  /**
   * @param languageCode The ISO language code to be used for this factory.
   * @param abbreviationDictionary The {@link Dictionary} which holds abbreviations.
   * @param useAlphaNumericOptimization Whether alphanumerics are skipped, or not.
   * @param alphanumericPolicy The explicit character policy, or {@code null} to resolve the
   *                           built-in policy for {@code languageCode}.
   */
  protected void init(String languageCode, Dictionary abbreviationDictionary,
      boolean useAlphaNumericOptimization, TokenizerCharacterPolicy alphanumericPolicy) {
    configure(languageCode, abbreviationDictionary,
        useAlphaNumericOptimization, alphanumericPolicy);
  }

  private void configure(String languageCode, Dictionary abbreviationDictionary,
      boolean useAlphaNumericOptimization, TokenizerCharacterPolicy alphanumericPolicy) {
    if (alphanumericPolicy == null) {
      alphanumericPolicy = new Factory().getAlphanumericPolicy(languageCode);
    }
    this.languageCode = languageCode;
    this.useAlphaNumericOptimization = useAlphaNumericOptimization;
    this.alphanumericPolicy = alphanumericPolicy;
    this.abbreviationDictionary = abbreviationDictionary;
  }

  @Override
  public void validateArtifactMap() throws InvalidFormatException {
    if (this.artifactProvider.getManifestProperty(USE_ALPHA_NUMERIC_OPTIMIZATION) == null)
      throw new InvalidFormatException(USE_ALPHA_NUMERIC_OPTIMIZATION
          + " is a mandatory property!");

    Object abbreviationsEntry = this.artifactProvider.getArtifact(ABBREVIATIONS_ENTRY_NAME);

    if (abbreviationsEntry != null && !(abbreviationsEntry instanceof Dictionary)) {
      throw new InvalidFormatException("Abbreviations dictionary '" + abbreviationsEntry +
              "' has wrong type, needs to be of type Dictionary!");
    }
    try {
      isUseAlphaNumericOptimization();
      getTokenizerCharacterPolicy();
    } catch (IllegalArgumentException | IllegalStateException e) {
      throw new InvalidFormatException("Invalid tokenizer character policy", e);
    }
  }

  @Override
  public Map<String, Object> createArtifactMap() {
    Map<String, Object> artifactMap = super.createArtifactMap();

    // Abbreviations are optional
    if (abbreviationDictionary != null) {
      artifactMap.put(ABBREVIATIONS_ENTRY_NAME, abbreviationDictionary);
    }

    return artifactMap;
  }

  @Override
  public Map<String, String> createManifestEntries() {
    Map<String, String> manifestEntries = super.createManifestEntries();

    manifestEntries.put(USE_ALPHA_NUMERIC_OPTIMIZATION,
        Boolean.toString(isUseAlphaNumericOptimization()));

    TokenizerCharacterPolicy policy = getTokenizerCharacterPolicy();
    manifestEntries.put(POLICY_VERSION, "1");
    manifestEntries.put(POLICY_LETTERS, encode(policy.getLetters()));
    manifestEntries.put(POLICY_DIGITS, encode(policy.getDigits()));
    manifestEntries.put(POLICY_MARKS, encode(policy.getMarks()));

    return manifestEntries;
  }

  /**
   * Factory method the framework uses instantiate a new {@link TokenizerFactory}.
   *
   * @param subclassName The name of the class implementing the {@link TokenizerFactory}.
   * @param languageCode The ISO language code the {@link Tokenizer} should use.
   * @param abbreviationDictionary An optional {@link Dictionary} containing abbreviations,
   *                               or {@code null} if not present.
   * @param useAlphaNumericOptimization Whether the alphanumeric optimization is be enabled or not.
   * @param alphanumericPolicy The policy the alphanumeric optimization should use, or
   *                           {@code null} to resolve the built-in language policy.
   *
   * @return A valid {@link TokenizerFactory} instance.
   *
   * @throws InvalidFormatException Thrown if one of the input parameters doesn't comply the expected format.
   */
  public static TokenizerFactory create(String subclassName, String languageCode,
                                        Dictionary abbreviationDictionary,
                                        boolean useAlphaNumericOptimization,
                                        TokenizerCharacterPolicy alphanumericPolicy)
      throws InvalidFormatException {
    if (subclassName == null) {
      // will create the default factory
      return new TokenizerFactory(languageCode, abbreviationDictionary,
          useAlphaNumericOptimization, alphanumericPolicy);
    }
    try {
      TokenizerFactory theFactory = ExtensionLoader.instantiateExtension(
          TokenizerFactory.class, subclassName);
      theFactory.init(languageCode, abbreviationDictionary,
          useAlphaNumericOptimization, alphanumericPolicy);
      return theFactory;
    } catch (Exception e) {
      String msg = "Could not instantiate the " + subclassName
          + ". The initialization throw an exception.";
      throw new InvalidFormatException(msg, e);
    }
  }

  /**
   * @return Retrieves the resolved tokenizer character policy.
   */
  public TokenizerCharacterPolicy getTokenizerCharacterPolicy() {
    if (this.alphanumericPolicy == null) {
      if (this.artifactProvider != null) {
        String version = this.artifactProvider.getManifestProperty(POLICY_VERSION);
        if (version != null) {
          if (!"1".equals(version)) {
            throw new IllegalArgumentException("Unsupported tokenizer character policy version: "
                + version);
          }
          String letters = requiredPolicyEntry(POLICY_LETTERS);
          String digits = requiredPolicyEntry(POLICY_DIGITS);
          String marks = requiredPolicyEntry(POLICY_MARKS);
          this.alphanumericPolicy = TokenizerCharacterPolicy.of(
              decode(POLICY_LETTERS, letters), decode(POLICY_DIGITS, digits),
              decode(POLICY_MARKS, marks));
        } else {
          if (this.artifactProvider.getManifestProperty(POLICY_LETTERS) != null
              || this.artifactProvider.getManifestProperty(POLICY_DIGITS) != null
              || this.artifactProvider.getManifestProperty(POLICY_MARKS) != null) {
            throw new IllegalArgumentException("Tokenizer character policy entries require "
                + POLICY_VERSION);
          }
          this.alphanumericPolicy = importLegacyPolicy(
              this.artifactProvider.getManifestProperty(ALPHA_NUMERIC_PATTERN));
        }
      }
    }
    if (this.alphanumericPolicy == null) {
      throw new IllegalStateException("No tokenizer character policy was configured");
    }
    return this.alphanumericPolicy;
  }

  private String requiredPolicyEntry(String name) {
    String value = artifactProvider.getManifestProperty(name);
    if (value == null) {
      throw new IllegalArgumentException("Missing tokenizer character policy entry: " + name);
    }
    return value;
  }

  private static String encode(CodePointSet set) {
    StringBuilder encoded = new StringBuilder();
    for (int codePoint : set.toArray()) {
      if (!encoded.isEmpty()) {
        encoded.append(',');
      }
      encoded.append(Integer.toHexString(codePoint));
    }
    return encoded.toString();
  }

  private static CodePointSet decode(String name, String encoded) {
    if (encoded.isEmpty()) {
      return CodePointSet.of();
    }
    int count = 1;
    for (int i = 0; i < encoded.length(); i++) {
      if (encoded.charAt(i) == ',') {
        count++;
      }
    }
    int[] values = new int[count];
    int index = 0;
    int start = 0;
    for (int i = 0; i <= encoded.length(); i++) {
      if (i == encoded.length() || encoded.charAt(i) == ',') {
        if (i == start) {
          throw new IllegalArgumentException("Malformed tokenizer character policy entry: " + name);
        }
        try {
          values[index++] = Integer.parseInt(encoded.substring(start, i), 16);
        } catch (NumberFormatException e) {
          throw new IllegalArgumentException(
              "Malformed tokenizer character policy entry: " + name, e);
        }
        start = i + 1;
      }
    }
    return CodePointSet.of(values);
  }

  private static TokenizerCharacterPolicy importLegacyPolicy(String expression) {
    if (expression == null || "^[A-Za-z0-9]+$".equals(expression)) {
      return TokenizerCharacterPolicy.ascii();
    }
    TokenizerCharacterPolicy imported = new Factory().importLegacyAlphanumeric(expression);
    if (imported == null) {
      throw new IllegalArgumentException("Unsupported legacy alphaNumericPattern. Supply explicit "
          + "TokenizerCharacterPolicy sets and retrain or repackage the model.");
    }
    return imported;
  }

  /**
   * @return {@code true} if the alphanumeric optimization is enabled, otherwise {@code false}.
   */
  public boolean isUseAlphaNumericOptimization() {
    if (artifactProvider != null) {
      String value = this.artifactProvider.getManifestProperty(USE_ALPHA_NUMERIC_OPTIMIZATION);
      if (!"true".equals(value) && !"false".equals(value)) {
        throw new IllegalArgumentException("Invalid " + USE_ALPHA_NUMERIC_OPTIMIZATION
            + " value: " + value);
      }
      this.useAlphaNumericOptimization = Boolean.valueOf(value);
    }
    return this.useAlphaNumericOptimization;
  }

  /**
   * @return The abbreviation {@link Dictionary} or {@code null} if none is active.
   */
  public Dictionary getAbbreviationDictionary() {
    if (this.abbreviationDictionary == null && artifactProvider != null) {
      this.abbreviationDictionary = this.artifactProvider.getArtifact(ABBREVIATIONS_ENTRY_NAME);
    }
    return this.abbreviationDictionary;
  }

  /**
   * @return Retrieves the ISO language code in use.
   */
  public String getLanguageCode() {
    if (this.languageCode == null && this.artifactProvider != null) {
      this.languageCode = this.artifactProvider.getLanguage();
    }
    return this.languageCode;
  }

  /**
   * @return Retrieves a {@link TokenContextGenerator} instance.
   */
  public TokenContextGenerator getContextGenerator() {
    Factory f = new Factory();
    Set<String> abbs;
    Dictionary abbDict = getAbbreviationDictionary();
    if (abbDict != null) {
      abbs = abbDict.asStringSet();
    } else {
      abbs = Collections.emptySet();
    }
    return f.createTokenContextGenerator(getLanguageCode(), abbs);
  }
}
