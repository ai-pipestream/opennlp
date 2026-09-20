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

package opennlp.tools.tokenize.lang;

import java.util.Map;
import java.util.Set;

import opennlp.tools.tokenize.DefaultTokenContextGenerator;
import opennlp.tools.tokenize.TokenContextGenerator;
import opennlp.tools.tokenize.TokenizerCharacterPolicy;
import opennlp.tools.util.normalizer.CodePointSet;

public class Factory {

  private static final String ASCII = "^[A-Za-z0-9]+$";
  private static final Map<String, String> LEGACY = Map.ofEntries(
      Map.entry("^[0-9a-záãâàéêíóõôúüçA-ZÁÃÂÀÉÊÍÓÕÔÚÜÇ]+$",
          "abcdefghijklmnopqrstuvwxyzáãâàéêíóõôúüç"
              + "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÃÂÀÉÊÍÓÕÔÚÜÇ"),
      Map.entry("^[a-zA-Z0-9àâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]+$",
          "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZàâäèéêëîïôœ"
              + "ùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ"),
      Map.entry("^[A-Za-z0-9äöüëèéïĳÄÖÜËÉÈÏĲ]+$",
          "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüëèéïĳÄÖÜËÉÈÏĲ"),
      Map.entry("^[A-Za-z0-9äéöüÄÉÖÜß]+$", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäéöüÄÉÖÜß"),
      Map.entry("^[A-Za-z0-9żźćńółęąśŻŹĆĄŚĘŁÓŃ]+$",
          "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzżźćńółęąśŻŹĆĄŚĘŁÓŃ"),
      Map.entry("^[0-9a-zàèéìîíòóùüA-ZÀÈÉÌÎÍÒÓÙÜ]+$",
          "abcdefghijklmnopqrstuvwxyzàèéìîíòóùüABCDEFGHIJKLMNOPQRSTUVWXYZÀÈÉÌÎÍÒÓÙÜ"),
      Map.entry("^[0-9a-záéíóúüýñA-ZÁÉÍÓÚÝÑ]+$",
          "abcdefghijklmnopqrstuvwxyzáéíóúüýñABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÝÑ"),
      Map.entry("^[0-9a-zàèéíïòóúüçA-ZÀÈÉÍÏÒÓÚÜÇ]+$",
          "abcdefghijklmnopqrstuvwxyzàèéíïòóúüçABCDEFGHIJKLMNOPQRSTUVWXYZÀÈÉÍÏÒÓÚÜÇ"));

  /** Imports only the exact historical expressions shipped by OpenNLP. */
  public TokenizerCharacterPolicy importLegacyAlphanumeric(String expression) {
    if (ASCII.equals(expression)) {
      return TokenizerCharacterPolicy.ascii();
    }
    String letters = LEGACY.get(expression);
    return letters == null ? null : TokenizerCharacterPolicy.of(
        CodePointSet.of(letters.codePoints().toArray()), CodePointSet.ofRange('0', '9'),
        CodePointSet.of());
  }

  /** Resolves the character policy for a newly trained model. */
  public TokenizerCharacterPolicy getAlphanumericPolicy(String languageCode) {
    if (isLatinLanguage(languageCode)) {
      return TokenizerCharacterPolicy.latinUnicode17();
    }
    return TokenizerCharacterPolicy.ascii();
  }

  private static boolean isLatinLanguage(String languageCode) {
    return "en".equals(languageCode) || "eng".equals(languageCode)
        || "es".equals(languageCode) || "spa".equals(languageCode)
        || "it".equals(languageCode) || "ita".equals(languageCode)
        || "pt".equals(languageCode) || "por".equals(languageCode)
        || "ca".equals(languageCode) || "cat".equals(languageCode)
        || "pl".equals(languageCode) || "pol".equals(languageCode)
        || "de".equals(languageCode) || "deu".equals(languageCode)
        || "ger".equals(languageCode) || "fr".equals(languageCode)
        || "fre".equals(languageCode) || "fra".equals(languageCode)
        || "nl".equals(languageCode) || "nld".equals(languageCode)
        || "dut".equals(languageCode);
  }

  public TokenContextGenerator createTokenContextGenerator(
      String languageCode, Set<String> abbreviations) {
    return new DefaultTokenContextGenerator(abbreviations);
  }
}
