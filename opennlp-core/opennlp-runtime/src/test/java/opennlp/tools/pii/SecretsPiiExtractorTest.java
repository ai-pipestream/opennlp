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

package opennlp.tools.pii;

import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

public class SecretsPiiExtractorTest {

  /** Counts character reads so scanner complexity can be asserted without wall-clock timing. */
  private static final class CountingCharSequence implements CharSequence {

    private final String value;
    private int reads;

    private CountingCharSequence(String value) {
      this.value = value;
    }

    @Override
    public int length() {
      return value.length();
    }

    @Override
    public char charAt(int index) {
      reads++;
      return value.charAt(index);
    }

    @Override
    public CharSequence subSequence(int start, int end) {
      return value.subSequence(start, end);
    }
  }

  /** The header of {@code {"alg":"HS256","typ":"JWT"}} in base64url. */
  private static final String HS256_HEADER = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9";

  /** The header of {@code {"alg":"RS256","kid":"abc123"}} in base64url. */
  private static final String RS256_HEADER = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImFiYzEyMyJ9";

  /** The header of {@code {"typ":"JWT"}}, which carries no algorithm. */
  private static final String NO_ALGORITHM_HEADER = "eyJ0eXAiOiJKV1QifQ";

  /** The header of {@code {"notalg":"HS256"}}, whose member only ends in those letters. */
  private static final String NOTALG_HEADER = "eyJub3RhbGciOiJIUzI1NiJ9";

  /** The header of {@code {"algorithm":"HS256"}}, whose member only begins with them. */
  private static final String ALGORITHM_HEADER = "eyJhbGdvcml0aG0iOiJIUzI1NiJ9";

  /** The header of <code>{"alg" : "HS256"}</code>, spaced out as JSON permits. */
  private static final String SPACED_HEADER = "eyJhbGciIDogIkhTMjU2In0";

  private static final String PAYLOAD =
      "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ";

  private static final String SIGNATURE = "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";

  private final SecretsPiiExtractor extractor = new SecretsPiiExtractor();

  @ParameterizedTest
  @ValueSource(strings = {"eyJ_", "ghp_"})
  void testPrefixHeavyNearMissesAreScannedLinearly(String prefix) {
    final CountingCharSequence text = new CountingCharSequence(prefix.repeat(1024));

    Assertions.assertTrue(extractor.extract(text).isEmpty());
    Assertions.assertTrue(text.reads <= text.length() * 20,
        () -> "read " + text.reads + " characters from an input of " + text.length());
  }

  @ParameterizedTest
  @ValueSource(strings = {
      "AKIAIOSFODNN7EXAMPLE",
      "ASIAIOSFODNN7EXAMPLE",
      "AKIA1234567890ABCDEF",
      "ASIAZZZZZZZZZZZZZZZZ",
      "AKIA0000000000000000"})
  void testAcceptsAwsAccessKeys(String text) {
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size(), text);
    Assertions.assertEquals(PiiMention.TYPE_AWS_ACCESS_KEY, mentions.get(0).type());
    Assertions.assertEquals(text, mentions.get(0).normalized());
    Assertions.assertEquals(0, mentions.get(0).span().getStart());
    Assertions.assertEquals(text.length(), mentions.get(0).span().getEnd());
  }

  @ParameterizedTest
  @ValueSource(strings = {
      "AKIAIOSFODNN7EXAMPL",
      "AKIAIOSFODNN7EXAMPLEX",
      "akiaiosfodnn7example",
      "AKIAiosfodnn7EXAMPLE",
      "AIDAIOSFODNN7EXAMPLE",
      "AROAIOSFODNN7EXAMPLE",
      "ANPAIOSFODNN7EXAMPLE",
      "APKAIOSFODNN7EXAMPLE",
      "XAKIAIOSFODNN7EXAMPLE",
      "AKIA-IOSFODNN7EXAMPL",
      "AKIA_IOSFODNN7EXAMPL"})
  void testRejectsAwsAccessKeyNearMisses(String text) {
    Assertions.assertTrue(
        extractor.extract(text).stream()
            .noneMatch(m -> PiiMention.TYPE_AWS_ACCESS_KEY.equals(m.type())), text);
  }

  @Test
  void testAwsAccessKeySpanInSentence() {
    final String text = "Rotate AKIAIOSFODNN7EXAMPLE right away.";
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size());
    Assertions.assertEquals("AKIAIOSFODNN7EXAMPLE", text.substring(
        mentions.get(0).span().getStart(), mentions.get(0).span().getEnd()));
  }

  @ParameterizedTest
  @ValueSource(strings = {
      "ghp_1234567890abcdefghijklmnopqrstuvwxyz",
      "gho_1234567890abcdefghijklmnopqrstuvwxyz",
      "ghu_1234567890abcdefghijklmnopqrstuvwxyz",
      "ghs_1234567890abcdefghijklmnopqrstuvwxyz",
      "ghr_1234567890abcdefghijklmnopqrstuvwxyz",
      "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"})
  void testAcceptsGithubTokens(String text) {
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size(), text);
    Assertions.assertEquals(PiiMention.TYPE_GITHUB_TOKEN, mentions.get(0).type());
    Assertions.assertEquals(text, mentions.get(0).normalized());
  }

  @Test
  void testAcceptsFineGrainedGithubToken() {
    final String token = "github_pat_11ABCDEFG0abcdefghijkl_"
        + "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVW";
    // The documented form is the 11-character prefix and 82 body characters.
    Assertions.assertEquals(93, token.length());
    final List<PiiMention> mentions = extractor.extract("token: " + token);

    Assertions.assertEquals(1, mentions.size());
    Assertions.assertEquals(PiiMention.TYPE_GITHUB_TOKEN, mentions.get(0).type());
    Assertions.assertEquals(token, mentions.get(0).normalized());
  }

  /**
   * GitHub explicitly reserves the right to change token lengths and asks scanners to
   * accept the documented alphabet up to 255 characters.
   */
  @Test
  void testAcceptsVariableLengthGithubTokens() {
    final String legacy = "ghp_" + "a".repeat(36) + "_" + "B".repeat(24);
    final String fineGrained = "github_pat_" + "a".repeat(82) + "_" + "7".repeat(40);

    Assertions.assertEquals(legacy,
        extractor.extract(legacy).get(0).normalized());
    Assertions.assertEquals(fineGrained,
        extractor.extract(fineGrained).get(0).normalized());
  }

  @Test
  void testGithubTokenMaximumLengthBoundary() {
    final String accepted = "ghp_" + "a".repeat(251);
    final String rejected = "ghp_" + "a".repeat(252);

    Assertions.assertEquals(255, accepted.length());
    Assertions.assertEquals(accepted, extractor.extract(accepted).get(0).normalized());
    Assertions.assertTrue(extractor.extract(rejected).isEmpty());
  }

  @ParameterizedTest
  @ValueSource(strings = {
      "ghp_1234567890abcdefghijklmnopqrstuvwxy",
      "ghx_1234567890abcdefghijklmnopqrstuvwxyz",
      "GHP_1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",
      "ghp1234567890abcdefghijklmnopqrstuvwxyz",
      "ghp_1234567890abcdefghijklmnopqrstuvwx-z",
      "xghp_1234567890abcdefghijklmnopqrstuvwxyz",
      "github_pat_tooshort"})
  void testRejectsGithubTokenNearMisses(String text) {
    Assertions.assertTrue(
        extractor.extract(text).stream()
            .noneMatch(m -> PiiMention.TYPE_GITHUB_TOKEN.equals(m.type())), text);
  }

  @Test
  void testAcceptsJsonWebToken() {
    final String token = HS256_HEADER + "." + PAYLOAD + "." + SIGNATURE;
    final List<PiiMention> mentions = extractor.extract("Authorization: Bearer " + token);

    Assertions.assertEquals(1, mentions.size());
    Assertions.assertEquals(PiiMention.TYPE_JWT, mentions.get(0).type());
    Assertions.assertEquals(token, mentions.get(0).normalized());
  }

  /**
   * Verifies that requiring the colon of the {@code alg} member does not reject a header
   * whose JSON is spaced out, which the grammar allows.
   */
  @ParameterizedTest
  @ValueSource(strings = {HS256_HEADER, RS256_HEADER, SPACED_HEADER})
  void testAcceptsJsonWebTokensOfSeveralAlgorithms(String header) {
    final String token = header + "." + PAYLOAD + "." + SIGNATURE;

    final List<PiiMention> mentions = extractor.extract(token);

    Assertions.assertEquals(1, mentions.size(), header);
    Assertions.assertEquals(PiiMention.TYPE_JWT, mentions.get(0).type());
    Assertions.assertEquals(0, mentions.get(0).span().getStart());
    Assertions.assertEquals(token.length(), mentions.get(0).span().getEnd());
  }

  /**
   * Verifies the near misses that a plain three-segment test would accept: a header
   * without the algorithm parameter that RFC 7515 requires, a header whose member name
   * merely contains those three letters, a header that is not base64url of a JSON object,
   * and a token missing a segment.
   */
  @ParameterizedTest
  @ValueSource(strings = {
      NO_ALGORITHM_HEADER + ".eyJzdWIiOiIxIn0.abcdefgh",
      NOTALG_HEADER + "." + PAYLOAD + "." + SIGNATURE,
      ALGORITHM_HEADER + "." + PAYLOAD + "." + SIGNATURE,
      "notbase64url." + PAYLOAD + "." + SIGNATURE,
      HS256_HEADER + "." + PAYLOAD,
      HS256_HEADER + "." + PAYLOAD + ".",
      HS256_HEADER + ".." + SIGNATURE,
      "eyJ." + PAYLOAD + "." + SIGNATURE,
      "x" + HS256_HEADER + "." + PAYLOAD + "." + SIGNATURE})
  void testRejectsJsonWebTokenNearMisses(String text) {
    Assertions.assertTrue(
        extractor.extract(text).stream()
            .noneMatch(m -> PiiMention.TYPE_JWT.equals(m.type())), text);
  }

  @ParameterizedTest
  @CsvSource({
      "https://user:secret@example.com/path, user:secret",
      "http://admin:s3cr3t@10.0.0.1:8080/, admin:s3cr3t",
      "ftp://jane.doe:pw%21@files.example.org, jane.doe:pw%21",
      "postgres://app:hunter2@db.internal:5432/main, app:hunter2",
      "redis://default:abc-123_x@cache.example.net, default:abc-123_x",
      "amqp://guest:guest@broker/, guest:guest"
  })
  void testAcceptsUrlCredentials(String text, String credential) {
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size(), text);
    Assertions.assertEquals(PiiMention.TYPE_URL_CREDENTIAL, mentions.get(0).type());
    Assertions.assertEquals(credential, mentions.get(0).normalized());
    Assertions.assertEquals(credential, text.substring(
        mentions.get(0).span().getStart(), mentions.get(0).span().getEnd()));
  }

  @ParameterizedTest
  @ValueSource(strings = {
      "https://example.com/path",
      "https://user@example.com",
      "https://:secret@example.com",
      "https://user:@example.com",
      "https://example.com/a?b=c@d",
      "mailto:jane@example.com",
      "://user:secret@example.com",
      "user:secret@example.com"})
  void testRejectsUrlCredentialNearMisses(String text) {
    Assertions.assertTrue(
        extractor.extract(text).stream()
            .noneMatch(m -> PiiMention.TYPE_URL_CREDENTIAL.equals(m.type())), text);
  }

  @Test
  void testUrlCredentialContainingATokenIsReportedOnce() {
    final String text = "https://oauth2:ghp_1234567890abcdefghijklmnopqrstuvwxyz@github.com/x.git";
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(1, mentions.size());
    Assertions.assertEquals(PiiMention.TYPE_URL_CREDENTIAL, mentions.get(0).type());
    Assertions.assertEquals("oauth2:ghp_1234567890abcdefghijklmnopqrstuvwxyz",
        mentions.get(0).normalized());
  }

  @Test
  void testFindsSeveralSecretsInOneText() {
    final String text = "key AKIAIOSFODNN7EXAMPLE token ghp_1234567890abcdefghijklmnopqrstuvwxyz "
        + "jwt " + HS256_HEADER + "." + PAYLOAD + "." + SIGNATURE
        + " url https://u:p@example.com/";
    final List<PiiMention> mentions = extractor.extract(text);

    Assertions.assertEquals(List.of(PiiMention.TYPE_AWS_ACCESS_KEY,
            PiiMention.TYPE_GITHUB_TOKEN, PiiMention.TYPE_JWT, PiiMention.TYPE_URL_CREDENTIAL),
        mentions.stream().map(PiiMention::type).toList());
    int lastEnd = 0;
    for (final PiiMention mention : mentions) {
      Assertions.assertTrue(mention.span().getStart() >= lastEnd);
      lastEnd = mention.span().getEnd();
    }
  }

  @Test
  void testTwoTokensSideBySideAreBothFound() {
    final String text = "ghp_1234567890abcdefghijklmnopqrstuvwxyz "
        + "ghs_abcdefghijklmnopqrstuvwxyz1234567890";
    Assertions.assertEquals(2, extractor.extract(text).size());
  }

  @Test
  void testTypeSubsetLimitsWhatIsReported() {
    final String text = "AKIAIOSFODNN7EXAMPLE and https://u:p@example.com/";

    Assertions.assertEquals(List.of(PiiMention.TYPE_AWS_ACCESS_KEY),
        new SecretsPiiExtractor(Set.of(PiiMention.TYPE_AWS_ACCESS_KEY)).extract(text)
            .stream().map(PiiMention::type).toList());
    Assertions.assertEquals(List.of(PiiMention.TYPE_URL_CREDENTIAL),
        new SecretsPiiExtractor(Set.of(PiiMention.TYPE_URL_CREDENTIAL)).extract(text)
            .stream().map(PiiMention::type).toList());
  }

  @ParameterizedTest
  @ValueSource(strings = {
      "nothing to see here",
      "AKIA",
      "ghp_",
      "eyJ",
      "https://",
      "",
      "The word algorithm contains alg but is no token"})
  void testTextWithoutASecretYieldsNoMention(String text) {
    Assertions.assertTrue(extractor.extract(text).isEmpty(), text);
  }

  @Test
  void testRejectsUnrecognizedTypeAndMissingArguments() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new SecretsPiiExtractor(Set.of(PiiMention.TYPE_EMAIL)));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new SecretsPiiExtractor(Set.of()));
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> new SecretsPiiExtractor(null));
    Assertions.assertThrows(IllegalArgumentException.class, () -> extractor.extract(null));
  }
}
