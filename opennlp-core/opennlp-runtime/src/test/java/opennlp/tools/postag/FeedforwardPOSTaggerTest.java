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

package opennlp.tools.postag;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import opennlp.tools.parser.AbstractBottomUpParser;
import opennlp.tools.parser.HeadRules;
import opennlp.tools.parser.Parse;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Sequence;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests the pure-Java neural tagging tier end to end: training on a tiny corpus must
 * let the greedy feedforward tagger reproduce the training sentences, and a model must
 * survive the serialization round trip with identical behavior.
 */
public class FeedforwardPOSTaggerTest {

  private static FeedforwardPOSModel model;
  private static FeedforwardPOSTagger tagger;

  private static List<POSSample> corpus() {
    final List<POSSample> distinct = List.of(
        new POSSample(new String[] {"the", "dog", "barks"},
            new String[] {"DT", "NN", "VBZ"}),
        new POSSample(new String[] {"dogs", "bark"}, new String[] {"NNS", "VBP"}),
        new POSSample(new String[] {"she", "eats", "fish"},
            new String[] {"PRP", "VBZ", "NN"}));
    final List<POSSample> corpus = new ArrayList<>();
    for (int i = 0; i < 40; i++) {
      corpus.addAll(distinct);
    }
    return corpus;
  }

  @BeforeAll
  static void trainTagger() throws IOException {
    // dropout off so the tiny network memorizes deterministically
    final FeedforwardPOSTrainer.Settings settings = new FeedforwardPOSTrainer.Settings(
        16, 32, 80, 32, 0.05, 0.0, 0.0, 1, 1, 17L);
    model = FeedforwardPOSTrainer.train(
        ObjectStreamUtils.createObjectStream(corpus()), settings);
    tagger = new FeedforwardPOSTagger(model);
  }

  @Test
  void testMemorizesTrainingSentences() {
    assertArrayEquals(new String[] {"DT", "NN", "VBZ"},
        tagger.tag(new String[] {"the", "dog", "barks"}));
    assertArrayEquals(new String[] {"PRP", "VBZ", "NN"},
        tagger.tag(new String[] {"she", "eats", "fish"}));
  }

  @Test
  void testUnknownWordsStillGetTagsFromTheInventory() {
    final String[] assigned = tagger.tag(new String[] {"the", "cat", "sleeps"});
    assertEquals(3, assigned.length);
    final List<String> inventory = List.of(model.tags());
    for (final String tag : assigned) {
      assertEquals(true, inventory.contains(tag));
    }
  }

  @Test
  void testModelRoundTripThroughSerialization() throws IOException {
    final ByteArrayOutputStream out = new ByteArrayOutputStream();
    model.serialize(out);
    final FeedforwardPOSModel reloaded =
        FeedforwardPOSModel.load(new ByteArrayInputStream(out.toByteArray()));
    assertArrayEquals(tagger.tag(new String[] {"the", "dog", "barks"}),
        new FeedforwardPOSTagger(reloaded).tag(new String[] {"the", "dog", "barks"}));
  }

  @Test
  void testCorruptModelFailsLoud() {
    assertThrows(IOException.class, () -> FeedforwardPOSModel.load(
        new ByteArrayInputStream("not a model".getBytes())));
  }

  @Test
  void testShapesAndSuffixes() {
    assertEquals("*cap*", FeedforwardPOSContext.shape("Paris"));
    assertEquals("*allcaps*", FeedforwardPOSContext.shape("USA"));
    assertEquals("*digit*", FeedforwardPOSContext.shape("2020"));
    assertEquals("*alnum*", FeedforwardPOSContext.shape("B2B"));
    assertEquals("*other*", FeedforwardPOSContext.shape("--"));
    assertEquals("*lower*", FeedforwardPOSContext.shape("dog"));
    assertEquals("og", FeedforwardPOSContext.suffix("dog", 2));
    assertEquals("dog", FeedforwardPOSContext.suffix("dog", 3));
    assertEquals("ing", FeedforwardPOSContext.suffix("running", 3));
  }

  @Test
  void testTopKSequencesReturnsTheGreedyTaggingWithRealProbabilities() {
    final String[] sentence = {"the", "dog", "barks"};
    final Sequence[] sequences = tagger.topKSequences(sentence);
    assertEquals(1, sequences.length);
    assertEquals(List.of(tagger.tag(sentence)), sequences[0].getOutcomes());
    final double[] probs = sequences[0].getProbs();
    assertEquals(sentence.length, probs.length);
    double expectedScore = 0.0;
    for (final double prob : probs) {
      assertTrue(prob > 0.0 && prob <= 1.0, "probability out of range: " + prob);
      expectedScore += StrictMath.log(prob);
    }
    assertEquals(expectedScore, sequences[0].getScore(), 1.0e-9);
  }

  /**
   * Pins the scoring cache against the direct path: the tagger built in setup turned
   * the cache on for the shared model, so scoring the same features through a fresh
   * reloaded model, which no tagger has touched, must agree to float rounding and
   * pick the same tag.
   *
   * @throws IOException Thrown if the round trip fails.
   */
  @Test
  void testScoringCacheMatchesTheDirectPath() throws IOException {
    final ByteArrayOutputStream out = new ByteArrayOutputStream();
    model.serialize(out);
    final FeedforwardPOSModel direct =
        FeedforwardPOSModel.load(new ByteArrayInputStream(out.toByteArray()));
    final String[] sentence = {"the", "dog", "barks"};
    final int[] features = model.featureIds(
        FeedforwardPOSContext.extract(sentence, 1, "DT", null));

    for (int round = 0; round < 3; round++) {
      final double[] cached = model.score(features);
      final double[] plain = direct.score(features);
      int bestCached = 0;
      int bestPlain = 0;
      for (int o = 0; o < cached.length; o++) {
        assertEquals(plain[o], cached[o],
            Math.max(1.0e-6, Math.abs(plain[o]) * 1.0e-6));
        if (cached[o] > cached[bestCached]) {
          bestCached = o;
        }
        if (plain[o] > plain[bestPlain]) {
          bestPlain = o;
        }
      }
      assertEquals(bestPlain, bestCached);
    }
  }

  /**
   * Pins the probabilities against the model's actual behavior rather than the range
   * alone: the tiny network memorizes its training corpus, so every per-token
   * probability of a memorized sentence must be near certainty. A decoder that
   * reported any fixed placeholder constant in the unit interval would fail here.
   */
  @Test
  void testMemorizedSentenceProbabilitiesAreConfident() {
    final Sequence[] sequences = tagger.topKSequences(new String[] {"the", "dog", "barks"});
    for (final double prob : sequences[0].getProbs()) {
      assertTrue(prob > 0.9, "memorized token should be tagged near certainty: " + prob);
    }
  }

  /**
   * Pins the javadoc-promised boundary: every sentence yields a length-one array, so
   * an empty sentence yields one empty sequence with no outcomes, no probabilities,
   * and a score of zero, the empty sum of log probabilities.
   */
  @Test
  void testTopKSequencesOnEmptySentenceYieldsOneEmptySequence() {
    final Sequence[] sequences = tagger.topKSequences(new String[0]);
    assertEquals(1, sequences.length);
    assertEquals(List.of(), sequences[0].getOutcomes());
    assertEquals(0, sequences[0].getProbs().length);
    assertEquals(0.0, sequences[0].getScore());
  }

  /**
   * Pins the null rejection across all four tagging overloads, which share one
   * decoder: each throws the documented exception rather than a raw
   * {@link NullPointerException}.
   */
  @Test
  void testNullSentenceIsRejectedByEveryOverload() {
    assertThrows(IllegalArgumentException.class, () -> tagger.tag(null));
    assertThrows(IllegalArgumentException.class,
        () -> tagger.tag(null, new Object[0]));
    assertThrows(IllegalArgumentException.class, () -> tagger.topKSequences(null));
    assertThrows(IllegalArgumentException.class,
        () -> tagger.topKSequences(null, new Object[0]));
  }

  @Test
  void testTopKSequencesIgnoresAdditionalContext() {
    final String[] sentence = {"she", "eats", "fish"};
    assertArrayEquals(tagger.topKSequences(sentence),
        tagger.topKSequences(sentence, new Object[] {"ignored"}));
  }

  /**
   * Pins the contract {@link opennlp.tools.parser.AbstractBottomUpParser#advanceTags}
   * depends on: it calls {@code topKSequences} unconditionally and turns every returned
   * probability into a log probability, so the tagger must return at least one sequence
   * whose probabilities are strictly positive.
   */
  @Test
  void testTopKSequencesFeedsTheBottomUpParser() {
    final Parse tokens = Parse.createFromTokens(new String[] {"the", "dog", "barks"});
    final Parse[] tagged = new TagOnlyParser(tagger).advanceTagsOf(tokens);
    assertEquals(1, tagged.length);
    assertTrue(Double.isFinite(tagged[0].getProb()),
        "the parser derived a non-finite probability: " + tagged[0].getProb());
  }

  /**
   * The smallest possible {@link AbstractBottomUpParser} that exercises the real
   * {@code advanceTags} implementation without needing a chunker or a parser model.
   */
  private static final class TagOnlyParser extends AbstractBottomUpParser {

    TagOnlyParser(POSTagger tagger) {
      super(tagger, null, new NoPunctuationHeadRules(), defaultBeamSize,
          defaultAdvancePercentage);
    }

    Parse[] advanceTagsOf(Parse p) {
      return advanceTags(p);
    }

    @Override
    protected Parse[] advanceParses(Parse p, double probMass) {
      return new Parse[0];
    }

    @Override
    protected void advanceTop(Parse p) {
      // Tagging is the only stage under test, so there is no top stage to advance.
    }
  }

  /** Head rules that model no punctuation, which is all the tagging stage consults. */
  private static final class NoPunctuationHeadRules implements HeadRules {

    @Override
    public Parse getHead(Parse[] constituents, String type) {
      return null;
    }

    @Override
    public Set<String> getPunctuationTags() {
      return Set.of();
    }
  }

  @Test
  void testArgumentValidation() {
    assertThrows(IllegalArgumentException.class, () -> new FeedforwardPOSTagger(null));
    assertThrows(IllegalArgumentException.class, () -> tagger.tag(null));
    assertThrows(IllegalArgumentException.class,
        () -> FeedforwardPOSTrainer.train(null, FeedforwardPOSTrainer.Settings.defaults()));
    assertThrows(IllegalArgumentException.class, () -> new FeedforwardPOSTrainer.Settings(
        0, 32, 10, 32, 0.05, 0.0, 0.0, 1, 1, 17L));
    assertThrows(IllegalArgumentException.class, () -> new FeedforwardPOSTrainer.Settings(
        16, 32, 10, 32, 0.05, 0.0, 1.0, 1, 1, 17L));
  }
}
