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

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import opennlp.tools.ml.model.Event;
import opennlp.tools.util.AbstractEventStream;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.Span;

/**
 * This class reads the {@link TokenSample samples} via an {@link Iterator}
 * and converts the samples into {@link Event events} which
 * can be used by the maxent library for training.
 */
public class TokSpanEventStream extends AbstractEventStream<TokenSample> {

  private static final Logger logger = LoggerFactory.getLogger(TokSpanEventStream.class);
  private final TokenContextGenerator cg;

  private final boolean skipAlphaNumerics;

  private final TokenizerCharacterPolicy characterPolicy;

  /**
   * Initializes a new event stream based on the data stream using a {@link TokenContextGenerator}.
   *
   * @param tokenSamples The {@link ObjectStream data stream} for this event stream.
   * @param skipAlphaNumerics Whether alphanumerics are skipped, or not.
   * @param characterPolicy The policy shared with tokenizer inference.
   * @param cg A {@link TokenContextGenerator} which should be used for the event stream {@code d}.
   */
  public TokSpanEventStream(ObjectStream<TokenSample> tokenSamples, boolean skipAlphaNumerics,
                            TokenizerCharacterPolicy characterPolicy, TokenContextGenerator cg) {
    super(requireTokenSamples(tokenSamples));
    if (characterPolicy == null || cg == null) {
      throw new IllegalArgumentException("characterPolicy and context generator must not be null");
    }
    this.characterPolicy = characterPolicy;
    this.skipAlphaNumerics = skipAlphaNumerics;
    this.cg = cg;
  }

  /**
   * Initializes a new event stream based on the data stream using a {@link TokenContextGenerator}.
   *
   * @param tokenSamples The {@link ObjectStream data stream} for this event stream.
   * @param skipAlphaNumerics Whether alphanumerics are skipped, or not.
   * @param cg A {@link TokenContextGenerator} which should be used for the event stream {@code d}.
   */
  public TokSpanEventStream(ObjectStream<TokenSample> tokenSamples, boolean skipAlphaNumerics,
                            TokenContextGenerator cg) {
    this(tokenSamples, skipAlphaNumerics, TokenizerCharacterPolicy.ascii(), cg);
  }

  /**
   * Initializes a new event stream based on the data stream using a {@link TokenContextGenerator}
   * that relies on a {@link DefaultTokenContextGenerator}.
   *
   * @param tokenSamples The {@link ObjectStream data stream} for this event stream.
   * @param skipAlphaNumerics Whether alphanumerics are skipped, or not.
   */
  public TokSpanEventStream(ObjectStream<TokenSample> tokenSamples,
      boolean skipAlphaNumerics) {
    this(tokenSamples, skipAlphaNumerics, new DefaultTokenContextGenerator());
  }

  /**
   * Adds training events to the event stream for each of the specified {@link TokenSample sample}.
   *
   * @param tokenSample character offsets into the specified text.
   * @return An {@link Iterator} for text {@link Event events} representing the {@code tokenSample}.
   */
  @Override
  protected Iterator<Event> createEvents(TokenSample tokenSample) {

    List<Event> events = new ArrayList<>(50);

    Span[] tokens = tokenSample.getTokenSpans();
    String text = tokenSample.getText();

    int previousEnd = -1;
    for (Span token : tokens) {
      if (token.getStart() == token.getEnd()) {
        throw new IllegalArgumentException("Training token spans must not be empty: " + token);
      }
      if (token.getStart() < previousEnd) {
        throw new IllegalArgumentException(
            "Training token spans must be ordered and non-overlapping: " + token);
      }
      if (token.getStart() < 0 || token.getEnd() > text.length()) {
        throw new IllegalArgumentException("Training token span is outside the text: " + token);
      }
      requireCodePointBoundary(text, token.getStart());
      requireCodePointBoundary(text, token.getEnd());
      previousEnd = token.getEnd();
    }

    if (tokens.length > 0) {

      int start = tokens[0].getStart();
      int end = tokens[tokens.length - 1].getEnd();

      String sent = text.substring(start, end);

      Span[] candTokens = WhitespaceTokenizer.INSTANCE.tokenizePos(sent);

      int firstTrainingToken = -1;
      int lastTrainingToken = -1;
      int annotationCursor = 0;
      for (Span candToken : candTokens) {
        Span cSpan = candToken;
        String ctok = sent.substring(cSpan.getStart(), cSpan.getEnd());
        //adjust cSpan to text offsets
        cSpan = new Span(cSpan.getStart() + start, cSpan.getEnd() + start);
        boolean policyMatch = ctok.length() > 1 && characterPolicy.test(ctok);
        while (annotationCursor < tokens.length
            && tokens[annotationCursor].getEnd() <= cSpan.getStart()) {
          annotationCursor++;
        }
        if (skipAlphaNumerics && policyMatch
            && hasAnnotatedBoundaryInside(cSpan, tokens, annotationCursor)) {
          throw new IllegalArgumentException("Training annotation splits a candidate accepted by "
              + "the tokenizer character policy: " + cSpan);
        }
        //should we skip this token
        if (ctok.length() > 1 && (!skipAlphaNumerics || !policyMatch)) {

          //find offsets of annotated tokens inside of candidate tokens
          boolean foundTrainingTokens = false;
          for (int ti = lastTrainingToken + 1; ti < tokens.length; ti++) {
            if (cSpan.contains(tokens[ti])) {
              if (!foundTrainingTokens) {
                firstTrainingToken = ti;
                foundTrainingTokens = true;
              }
              lastTrainingToken = ti;
            }
            else if (cSpan.getEnd() < tokens[ti].getEnd()) {
              break;
            }
            else if (tokens[ti].getEnd() < cSpan.getStart()) {
              //keep looking
            }
            else {
              logger.warn("Bad training token: {} cand: {} token={}", tokens[ti], cSpan,
                  text.substring(tokens[ti].getStart(), tokens[ti].getEnd()));
            }
          }

          // create training data
          if (foundTrainingTokens) {

            for (int ti = firstTrainingToken; ti <= lastTrainingToken; ti++) {
              Span tSpan = tokens[ti];
              int cStart = cSpan.getStart();
              for (int i = tSpan.getStart();;) {
                int boundary = i + Character.charCount(text.codePointAt(i));
                if (boundary >= tSpan.getEnd()) {
                  break;
                }
                String[] context = cg.getContext(ctok, boundary - cStart);
                events.add(new Event(TokenizerME.NO_SPLIT, context));
                i = boundary;
              }

              if (tSpan.getEnd() != cSpan.getEnd()) {
                String[] context = cg.getContext(ctok, tSpan.getEnd() - cStart);
                events.add(new Event(TokenizerME.SPLIT, context));
              }
            }
          }
        }
      }
    }

    return events.iterator();
  }

  private static boolean hasAnnotatedBoundaryInside(
      Span candidate, Span[] tokens, int firstPossibleToken) {
    for (int i = firstPossibleToken; i < tokens.length; i++) {
      Span token = tokens[i];
      if (token.getStart() >= candidate.getEnd()) {
        break;
      }
      if (token.getEnd() > candidate.getStart() && token.getEnd() < candidate.getEnd()) {
        return true;
      }
    }
    return false;
  }

  private static ObjectStream<TokenSample> requireTokenSamples(
      ObjectStream<TokenSample> tokenSamples) {
    if (tokenSamples == null) {
      throw new IllegalArgumentException("tokenSamples must not be null");
    }
    return tokenSamples;
  }

  private static void requireCodePointBoundary(String text, int offset) {
    if (offset > 0 && offset < text.length() && Character.isLowSurrogate(text.charAt(offset))
        && Character.isHighSurrogate(text.charAt(offset - 1))) {
      throw new IllegalArgumentException("Training token boundary splits a UTF-16 surrogate pair "
          + "at offset " + offset);
    }
  }
}
