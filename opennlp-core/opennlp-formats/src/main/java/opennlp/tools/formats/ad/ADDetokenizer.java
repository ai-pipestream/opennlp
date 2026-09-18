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

package opennlp.tools.formats.ad;

import opennlp.tools.tokenize.Detokenizer;
import opennlp.tools.tokenize.TokenSample;
import opennlp.tools.util.Span;

/**
 * Joins an AD token that keeps a trailing hyphen to the token after it, when the detokenizer
 * dictionary says a hyphen joins to the right. The corpus writes the second part of such a word
 * on the next line and leaves the hyphen on the first part, for clitics
 * ({@code ofereceu-} {@code me}) and for names and codes alike
 * ({@code Projeto_Baleia--} {@code Jubarte}, {@code CEP_01290-} {@code 900}).
 */
final class ADDetokenizer implements Detokenizer {

  private final Detokenizer delegate;

  /** Whether the dictionary joins a hyphen to the token that follows it; read once. */
  private final boolean hyphenJoinsRight;

  ADDetokenizer(Detokenizer delegate) {
    this.delegate = delegate;
    DetokenizationOperation hyphen = delegate.detokenize(new String[] {"-"})[0];
    this.hyphenJoinsRight = hyphen == DetokenizationOperation.MERGE_TO_RIGHT
        || hyphen == DetokenizationOperation.MERGE_BOTH;
  }

  /** {@inheritDoc} */
  @Override
  public DetokenizationOperation[] detokenize(String[] tokens) {
    DetokenizationOperation[] operations = delegate.detokenize(tokens);
    if (hyphenJoinsRight) {
      for (int i = 0; i + 1 < tokens.length; i++) {
        if (tokens[i].length() > 1 && tokens[i].endsWith("-")) {
          operations[i] = switch (operations[i]) {
            case NO_OPERATION -> DetokenizationOperation.MERGE_TO_RIGHT;
            case MERGE_TO_LEFT -> DetokenizationOperation.MERGE_BOTH;
            default -> operations[i];
          };
        }
      }
    }
    return operations;
  }

  /** {@inheritDoc} */
  @Override
  public String detokenize(String[] tokens, String splitMarker) {
    TokenSample sample = new TokenSample(this, tokens);
    if (splitMarker == null) {
      return sample.getText();
    }
    StringBuilder text = new StringBuilder();
    int previousEnd = 0;
    Span[] spans = sample.getTokenSpans();
    for (int i = 0; i < spans.length; i++) {
      Span span = spans[i];
      if (i > 0 && previousEnd == span.getStart()) {
        text.append(splitMarker);
      } else {
        text.append(sample.getText(), previousEnd, span.getStart());
      }
      text.append(sample.getText(), span.getStart(), span.getEnd());
      previousEnd = span.getEnd();
    }
    return text.toString();
  }
}
