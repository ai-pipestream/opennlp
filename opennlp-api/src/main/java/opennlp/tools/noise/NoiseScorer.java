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

package opennlp.tools.noise;

import java.util.Collection;
import java.util.List;

import opennlp.tools.util.Span;

/**
 * Scores the noisy stretches of a text and reports them as {@link NoiseSpan} findings
 * over the original, unmodified text.
 *
 * <p>Thread safety is implementation specific.</p>
 *
 * @since 3.0.0
 */
public interface NoiseScorer {

  /**
   * Scores the noise of a text.
   *
   * @param text The text to scan. Must not be {@code null}.
   * @param exclude Regions to leave out of scoring, such as spans another detector
   *                already explained. May be empty; must not be {@code null} or hold
   *                {@code null}.
   * @return The noise spans in order of appearance; empty when the text is clean.
   *         Never {@code null}.
   * @throws IllegalArgumentException Thrown if an argument is {@code null} or
   *         {@code exclude} holds {@code null}.
   */
  List<NoiseSpan> score(CharSequence text, Collection<Span> exclude);
}
