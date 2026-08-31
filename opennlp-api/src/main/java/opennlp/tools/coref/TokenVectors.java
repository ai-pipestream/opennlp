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

package opennlp.tools.coref;

/**
 * Supplies one vector per token of a sentence, computed in the context of the whole
 * sentence, so a coreference model can compare mentions by meaning in context rather
 * than by surface form. Implementations wrap an encoder such as a transformer exported
 * to ONNX; the caller supplies the tokens it works with and receives their vectors in
 * the same order.
 *
 * <p>Thread safety is implementation specific.</p>
 *
 * @since 3.0.0
 */
@FunctionalInterface
public interface TokenVectors {

  /**
   * Encodes the tokens of one sentence.
   *
   * @param tokens The tokens in order. Must not be {@code null} or empty.
   * @return One vector per token, in the same order and all of the same length. Never
   *         {@code null}.
   */
  float[][] vectors(String[] tokens);
}
