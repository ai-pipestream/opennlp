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

/*
 * This algorithm is updated based on code located at:
 * http://members.unine.ch/jacques.savoy/clef/
 *
 * Full copyright for that code follows:
 */
/*
 * Copyright (c) 2005, Jacques Savoy
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer. Redistributions in binary
 * form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials
 * provided with the distribution. Neither the name of the author nor the names
 * of its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package opennlp.tools.stemmer.light;

import opennlp.tools.commons.ThreadSafe;
import opennlp.tools.stemmer.Stemmer;
import opennlp.tools.stemmer.StemmerFactory;
/**
 * Light Stemmer for French.
 *
 * <p>This stemmer implements the following algorithm: <i>A Stemming procedure and stopword list for
 * general French corpora.</i> Jacques Savoy.
 *
 * <p>Adapted from the identically named algorithm in Apache Lucene's analysis-common module.
 * Instances are stateless and safe for concurrent use by multiple threads; each instance is also
 * its own {@link StemmerFactory}. Input is expected to be lowercase, as produced by a
 * case-folding normalization step; the stemmer does not fold case itself.</p>
 */
@ThreadSafe
public final class FrenchMinimalStemmer implements Stemmer, StemmerFactory {
  @Override
  public CharSequence stem(CharSequence word) {
    if (word == null) {
      throw new IllegalArgumentException("The word must not be null.");
    }
    final int length = word.length();
    final char[] buffer = new char[length];
    for (int i = 0; i < length; i++) {
      buffer[i] = word.charAt(i);
    }
    final int stemmedLength = stem(buffer, length);
    return new String(buffer, 0, stemmedLength);
  }

  @Override
  public Stemmer newStemmer() {
    return this;
  }

  private int stem(char[] s, int len) {
    if (len < 6) return len;

    if (s[len - 1] == 'x') {
      if (s[len - 3] == 'a' && s[len - 2] == 'u') s[len - 2] = 'l';
      return len - 1;
    }

    if (s[len - 1] == 's') len--;
    if (s[len - 1] == 'r') len--;
    if (s[len - 1] == 'e') len--;
    if (s[len - 1] == 'é') len--;
    if (s[len - 1] == s[len - 2] && Character.isLetter(s[len - 1])) len--;
    return len;
  }
}
