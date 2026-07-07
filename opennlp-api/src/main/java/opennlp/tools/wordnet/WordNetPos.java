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
package opennlp.tools.wordnet;

/**
 * The four parts of speech a wordnet-style lexicon distinguishes.
 *
 * <p>This enum is deliberately format-free: it carries none of the single-letter codes the
 * on-disk formats use, so the contract does not leak a storage detail. Readers own the mapping
 * from their format's codes to these values.</p>
 *
 * <p>Adjective satellites (the {@code s} code some formats use for adjectives clustered around
 * a head adjective) normalize to {@link #ADJECTIVE}: the head/satellite split is a storage
 * layout, not a part of speech, and the cluster structure is preserved through the
 * {@link WordNetRelation#SIMILAR_TO} relation instead.</p>
 *
 * <p>Enum constants are immutable and safe to share across threads.</p>
 */
public enum WordNetPos {

  /** Nouns. */
  NOUN,

  /** Verbs. */
  VERB,

  /** Adjectives, including adjective satellites. */
  ADJECTIVE,

  /** Adverbs. */
  ADVERB
}
