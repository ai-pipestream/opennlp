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

import java.util.List;
import java.util.Optional;

/**
 * The wordnet lexicon seam: lemma and synset lookup over a loaded wordnet-style resource.
 *
 * <p>This interface is the contract every wordnet-shaped dataset sits behind. A legacy Princeton
 * WNDB directory, a WN-LMF XML document (the Global WordNet Association interchange format used
 * by Open English WordNet and many other language wordnets), a future bundled
 * permissively-licensed lexicon, and a future user-downloaded CC-BY lexicon are all
 * implementations of this one seam, so a consumer written against {@code WordNetLexicon} never
 * changes when the data tier does. Nothing in the contract names a particular resource; synset
 * identity is opaque and source-qualified (see {@link Synset#id()}).</p>
 *
 * <p>The surface is intentionally minimal but sufficient for the layered features above it:
 * lemma lookup and membership for morphological analysis (Morphy-style lemmatization), and
 * synset retrieval with typed relation navigation for query expansion and, later, similarity
 * measures. Feature layers stack on these four operations without a contract change.</p>
 *
 * <p>Lemma matching semantics are the implementation's concern. The reference implementations
 * match case-insensitively (case folding with the root locale) and treat the underscore some
 * formats store in multiword lemmas as a space, which is the recommended behavior; an
 * implementation with different semantics must document them. Returned {@link Synset#lemmas()
 * lemmas} preserve the source's written forms, with spaces in multiword lemmas.</p>
 *
 * <p>Implementations must be immutable and thread-safe after loading: one instance is meant to
 * be shared across an application's threads for concurrent lookups.</p>
 */
public interface WordNetLexicon {

  /**
   * Finds the synsets containing a lemma with a part of speech, in the source's sense order
   * (the most salient sense first when the source ranks senses).
   *
   * @param lemma The lemma to look up. Must not be {@code null}.
   * @param pos   The part of speech to look it up as. Must not be {@code null}.
   * @return The matching synsets, never {@code null}; empty when the lexicon does not contain
   *     the lemma with that part of speech.
   * @throws IllegalArgumentException Thrown if {@code lemma} or {@code pos} is {@code null}.
   */
  List<Synset> lookup(String lemma, WordNetPos pos);

  /**
   * Finds a synset by its opaque identifier.
   *
   * @param synsetId The synset identifier, as minted by this lexicon. Must not be {@code null}.
   * @return The synset, or empty when this lexicon has no synset with that identifier.
   * @throws IllegalArgumentException Thrown if {@code synsetId} is {@code null}.
   */
  Optional<Synset> synset(String synsetId);

  /**
   * Navigates one typed relation from a synset.
   *
   * @param synsetId The source synset identifier. Must not be {@code null}.
   * @param relation The relation type to follow. Must not be {@code null}.
   * @return The target synset ids in source order, never {@code null}; empty when the synset is
   *     unknown or has no relation of that type.
   * @throws IllegalArgumentException Thrown if {@code synsetId} or {@code relation} is
   *     {@code null}.
   */
  default List<String> related(String synsetId, WordNetRelation relation) {
    if (relation == null) {
      throw new IllegalArgumentException("Relation must not be null");
    }
    return synset(synsetId).map(s -> s.related(relation)).orElse(List.of());
  }

  /**
   * Tests whether the lexicon contains a lemma with a part of speech. This is the membership
   * check morphological rules validate their candidates against; implementations may override
   * it with a cheaper check than {@link #lookup(String, WordNetPos)}.
   *
   * @param lemma The lemma to test. Must not be {@code null}.
   * @param pos   The part of speech to test it as. Must not be {@code null}.
   * @return {@code true} if the lexicon contains the lemma with that part of speech.
   * @throws IllegalArgumentException Thrown if {@code lemma} or {@code pos} is {@code null}.
   */
  default boolean contains(String lemma, WordNetPos pos) {
    return !lookup(lemma, pos).isEmpty();
  }
}
