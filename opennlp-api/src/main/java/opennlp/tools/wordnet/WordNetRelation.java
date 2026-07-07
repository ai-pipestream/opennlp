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
 * The typed relations a wordnet-style lexicon draws between {@link Synset synsets}.
 *
 * <p>The value set covers the pointer types documented for the legacy Princeton WNDB format,
 * which is also the common core the WN-LMF interchange format expresses; both readers map their
 * format's names onto these values, so consumers navigate one relation vocabulary regardless of
 * the data tier behind the {@link WordNetLexicon} seam. {@link #PARTICIPLE} is part of that
 * documented set (the adjective &quot;participle of verb&quot; pointer) even though it is easy
 * to overlook; without it, real Princeton data would be unreadable. Two values go beyond the
 * WNDB pointer set: {@link #ENTAILED_BY} and {@link #CAUSED_BY} cover the inverse relations
 * WN-LMF documents such as Open English WordNet materialize, so reading such a document loses
 * no typed relation; a WNDB-backed lexicon never produces them because the legacy format stores
 * only the forward direction.</p>
 *
 * <p>Some of these relations are, in the source formats, drawn between individual word senses
 * rather than whole synsets (antonymy and derivation are the common examples). In this v1
 * contract they surface at the synset level: the synset containing the source sense carries the
 * relation to the synset containing the target sense. A sense-level view is a later, additive
 * layer.</p>
 *
 * <p>Enum constants are immutable and safe to share across threads.</p>
 */
public enum WordNetRelation {

  /** Opposition in meaning, for example between the adjectives for tall and short. */
  ANTONYM,

  /** The more general concept: a dog is a kind of canid. */
  HYPERNYM,

  /** The class a named instance belongs to: a specific river is an instance of river. */
  INSTANCE_HYPERNYM,

  /** The more specific concept: canid has the hyponym dog. */
  HYPONYM,

  /** A named instance of this class. */
  INSTANCE_HYPONYM,

  /** The group this synset is a member of. */
  MEMBER_HOLONYM,

  /** The whole this synset is a substance of. */
  SUBSTANCE_HOLONYM,

  /** The whole this synset is a part of. */
  PART_HOLONYM,

  /** A member of this group. */
  MEMBER_MERONYM,

  /** A substance this synset is made of. */
  SUBSTANCE_MERONYM,

  /** A part of this synset. */
  PART_MERONYM,

  /** The attribute a value expresses, or a value of this attribute. */
  ATTRIBUTE,

  /** A derivationally related form, typically across parts of speech. */
  DERIVATIONALLY_RELATED,

  /** An action entailed by this verb: snoring entails sleeping. */
  ENTAILMENT,

  /** The verb that entails this one; the inverse of {@link #ENTAILMENT}. */
  ENTAILED_BY,

  /** An effect this verb causes. */
  CAUSE,

  /** The cause of this verb; the inverse of {@link #CAUSE}. */
  CAUSED_BY,

  /** A related synset worth consulting. */
  ALSO_SEE,

  /** A verb sense grouped with this one. */
  VERB_GROUP,

  /** A satellite or head adjective in the same similarity cluster. */
  SIMILAR_TO,

  /** The verb an adjective is the participle of. */
  PARTICIPLE,

  /**
   * The noun an adjective pertains to, or the adjective an adverb derives from. The source
   * formats use one pointer for both directions of derivation, so this value does too.
   */
  PERTAINYM,

  /** The topical domain this synset belongs to. */
  DOMAIN_TOPIC,

  /** A synset belonging to this topical domain. */
  MEMBER_OF_DOMAIN_TOPIC,

  /** The regional domain this synset belongs to. */
  DOMAIN_REGION,

  /** A synset belonging to this regional domain. */
  MEMBER_OF_DOMAIN_REGION,

  /** The usage domain this synset belongs to, for example slang or archaism. */
  DOMAIN_USAGE,

  /** A synset belonging to this usage domain. */
  MEMBER_OF_DOMAIN_USAGE
}
