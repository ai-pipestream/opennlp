/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package opennlp.wordnet;

import java.io.IOException;

/**
 * Supplies the base lexicon documents a WN-LMF {@code LexiconExtension} composes against. The
 * application controls all I/O: {@link WnLmfReader} never opens a file or fetches a URL itself,
 * it only asks a caller-supplied resolver for the {@code Extends} reference of each extension it
 * encounters, including the bases of extension chains.
 *
 * <p>A resolver must return a fresh {@link WnLmfSource} per call and must never return
 * {@code null}; when it cannot obtain the referenced document it must throw an
 * {@link IOException} naming the requested id and version. The reader owns a successfully
 * returned source and closes it exactly once, whether reading succeeds or fails. During one
 * top-level read the reader caches resolutions per exact {@code (ref, version)} pair, so a
 * resolver is consulted at most once per referenced base.</p>
 *
 * @since 3.0.0
 */
@FunctionalInterface
public interface WnLmfResolver {

  /**
   * Opens the document containing the referenced lexicon.
   *
   * @param reference The {@code Extends} reference to resolve, carrying the id, exact version,
   *                  and, when the source declared one, a url hint. Never {@code null}.
   * @return A freshly opened source for the document that contains the referenced lexicon.
   *         Must not be {@code null} and must not have been returned before.
   * @throws IOException Thrown if the referenced document cannot be obtained.
   */
  WnLmfSource resolve(WnLmfDependency reference) throws IOException;
}
