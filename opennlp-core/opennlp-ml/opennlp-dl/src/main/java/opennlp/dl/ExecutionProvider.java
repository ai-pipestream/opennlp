/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package opennlp.dl;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * One ONNX Runtime execution provider, with its provider options already read, ready to be
 * appended to a session's options.
 *
 * <p>An instance is produced by an {@link ExecutionProviderConfigurer} from an
 * {@link ExecutionProviderRequest}, so every value the request carried has been validated by the
 * time this exists. Appending is separate from creating because the two fail differently: a
 * malformed provider option is rejected while the instance is created, whereas ONNX Runtime
 * rejects a provider it cannot register while it is being appended.</p>
 *
 * <p>Implementations must be immutable and may be appended to more than one set of session
 * options.</p>
 *
 * @see ExecutionProviders#addTo(OrtSession.SessionOptions, java.util.List)
 * @since 3.0.0
 */
@FunctionalInterface
public interface ExecutionProvider {

  /**
   * Appends this execution provider to {@code sessionOptions}. ONNX Runtime keeps the execution
   * providers of a session in the order they were appended and runs a node on the first one that
   * accepts it, so the call order is the priority order.
   *
   * <p>ONNX Runtime does not fall back to the CPU when a provider cannot be registered; it
   * reports the failure. Implementations must not catch such a failure, because a session that
   * silently ran somewhere else than the caller asked for is worse than one that did not start.
   * </p>
   *
   * @param sessionOptions The session options to append to. Must not be {@code null}.
   * @throws IllegalArgumentException Thrown if {@code sessionOptions} is {@code null}.
   * @throws OrtException Thrown if ONNX Runtime cannot register this execution provider, which is
   *     what happens when the runtime on the class path was not built with it, when its shared
   *     library cannot be loaded, or when a provider option or a device it names is rejected.
   */
  void addTo(OrtSession.SessionOptions sessionOptions) throws OrtException;
}
