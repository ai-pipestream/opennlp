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
 * Exposes {@link AbstractDL#sessionOptions(InferenceOptions)} and
 * {@link AbstractDL#configureSession(OrtSession.SessionOptions, InferenceOptions)} to tests outside
 * this package, which cannot reach a {@code protected static} member of {@link AbstractDL} on their
 * own. It exists so that a test can compare the session options a component builds against a plain
 * default {@code OrtSession.SessionOptions} without loading a model, and so that a test can hand
 * the configuration step session options of its own that record what was done to them.
 */
public final class SessionOptionsProbe {

  private SessionOptionsProbe() {
  }

  /**
   * Applies the given inference options to session options the caller owns, which is what
   * {@link AbstractDL} does to the options it hands to a new session.
   *
   * @param sessionOptions The session options to configure, which may be a subclass that records
   *     the calls it receives.
   * @param inferenceOptions The options to apply.
   * @throws OrtException Thrown if a requested execution provider cannot be registered or a
   *     session setting is rejected.
   */
  public static void configure(final OrtSession.SessionOptions sessionOptions,
      final InferenceOptions inferenceOptions) throws OrtException {
    AbstractDL.configureSession(sessionOptions, inferenceOptions);
  }

  /**
   * {@return the session options {@link AbstractDL} builds for these inference options}
   *
   * <p>The caller owns the returned options and must close them.</p>
   *
   * @param inferenceOptions The options to build from.
   * @throws OrtException Thrown if the requested execution provider cannot be added.
   */
  public static OrtSession.SessionOptions of(final InferenceOptions inferenceOptions)
      throws OrtException {
    return AbstractDL.sessionOptions(inferenceOptions);
  }

}
