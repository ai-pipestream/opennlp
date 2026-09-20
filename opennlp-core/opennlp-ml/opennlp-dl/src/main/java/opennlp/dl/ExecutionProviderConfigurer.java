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

import opennlp.tools.util.ext.Provider;
import opennlp.tools.util.ext.ProviderSpec;
import opennlp.tools.util.ext.Providers;

/**
 * The SPI for {@link ExecutionProvider ONNX Runtime execution providers}. Implementations are
 * listed in {@code META-INF/services/opennlp.dl.ExecutionProviderConfigurer} and are resolved by
 * {@link Provider#name()}, which is the id an {@link ExecutionProviderRequest} carries.
 *
 * <p>This is the seam that lets an execution provider live outside {@code opennlp-dl}. ONNX
 * Runtime's Java API exposes fourteen of them, most of which need a runtime built for the
 * accelerator and sometimes a further artifact, so {@code opennlp-dl} itself supplies only
 * {@value ExecutionProviders#CPU} and {@value ExecutionProviders#CUDA}, the two that the base
 * {@code onnxruntime} API can always express. An addon on the class path contributes any other by
 * registering a configurer under the id it wants to answer to; nothing in {@code opennlp-dl} has
 * to know it exists.</p>
 *
 * <p>The provider options of a request become the {@link ProviderSpec#options() options} of the
 * spec passed to {@link Provider#create(ProviderSpec)}, and the spec has no location. They are
 * strings on both sides, because ONNX Runtime's own provider option maps are string keyed.</p>
 *
 * <p>As for every OpenNLP SPI, an implementation needs a public no-argument constructor and must
 * be stateless and thread-safe, and neither its constructor nor {@link Provider#isAvailable()} nor
 * {@link Provider#supports(ProviderSpec)} may initialize a native library: those run while
 * providers are being looked up, before a caller has chosen one.</p>
 *
 * @see Providers
 * @see ExecutionProviders
 * @since 3.0.0
 */
public interface ExecutionProviderConfigurer extends Provider<ExecutionProvider> {
}
