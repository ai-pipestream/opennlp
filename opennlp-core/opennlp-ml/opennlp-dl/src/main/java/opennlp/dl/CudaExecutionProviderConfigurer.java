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

import java.util.Map;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.providers.OrtCUDAProviderOptions;

import opennlp.tools.util.ext.ProviderSpec;

/**
 * Registers the ONNX Runtime CUDA execution provider under the id {@value #ID}. This is one of the
 * two configurers {@code opennlp-dl} supplies itself, because
 * {@link ai.onnxruntime.OrtSession.SessionOptions#addCUDA(int)} is part of the base
 * {@code onnxruntime} API and needs no further artifact at compile time.
 *
 * <p>Running on it does need the {@code onnxruntime_gpu} runtime at execution time, which the
 * {@code opennlp-dl-gpu} module brings in, plus a CUDA installation the runtime can load and a
 * device that answers to the requested id. If any of that is missing, registering fails and the
 * failure is reported: ONNX Runtime does not fall back to the CPU, and neither does this.</p>
 *
 * <p>{@value #DEVICE_ID_OPTION} is read here so that a malformed value is rejected before the
 * runtime is reached. Every other option is handed to ONNX Runtime as it stands, through
 * {@link OrtCUDAProviderOptions}, which is also what decides whether a name is known; keeping a
 * list of accepted names here would go stale with every runtime release. A request that carries
 * only {@value #DEVICE_ID_OPTION}, or no option at all, is registered with
 * {@link ai.onnxruntime.OrtSession.SessionOptions#addCUDA(int)} instead, which needs no provider
 * option object.</p>
 *
 * @since 3.0.0
 */
public final class CudaExecutionProviderConfigurer implements ExecutionProviderConfigurer {

  /** The id this configurer answers to. */
  public static final String ID = "cuda";

  /**
   * The option that names the CUDA device to run on, a non-negative integer, {@code 0} by default.
   * It carries ONNX Runtime's own name for it, so a caller that also sets other CUDA provider
   * options writes them all the same way.
   */
  public static final String DEVICE_ID_OPTION = "device_id";

  /** {@inheritDoc} */
  @Override
  public String name() {
    return ID;
  }

  /**
   * {@inheritDoc}
   * Checks that the ONNX Runtime classes are present without initializing the runtime. This says
   * nothing about whether CUDA itself is usable, which cannot be answered without loading the
   * shared provider library, so it is answered by the attempt to register.
   */
  @Override
  public boolean isAvailable() {
    try {
      return OrtEnvironment.class.getName() != null;
    } catch (final LinkageError e) {
      return false;
    }
  }

  /**
   * {@inheritDoc}
   * Any option name is supported, since ONNX Runtime owns the CUDA option namespace.
   */
  @Override
  public boolean supports(final ProviderSpec spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    return spec.location().isEmpty();
  }

  /**
   * {@inheritDoc}
   * The returned {@link ExecutionProvider} appends the CUDA execution provider on the requested
   * device.
   *
   * @throws IllegalArgumentException Thrown if {@code spec} is {@code null}, carries a location,
   *     or gives {@value #DEVICE_ID_OPTION} a value that is not a non-negative integer.
   */
  @Override
  public ExecutionProvider create(final ProviderSpec spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    if (spec.location().isPresent()) {
      throw new IllegalArgumentException(
          "the " + ID + " execution provider takes no location: " + spec);
    }
    final int deviceId = deviceId(spec);
    final Map<String, String> options = spec.options();
    if (options.isEmpty() || options.size() == 1 && options.containsKey(DEVICE_ID_OPTION)) {
      return sessionOptions -> {
        if (sessionOptions == null) {
          throw new IllegalArgumentException("sessionOptions must not be null");
        }
        sessionOptions.addCUDA(deviceId);
      };
    }
    return sessionOptions -> {
      if (sessionOptions == null) {
        throw new IllegalArgumentException("sessionOptions must not be null");
      }
      // Created here rather than in create(): it holds a native handle, and the SPI contract keeps
      // provider lookup free of native initialization.
      try (OrtCUDAProviderOptions providerOptions = new OrtCUDAProviderOptions(deviceId)) {
        for (final Map.Entry<String, String> option : options.entrySet()) {
          if (!DEVICE_ID_OPTION.equals(option.getKey())) {
            providerOptions.add(option.getKey(), option.getValue());
          }
        }
        sessionOptions.addCUDA(providerOptions);
      }
    };
  }

  /**
   * Reads {@value #DEVICE_ID_OPTION}.
   *
   * @param spec The spec to read. Must not be {@code null}.
   * @return The device id, {@code 0} if the option is absent.
   * @throws IllegalArgumentException Thrown if the value is not a non-negative integer.
   */
  private int deviceId(final ProviderSpec spec) {
    final String value = spec.option(DEVICE_ID_OPTION, null);
    if (value == null) {
      return 0;
    }
    try {
      final int deviceId = Integer.parseInt(value);
      if (deviceId >= 0) {
        return deviceId;
      }
    } catch (final NumberFormatException e) {
      // reported below
    }
    throw new IllegalArgumentException("the " + ID + " execution provider option '"
        + DEVICE_ID_OPTION + "' must be a non-negative integer, not '" + value + "'");
  }
}
