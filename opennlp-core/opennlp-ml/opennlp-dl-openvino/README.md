# OpenNLP DL (OpenVINO)

This module contributes the ONNX Runtime **OpenVINO execution provider** to `opennlp-dl`, under the
execution provider id `openvino`. It is an addon: it adds no class to `opennlp-dl` and changes none.
Putting it on the classpath is what makes the id usable, because it lists its configurer in
`META-INF/services/opennlp.dl.ExecutionProviderConfigurer` and `opennlp-dl` resolves execution
providers through that SPI.

```xml
<dependency>
    <groupId>org.apache.opennlp</groupId>
    <artifactId>opennlp-dl-openvino</artifactId>
    <version>${opennlp.version}</version>
 </dependency>
```

## Asking for OpenVINO

```java
InferenceOptions options = new InferenceOptions();
options.setExecutionProviders(List.of(
    ExecutionProviderRequest.of("openvino", Map.of("device_type", "GPU.0")),
    ExecutionProviderRequest.of("cpu")));
```

ONNX Runtime keeps the execution providers of a session in the order they were appended and runs a
node on the first one that accepts it, so the list above means "OpenVINO where it can, CPU for the
rest". Pass the options to any component:

```java
// Sentence vectors, with the padding strategy derived from the device type
SentenceVectorsDL.withDerivedPadding(model, vocabulary, true, Pooling.MEAN, true,
    SentenceVectorsDL.DEFAULT_MAX_LENGTH, options);

// Document categorization
new DocumentCategorizerDL(model, vocabulary, categories, options);

// Name finding
new NameFinderDL(model, vocabulary, ids2Labels, options, sentenceDetector);
```

## Asking for OpenVINO from a spec

A component configured by name, through `TextEmbedderProvider`, asks for this execution provider in
the `executionProviders` option of its spec:

```
executionProviders=openvino(device_type=GPU.0),cpu
```

`OnnxTextEmbedderProvider` documents the syntax. The short form: `,` separates the requests of the
ordered list and `;` separates the provider options inside one request, which is why a device type
holding a comma, such as `openvino(device_type=MULTI:GPU,CPU),cpu`, needs no escaping. The id is the
id this module's configurer answers to, so nothing about this option is specific to OpenVINO: an
addon becomes reachable from a spec by registering a configurer, and the id is how a spec names it.

The deprecated `gpu` and `gpuDeviceId` spec options still work and still mean CUDA. They are read
only while `executionProviders` is absent, which is the precedence
`ExecutionProviders.resolve(InferenceOptions)` documents for the two settings they map onto.

## The device type is the whole configuration

ONNX Runtime registers this execution provider with `SessionOptions.addOpenVINO(String)`, whose one
argument is an OpenVINO device type, so `device_type` is the only provider option this configurer
accepts. Any other name is reported rather than ignored.

| `device_type`                      | Runs on               | Placement stated |
|------------------------------------|-----------------------|------------------|
| `CPU`                              | the host CPU          | CPU              |
| `GPU`, `GPU.0`, `GPU.1`            | an integrated or discrete Intel GPU | accelerator |
| `NPU`                              | an Intel NPU          | accelerator      |
| `AUTO`, `MULTI:GPU,CPU`, `HETERO:GPU,CPU` | whichever OpenVINO picks | unstated  |
| absent, or a device type a later OpenVINO release added | whichever OpenVINO picks | unstated |

The device type is handed to ONNX Runtime as it stands: OpenVINO's list of device types grows with
its releases and with what the machine has, and a list kept here would go stale. A blank value is the
one rejected case, since it is a configuration mistake rather than a request for the default. Leave
the option out to let OpenVINO choose.

## Why the placement matters, and why it is not in the id

The placement column above is what
`ExecutionProviderConfigurer.placement(ProviderSpec)` returns, and a setting that depends on where a
session runs reads it. The first one is the padding strategy of `SentenceVectorsDL`: padding a batch
to its longest row is six to eight times faster than grouping by exact tokenized length on an
accelerator and 0.66 to 0.81 times as fast on a CPU, so the better default is the opposite one on the
two placements.

This execution provider is the reason that question is answered by the configurer rather than by a
list of ids in `opennlp-dl`. One id, `openvino`, covers `device_type=GPU.0` and `device_type=CPU`,
which are two placements, so `openvino` in a list of accelerator ids would be right half the time and
wrong the other half. The configurer reads its own provider option and answers for the request in
hand, and `opennlp-dl` holds no addon id at all.

A device type that decides nothing leaves the placement unstated, and an unstated placement keeps
the conservative default, which is grouping by exact tokenized length. So this jar on the classpath
changes no vectors until a `device_type` asks it to. Answering for `AUTO` would mean asking the
machine which device OpenVINO would pick, and provider lookup may not load a native library.

## What running on it needs

`addOpenVINO` is part of the base `onnxruntime` API, so this module needs no further artifact to
compile or to register a request. **Running** a session on OpenVINO needs two things this module
cannot bring:

- an ONNX Runtime built with the OpenVINO execution provider, which means built with
  `--use_openvino`. No published Maven artifact carries one. Both `onnxruntime` and `onnxruntime_gpu`
  hold a runtime that dynamically loads `libonnxruntime_providers_openvino.so`, and neither jar
  contains that library, so `addOpenVINO` on either of them fails with
  `ORT_EP_FAIL: Failed to find OpenVINO shared provider` whatever the device type is.
- the Intel OpenVINO toolkit that the runtime was built against, on the library path of the process.

A deployment that wants this supplies both, the same way a deployment that wants CUDA supplies a CUDA
installation `onnxruntime_gpu` can load. Until it does, a request for `openvino` fails at
construction with an `OrtException` and is reported: ONNX Runtime does not fall back to the CPU, and
neither does OpenNLP. A session that silently ran somewhere else than the caller asked for is what
this SPI exists to prevent.

The tests of this module therefore observe the registration rather than perform it, with the
`RecordingSessionOptions` that `opennlp-dl` uses for the same reason. They establish that the device
type of a request reaches `addOpenVINO` and that it decides the padding default. They do not
establish that OpenVINO ran anything, and two of them record what the pinned runtime does with a real
request so that the gap is on the record rather than assumed.
