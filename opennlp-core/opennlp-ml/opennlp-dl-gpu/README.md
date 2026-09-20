# OpenNLP DL (GPU)

This module brings in `onnxruntime_gpu` bindings to the existing `opennlp-dl` module. If you are planning to run with GPU acceleration, please use this BOM.

You can use it in your code by adding the following as a dependency:

```xml
<dependency>
    <groupId>org.apache.opennlp</groupId>
    <artifactId>opennlp-dl-gpu</artifactId>
    <version>${opennlp.version}</version>
 </dependency>
```
## Asking for the GPU

Adding this module puts the CUDA execution provider on the classpath. It does not by itself move
any inference onto it: each component takes an `InferenceOptions` and only uses the GPU when that
says so.

```java
InferenceOptions options = new InferenceOptions();
options.setGpu(true);
options.setGpuDeviceId(0);   // the default, set it for a machine with more than one card
```

Pass it to the component:

```java
// Sentence vectors
new SentenceVectorsDL(model, vocabulary, true, Pooling.MEAN, true,
    SentenceVectorsDL.DEFAULT_MAX_LENGTH, PaddingStrategy.LONGEST, options);

// Document categorization
new DocumentCategorizerDL(model, vocabulary, categories, options);

// Name finding
new NameFinderDL(model, vocabulary, ids2Labels, options, sentenceDetector);
```

Through the text embedder SPI the same request is two options on the spec:

```
onnx:/path/to/model.onnx?vocabulary=vocab.txt&gpu=true&gpuDeviceId=0
```

If the CUDA execution provider cannot be used, construction fails with an `OrtException` rather
than falling back to the CPU. That covers a CPU-only `onnxruntime` on the classpath, a CUDA
installation the runtime cannot load, and a device id no card answers to. The message names which
of those it was.

Batch size, not thread count, is the lever on the GPU: concurrent calls on one session share a
CUDA stream and largely serialize, while `embedAll(...)` under `PaddingStrategy.LONGEST` turns a
whole call into one inference.

### Version pinning

`onnxruntime_gpu` is built against a specific CUDA minor version and will not load an older CUDA
runtime. If `addCUDA` reports `Failed to load shared library`, check the CUDA version on the
machine against the one the `onnxruntime_gpu` release notes name before looking anywhere else.
Never put both `onnxruntime` and `onnxruntime_gpu` on one classpath: whichever loads first wins,
and a CUDA provider library from one build will not link against the `libonnxruntime.so` of the
other.
