# OpenNLP DL

This module provides OpenNLP interface implementations for ONNX models using the `onnxruntime` dependency.

**Important**: This does not provide the ability to train models. Model training is done outside of OpenNLP. This code provides the ability to use ONNX models from OpenNLP.

Models used in the tests are available in the [opennlp evaluation test data](https://nightlies.apache.org/opennlp/opennlp-data.zip) location.

## NameFinderDL

`NameFinderDL` runs ONNX token-classification models that use BIO labels. Any
label in the form `B-<TYPE>` starts an entity and subsequent `I-<TYPE>` labels
continue that entity. The text after the prefix is reported as the OpenNLP span
type, for example `B-PER` and `I-PER` produce spans with type `PER`.

The finder uses BERT basic tokenization followed by WordPiece tokenization and
then maps the reconstructed WordPiece text back to the caller's original input
so returned spans can be used with `Span#getCoveredText(...)`. Span probabilities
are normalized from the model logits and are reported in the range `(0, 1]`.

Named entity models are commonly cased, so lower casing is disabled by default.
Set `InferenceOptions#setLowerCase(true)` only for models trained with uncased
input.

### Unicode text handling

Long input is split into overlapping chunks on the full Unicode `White_Space`
set (not Java's `\s`), so no-break space, ideographic space, and the other UCD
whitespace characters are recognized as delimiters. `NameFinderDL` locates
reconstructed entity text in the original input with a cursor-based matcher that
treats span spaces as flexible Unicode whitespace and compares other code points
case-insensitively, so `Span#getCoveredText(...)` works on text from PDFs, the
web, and multilingual sources.

Optional input folding is off by default and controlled through
`InferenceOptions`:

```java
InferenceOptions options = new InferenceOptions();
options.setNormalizeWhitespace(true);  // each Unicode whitespace -> ASCII space (offset-preserving)
options.setNormalizeDashes(true);      // Unicode dashes -> hyphen-minus (offset note below)
NameFinderDL finder = new NameFinderDL(model, vocab, ids2Labels, options, sentenceDetector);
```

Whitespace folding is length-preserving, so it never moves offsets. Dash folding can shrink a
non-BMP dash by one UTF-16 unit, but `NameFinderDL.findInOriginal` maps decoded spans back through
the normalization `Alignment`, so reported spans stay correct in the original input even for
non-BMP dashes. (`NameFinderDL.find` returns normalized-text offsets, which differ from the
original only in that non-BMP-dash case.)

The same options apply to `DocumentCategorizerDL`. The underlying
`CharClass` / `CodePointSet` engine and the broader normalization pipeline live
in `opennlp.tools.util.normalizer` and are documented in the OpenNLP manual
chapter *Text Normalization*.

Export a Hugging Face NER model to ONNX, e.g.:

```bash
python -m transformers.onnx --model=dslim/bert-base-NER --feature token-classification exported
```

## DocumentCategorizerDL

Uses the same Unicode whitespace chunking and optional `InferenceOptions`
normalization as `NameFinderDL` (see above).

Export a Huggingface classification (e.g. sentiment) model to ONNX, e.g.:

```bash
python -m transformers.onnx --model=nlptown/bert-base-multilingual-uncased-sentiment --feature sequence-classification exported
```

## Behavior changes in this release

Integrators upgrading from an earlier `opennlp-dl` should note these intentional changes (OPENNLP-1850):

- `NameFinderDL.find(...)` reports spans in the coordinates of the joined input it ran inference on,
  which differ from the original text only when length-changing dash folding is enabled. Use the new
  `NameFinderDL.findInOriginal(...)` (from `OffsetMappingNameFinder`) for original-text coordinates.
- Spans that overlap at chunk boundaries are now merged longest-wins; `find(...)` previously returned
  every decoded span, overlaps included.
- Chunking splits on the Unicode `White_Space` set rather than `String#split("\\s+")`, and
  whitespace-only input now yields no spans without running the model.
- `DocumentCategorizerDL.categorize(...)` now rejects `null`/empty input, and a document with no
  non-whitespace token, with `IllegalArgumentException` rather than running the model on empty input.
- A `null` input array, a `null` token passed to `NameFinderDL`, or a `null` document passed to
  `DocumentCategorizerDL` is rejected with `IllegalArgumentException` instead of surfacing as a
  `NullPointerException` deep in processing or being joined into the text as the literal string
  `"null"`.
- The example label constants `NameFinderDL.I_PER` and `NameFinderDL.B_PER` were removed; supply your
  own label strings (any `B-<TYPE>`/`I-<TYPE>` pair works, as described above).

## SentenceVectors

Convert a sentence vectors model to ONNX, e.g.:

Install dependencies:

```bash
python3 -m pip install optimum onnx onnxruntime
```

Convert the model:

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path


model_id="sentence-transformers/all-MiniLM-L6-v2"
onnx_path = Path("onnx")

# load vanilla transformers and convert to onnx
model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# save onnx checkpoint and tokenizer
model.save_pretrained(onnx_path)
tokenizer.save_pretrained(onnx_path)
```

## Android

This module does **not** run on Android and produces no Android artifact. What it does hold is a
check that keeps a future Android addon possible: `AndroidReachabilityTest` compiles the ONNX Runtime
facing sources of `opennlp.dl` a second time with `--release 11`, in process, and fails if one of
them has picked up a Java API or a language feature newer than that. Java 11 is the level Android's
desugaring covers, and a source that compiles under a release by definition uses nothing newer.

```bash
./mvnw -o test -pl opennlp-core/opennlp-ml/opennlp-dl -am -Dtest=AndroidReachabilityTest
```

The guarded sources are `OnnxInference`, `InferenceOptions`, `ExecutionProvider`,
`ExecutionProviderConfigurer`, `ExecutionProviderPlacement`, `CpuExecutionProviderConfigurer` and
`CudaExecutionProviderConfigurer`. `OnnxInference` is the one that matters, since it is the single ORT
interaction of the package; the rest are what an addon would have to implement to register an
execution provider. The test also asserts that every other source of the package is on its list of
known exceptions, so a new file cannot slip past unnoticed.

The usual tool for this is animal-sniffer with an Android signature set. It is not used here because
neither `animal-sniffer-maven-plugin` nor any Android signature artifact resolves in an offline
build, and a check that only an online build can run is not a check. The trade is stated in the test:
`--release` catches newer language features as well as newer APIs, which a signature set does not,
and it misses an API that is in Java 11 and absent from Android, which a signature set would catch.

### What an actual Android build would still need

The check is about one property. A real Android addon would have to deal with all of this:

1. **The language level.** This module compiles at the project baseline, JDK 21, so it emits class
   file version 65. An Android build needs a level its toolchain accepts, which means either a
   separate source set for the addon or a module whose `maven.compiler.release` differs from the
   project's. The check does not change what this module targets and must not be read as lowering it.
2. **Four sources of `opennlp.dl` that are not reachable today**, each for a concrete reason:
   `Tokens` is a record; `AbstractDL` holds records and switch rules; `ExecutionProviderRequest` uses
   a pattern `instanceof`; `ExecutionProviders` uses `Stream.toList`, which is Java 16. `Tokens` is
   the awkward one, because every component passes it and its accessors are its API, so replacing the
   record with a class is a visible change rather than a rewrite of a method body.
   `AndroidReachabilityTest` holds this list and requires each of them to really fail under
   `--release 11`, so the list cannot go stale in the optimistic direction.
3. **A different ONNX Runtime artifact.** This module compiles against
   `com.microsoft.onnxruntime:onnxruntime`, a jar carrying desktop native libraries. Android builds
   depend on `com.microsoft.onnxruntime:onnxruntime-android`, published as an AAR, and an AAR is not
   a Maven jar dependency a module like this one can simply add. Which execution providers that
   artifact registers is its own matter, so `ExecutionProviderConfigurer` implementations for an
   Android accelerator would be new addons rather than the CUDA one that ships here.
4. **Service loading under shrinking.** Execution providers and text embedders are resolved through
   `java.util.ServiceLoader`. R8 and resource shrinking drop `META-INF/services` entries unless they
   are kept, so an Android addon needs the keep rules to go with its service files.
