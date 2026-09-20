# Corpus reader diagnostics

CoNLL-U, brat and Leipzig reader errors construct their fixed messages directly.
They do not invoke Java's general formatter, which uses regex on Java 21.
Exception types, document identifiers and nested causes are retained.

For example, a CoNLL-U sentence whose text is `café 😀` and whose word line
contains `absent%😀` reports the unmatched token and the original sentence text.
Percent signs and supplementary Unicode characters remain literal diagnostic
content. The `ConlluTokenSampleStreamTest` and `BratAnnotationStreamTest` cases
provide executable examples, including preservation of a brat parser's cause.
