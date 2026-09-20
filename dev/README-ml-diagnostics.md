# Training diagnostics

Data-indexer duration messages use exact integer milliseconds, a locale-independent
decimal point and two fractional digits with HALF_UP rounding. For example, 1005 ms
is reported as `Done indexing in 1.01 s.` even when the JVM's default locale is German.
`ElapsedTimeFormatterTest` contains executable rounding and locale examples.

Fixed progress and malformed-event diagnostics are built directly. An event such
as `outcome/context;` still raises `IllegalArgumentException` identifying the
incomplete `name;value` field. This branch builds on OPENNLP-1989's event parser;
it changes message construction, not event identities or parsing rules.
