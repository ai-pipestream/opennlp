# CLI duration messages

CLI execution, model loading and model writing report elapsed milliseconds as
seconds with three fractional digits and a decimal point. For example, 5 ms is
`0.005` seconds under both English and German JVM locales.

The conversion uses `BigDecimal.valueOf(milliseconds, 3).toPlainString()` so it
preserves integer millisecond precision without a floating-point conversion or
Java's general formatter. The latter uses regex on Java 21. This only changes
display; the existing clock and duration measurement are unchanged.

`ModelLoaderDurationTest` loads a real temporary input through the public loader
method under a German formatting locale and checks the resulting log message.
