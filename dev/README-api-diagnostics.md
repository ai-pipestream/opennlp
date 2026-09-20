# API diagnostics

Fixed API diagnostics construct their text directly. Invalid argument values,
provider names and annotation text are diagnostic data, including literal percent
signs and supplementary Unicode characters.

The Unicode whitespace and dash APIs render code points with uppercase hexadecimal
notation and at least four digits. For example, the non-breaking space is `U+00A0`.
The shared internal renderer also keeps all digits for supplementary code points,
such as `U+1F600`, rather than truncating them to a UTF-16 code unit.

`UnicodeNotationTest` and the existing public whitespace/dash API tests provide
executable examples. Rendering uses the JDK's typed `HexFormat` API, not the
general formatter whose Java 21 implementation uses regex.
