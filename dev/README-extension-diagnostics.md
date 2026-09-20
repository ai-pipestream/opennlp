# Extension diagnostics

SymSpell binary model errors use `HexFormat` to display the received and expected
32-bit signatures as eight uppercase hexadecimal digits. For example, an input
signature of zero reports `magic was 0x00000000, expected 0x53594D53`.

The UIMA name finder constructs its invalid type-mapping warning directly,
retaining the original mapping text. Neither diagnostic invokes Java's general
formatter, whose Java 21 implementation uses regex. Model decoding and UIMA
mapping rules are unchanged.
