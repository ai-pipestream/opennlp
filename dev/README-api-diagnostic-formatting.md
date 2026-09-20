# API diagnostic formatting

Fixed API diagnostics use direct string construction instead of a general format
interpreter. Keep their wording literal and cover the complete message in tests when
changing one.

Public Unicode reference records provide `toUnicodeNotation()`. For example:

```java
String notation = UnicodeDash.byCodePoint(0x10D6E)
    .orElseThrow()
    .toUnicodeNotation();
// notation is "U+10D6E"
```

Unicode notation uses uppercase hexadecimal, at least four digits, and retains every
digit of supplementary code points.
