/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package opennlp.tools.assets;

import java.io.ByteArrayOutputStream;
import java.util.Base64;
import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the built-in detector against real header bytes built in the test, so every
 * accepted case is a genuine file prefix and every rejected case fails for the reason
 * the class documents.
 */
public class CursorAssetDetectorTest {

  private final CursorAssetDetector detector = new CursorAssetDetector();

  /**
   * Builds a PNG prefix: signature plus an IHDR chunk declaring the dimensions,
   * padded so the payload is comfortably long.
   *
   * @param width The declared width.
   * @param height The declared height.
   * @return The leading bytes of a PNG file.
   */
  private static byte[] png(int width, int height) {
    final ByteArrayOutputStream out = new ByteArrayOutputStream();
    out.writeBytes(new byte[] {(byte) 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A});
    out.writeBytes(new byte[] {0, 0, 0, 13});
    out.writeBytes(new byte[] {'I', 'H', 'D', 'R'});
    out.writeBytes(new byte[] {(byte) (width >>> 24), (byte) (width >>> 16),
        (byte) (width >>> 8), (byte) width});
    out.writeBytes(new byte[] {(byte) (height >>> 24), (byte) (height >>> 16),
        (byte) (height >>> 8), (byte) height});
    out.writeBytes(new byte[] {8, 6, 0, 0, 0});
    out.writeBytes(new byte[16]);
    return out.toByteArray();
  }

  /**
   * Builds a GIF prefix with little-endian dimensions.
   *
   * @param width The declared width.
   * @param height The declared height.
   * @return The leading bytes of a GIF file.
   */
  private static byte[] gif(int width, int height) {
    final ByteArrayOutputStream out = new ByteArrayOutputStream();
    out.writeBytes(new byte[] {'G', 'I', 'F', '8', '9', 'a'});
    out.writeBytes(new byte[] {(byte) width, (byte) (width >>> 8),
        (byte) height, (byte) (height >>> 8)});
    out.writeBytes(new byte[20]);
    return out.toByteArray();
  }

  /**
   * Builds a RIFF prefix.
   *
   * @param kind The four-character form type, {@code WEBP} or {@code WAVE}.
   * @return The leading bytes of a RIFF file.
   */
  private static byte[] riff(String kind) {
    final ByteArrayOutputStream out = new ByteArrayOutputStream();
    out.writeBytes(new byte[] {'R', 'I', 'F', 'F', 100, 0, 0, 0});
    out.writeBytes(kind.getBytes(java.nio.charset.StandardCharsets.US_ASCII));
    out.writeBytes(new byte[20]);
    return out.toByteArray();
  }

  /**
   * Pads bytes to at least the length the detector demands of a bare run once
   * encoded.
   *
   * @param prefix The leading bytes.
   * @return The bytes padded to 30, so the base64 form exceeds the bare minimum.
   */
  private static byte[] padded(byte[] prefix) {
    final ByteArrayOutputStream out = new ByteArrayOutputStream();
    out.writeBytes(prefix);
    if (out.size() < 30) {
      out.writeBytes(new byte[30 - out.size()]);
    }
    return out.toByteArray();
  }

  @Test
  void testBarePngRunWithDimensions() {
    final byte[] bytes = png(640, 480);
    final String encoded = Base64.getEncoder().encodeToString(bytes);
    final String text = "before " + encoded + " after";
    final List<EmbeddedAsset> assets = detector.detect(text);
    assertEquals(1, assets.size());
    final EmbeddedAsset asset = assets.get(0);
    assertEquals(EmbeddedAsset.FORMAT_PNG, asset.format());
    assertEquals("image/png", asset.mediaType());
    assertEquals(640, asset.width());
    assertEquals(480, asset.height());
    assertEquals(bytes.length, asset.decodedLength());
    assertEquals(encoded, asset.span().getCoveredText(text).toString());
    assertArrayEquals(bytes, asset.decode(text));
  }

  @Test
  void testDataUriDeclaresTheMediaType() {
    final byte[] bytes = png(2, 3);
    final String uri = "data:image/png;base64," + Base64.getEncoder().encodeToString(bytes);
    final String text = "<img src=\"" + uri + "\">";
    final List<EmbeddedAsset> assets = detector.detect(text);
    assertEquals(1, assets.size());
    final EmbeddedAsset asset = assets.get(0);
    assertEquals(EmbeddedAsset.FORMAT_PNG, asset.format());
    assertEquals("image/png", asset.mediaType());
    assertEquals(uri, asset.span().getCoveredText(text).toString());
    assertArrayEquals(bytes, asset.decode(text));
  }

  @Test
  void testDataUriWithUnknownHeaderKeepsTheDeclaredFormat() {
    final byte[] bytes = "just some plain bytes padded out".getBytes(
        java.nio.charset.StandardCharsets.US_ASCII);
    final String uri = "data:text/plain;base64," + Base64.getEncoder().encodeToString(bytes);
    final List<EmbeddedAsset> assets = detector.detect("x " + uri + " y");
    assertEquals(1, assets.size());
    assertEquals("plain", assets.get(0).format());
    assertEquals("text/plain", assets.get(0).mediaType());
  }

  @Test
  void testGifJpegRiffPdfZipAllIdentify() {
    final String gif = Base64.getEncoder().encodeToString(padded(gif(12, 34)));
    final String jpeg = Base64.getEncoder().encodeToString(
        padded(new byte[] {(byte) 0xFF, (byte) 0xD8, (byte) 0xFF, (byte) 0xE0}));
    final String webp = Base64.getEncoder().encodeToString(riff("WEBP"));
    final String wave = Base64.getEncoder().encodeToString(riff("WAVE"));
    final String avi = Base64.getEncoder().encodeToString(riff("AVI "));
    final String pdf = Base64.getEncoder().encodeToString(
        padded("%PDF-1.7 stub content".getBytes(
            java.nio.charset.StandardCharsets.US_ASCII)));
    final String zip = Base64.getEncoder().encodeToString(
        padded(new byte[] {'P', 'K', 3, 4}));
    final String text = String.join("\n", gif, jpeg, webp, wave, avi, pdf, zip);
    final List<EmbeddedAsset> assets = detector.detect(text);
    assertEquals(7, assets.size());
    assertEquals(EmbeddedAsset.FORMAT_GIF, assets.get(0).format());
    assertEquals(12, assets.get(0).width());
    assertEquals(34, assets.get(0).height());
    assertEquals(EmbeddedAsset.FORMAT_JPEG, assets.get(1).format());
    assertEquals(-1, assets.get(1).width());
    assertEquals(EmbeddedAsset.FORMAT_WEBP, assets.get(2).format());
    assertEquals(EmbeddedAsset.FORMAT_WAV, assets.get(3).format());
    assertEquals(EmbeddedAsset.FORMAT_AVI, assets.get(4).format());
    assertEquals("video/x-msvideo", assets.get(4).mediaType());
    assertEquals(EmbeddedAsset.FORMAT_PDF, assets.get(5).format());
    assertEquals(EmbeddedAsset.FORMAT_ZIP, assets.get(6).format());
  }

  /**
   * The header bytes of every remaining recognized format, each the format's magic as
   * its specification publishes it.
   *
   * @return One case per format: the expected format tag, media type, and header.
   */
  private static Stream<Arguments> knownFormatHeaders() {
    final ByteArrayOutputStream sqlite = new ByteArrayOutputStream();
    sqlite.writeBytes("SQLite format 3".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
    sqlite.write(0);
    return Stream.of(
        Arguments.of(EmbeddedAsset.FORMAT_TIFF, "image/tiff",
            new byte[] {'I', 'I', 0x2A, 0x00}),
        Arguments.of(EmbeddedAsset.FORMAT_TIFF, "image/tiff",
            new byte[] {'M', 'M', 0x00, 0x2A}),
        Arguments.of(EmbeddedAsset.FORMAT_GZIP, "application/gzip",
            new byte[] {0x1F, (byte) 0x8B, 0x08}),
        Arguments.of(EmbeddedAsset.FORMAT_SEVEN_ZIP, "application/x-7z-compressed",
            new byte[] {'7', 'z', (byte) 0xBC, (byte) 0xAF, 0x27, 0x1C}),
        Arguments.of(EmbeddedAsset.FORMAT_RAR, "application/vnd.rar",
            new byte[] {'R', 'a', 'r', '!', 0x1A, 0x07, 0x00}),
        Arguments.of(EmbeddedAsset.FORMAT_FLAC, "audio/flac",
            new byte[] {'f', 'L', 'a', 'C'}),
        Arguments.of(EmbeddedAsset.FORMAT_OGG, "application/ogg",
            new byte[] {'O', 'g', 'g', 'S'}),
        Arguments.of(EmbeddedAsset.FORMAT_MIDI, "audio/midi",
            new byte[] {'M', 'T', 'h', 'd'}),
        Arguments.of(EmbeddedAsset.FORMAT_SQLITE, "application/vnd.sqlite3",
            sqlite.toByteArray()),
        Arguments.of(EmbeddedAsset.FORMAT_ELF, "application/x-elf",
            new byte[] {0x7F, 'E', 'L', 'F', 2, 1, 1}),
        Arguments.of(EmbeddedAsset.FORMAT_PE, "application/vnd.microsoft.portable-executable",
            new byte[] {'M', 'Z', (byte) 0x90, 0x00}),
        Arguments.of(EmbeddedAsset.FORMAT_CLASS, "application/java-vm",
            new byte[] {(byte) 0xCA, (byte) 0xFE, (byte) 0xBA, (byte) 0xBE, 0, 0, 0, 0x37}),
        Arguments.of(EmbeddedAsset.FORMAT_WOFF, "font/woff",
            new byte[] {'w', 'O', 'F', 'F'}),
        Arguments.of(EmbeddedAsset.FORMAT_WOFF2, "font/woff2",
            new byte[] {'w', 'O', 'F', '2'}),
        Arguments.of(EmbeddedAsset.FORMAT_MP3, "audio/mpeg",
            new byte[] {'I', 'D', '3', 3, 0, 0}),
        Arguments.of(EmbeddedAsset.FORMAT_OLE2, "application/x-ole-storage",
            new byte[] {(byte) 0xD0, (byte) 0xCF, 0x11, (byte) 0xE0,
                (byte) 0xA1, (byte) 0xB1, 0x1A, (byte) 0xE1}),
        Arguments.of(EmbeddedAsset.FORMAT_ZSTD, "application/zstd",
            new byte[] {0x28, (byte) 0xB5, 0x2F, (byte) 0xFD}),
        Arguments.of(EmbeddedAsset.FORMAT_WASM, "application/wasm",
            new byte[] {0, 'a', 's', 'm', 1, 0, 0, 0}),
        Arguments.of("rtf", "application/rtf",
            "{\\rtf1\\ansi minimal".getBytes(java.nio.charset.StandardCharsets.US_ASCII)),
        Arguments.of("xz", "application/x-xz",
            new byte[] {(byte) 0xFD, '7', 'z', 'X', 'Z', 0}),
        Arguments.of("matroska", "application/x-matroska",
            new byte[] {0x1A, 0x45, (byte) 0xDF, (byte) 0xA3}),
        Arguments.of("psd", "image/vnd.adobe.photoshop",
            new byte[] {'8', 'B', 'P', 'S', 0, 1}),
        Arguments.of("ico", "image/vnd.microsoft.icon",
            new byte[] {0, 0, 1, 0}),
        Arguments.of("macho", "application/x-mach-o",
            new byte[] {(byte) 0xFE, (byte) 0xED, (byte) 0xFA, (byte) 0xCE}),
        Arguments.of("pem-cert", "application/x-x509-cert",
            "-----BEGIN CERTIFICATE-----".getBytes(
                java.nio.charset.StandardCharsets.US_ASCII)),
        Arguments.of("torrent", "application/x-bittorrent",
            "d8:announce27:http".getBytes(java.nio.charset.StandardCharsets.US_ASCII)),
        Arguments.of("php", "text/x-php",
            "<?php echo 'shell';".getBytes(java.nio.charset.StandardCharsets.US_ASCII)),
        Arguments.of("parquet", "application/x-parquet",
            new byte[] {'P', 'A', 'R', '1'}));
  }

  @ParameterizedTest
  @MethodSource("knownFormatHeaders")
  void testExpandedMagicTableIdentifies(String format, String mediaType, byte[] header) {
    final String encoded = Base64.getEncoder().encodeToString(padded(header));
    final List<EmbeddedAsset> assets = detector.detect("x " + encoded + " y");
    assertEquals(1, assets.size());
    assertEquals(format, assets.get(0).format());
    assertEquals(mediaType, assets.get(0).mediaType());
    assertEquals(encoded, assets.get(0).span().getCoveredText("x " + encoded + " y").toString());
  }

  /**
   * A TIFF-looking run whose fourth byte breaks the magic: the base64 prefix pins only
   * the first two bits of that byte, so the decoded header check must reject it.
   */
  @Test
  void testTiffPrefixWithBrokenMagicByteIsIgnored() {
    final String lookalike = Base64.getEncoder().encodeToString(
        padded(new byte[] {'I', 'I', 0x2A, 0x01}));
    assertTrue(lookalike.startsWith("SUkqA"));
    assertEquals(List.of(), detector.detect("x " + lookalike + " y"));
  }

  /**
   * A PE-looking run whose fourth byte breaks the DOS-stub magic: the prefix matches,
   * the decoded header check rejects.
   */
  @Test
  void testPePrefixWithBrokenStubByteIsIgnored() {
    final String lookalike = Base64.getEncoder().encodeToString(
        padded(new byte[] {'M', 'Z', (byte) 0x90, 0x01}));
    assertTrue(lookalike.startsWith("TVqQA"));
    assertEquals(List.of(), detector.detect("x " + lookalike + " y"));
  }

  /** A short base64-looking token, such as a URL path segment, is never an asset. */
  @Test
  void testShortRunsAreIgnored() {
    assertEquals(List.of(), detector.detect("GET /9j/abc/thumbnail HTTP/1.1"));
  }

  /** A long run without a known magic prefix is not an asset. */
  @Test
  void testLongRunWithoutMagicIsIgnored() {
    final String encoded = Base64.getEncoder().encodeToString(
        "this is just text encoded as base64 for transport".getBytes(
            java.nio.charset.StandardCharsets.US_ASCII));
    assertEquals(List.of(), detector.detect("payload: " + encoded));
  }

  /** A magic prefix whose decoded header does not verify is not an asset. */
  @Test
  void testMagicPrefixWithoutVerifyingHeaderIsIgnored() {
    // A RIFF container whose form type is neither WEBP nor WAVE: the payload starts
    // with the RIFF prefix characters, but the header check rejects it.
    final ByteArrayOutputStream out = new ByteArrayOutputStream();
    out.writeBytes(new byte[] {'R', 'I', 'F', 'F', 100, 0, 0, 0});
    out.writeBytes(new byte[] {'J', 'U', 'N', 'K'});
    out.writeBytes(new byte[20]);
    final String lookalike = Base64.getEncoder().encodeToString(out.toByteArray());
    assertTrue(lookalike.startsWith("UklGR"));
    assertEquals(List.of(), detector.detect("x " + lookalike + " y"));
  }

  /** An ordinary paragraph, with words of base64 characters, yields nothing. */
  @Test
  void testPlainProseYieldsNothing() {
    assertEquals(List.of(), detector.detect(
        "Plain prose with ordinary words does not contain payloads, "
            + "not even AlphanumericRunsLikeThisOne0123456789 of some length."));
  }

  /** A truncated payload is trimmed to a decodable length instead of failing later. */
  @Test
  void testTruncatedPayloadStaysDecodable() {
    final byte[] bytes = padded(png(1, 1));
    final String encoded = Base64.getEncoder().withoutPadding().encodeToString(bytes);
    // Cut so the run length is 1 mod 4, the length no base64 payload can end at.
    final int cut = ((encoded.length() - 4) / 4) * 4 + 1;
    final String truncated = encoded.substring(0, cut);
    final List<EmbeddedAsset> assets = detector.detect(truncated);
    assertEquals(1, assets.size());
    assertEquals(cut - 1, assets.get(0).payload().getEnd());
    assertTrue(assets.get(0).decode(truncated).length > 0);
  }

  @Test
  void testRejectsNull() {
    assertThrows(IllegalArgumentException.class, () -> detector.detect(null));
  }
}
