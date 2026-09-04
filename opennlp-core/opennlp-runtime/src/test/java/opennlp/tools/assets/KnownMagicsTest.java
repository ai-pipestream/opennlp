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

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashSet;
import java.util.Set;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the invariants the magic table promises: the precomputed base64 prefixes really
 * are the images of their magic bytes, lookups favor the longest magic, and the format
 * constants of {@link EmbeddedAsset} resolve through the table.
 */
public class KnownMagicsTest {

  /** Every prefix is the base64 image of its magic, floored to whole characters. */
  @Test
  void testPrefixesAreTheFlooredBase64ImageOfTheMagic() {
    for (final KnownMagics.Entry entry : KnownMagics.ENTRIES) {
      final String encoded =
          Base64.getEncoder().withoutPadding().encodeToString(entry.magic());
      assertEquals(entry.magic().length * 8 / 6, entry.prefix().length(),
          entry.format() + ": a magic of n bytes fixes exactly floor(8n/6) characters");
      assertTrue(encoded.startsWith(entry.prefix()),
          entry.format() + ": the prefix must be the image of the magic");
    }
  }

  /** Entries are held longest first, so a lookup yields the most specific match. */
  @Test
  void testEntriesAreSortedLongestFirst() {
    for (int i = 1; i < KnownMagics.ENTRIES.size(); i++) {
      assertTrue(KnownMagics.ENTRIES.get(i - 1).magic().length
              >= KnownMagics.ENTRIES.get(i).magic().length,
          "entry " + i + " breaks the longest-first order");
    }
  }

  /** No two entries carry identical magic bytes. */
  @Test
  void testNoDuplicateMagics() {
    final Set<String> seen = new HashSet<>();
    for (final KnownMagics.Entry entry : KnownMagics.ENTRIES) {
      assertTrue(seen.add(Arrays.toString(entry.magic())),
          "duplicate magic for " + entry.format());
    }
  }

  /** The longest matching magic wins where one magic extends another. */
  @Test
  void testLookupFavorsTheLongestMagic() {
    final byte[] deb = "!<arch>\ndebian-binary junk".getBytes(StandardCharsets.US_ASCII);
    assertEquals("deb", KnownMagics.formatOf(deb));
    final byte[] ar = "!<arch>\nsomething else".getBytes(StandardCharsets.US_ASCII);
    assertEquals("ar", KnownMagics.formatOf(ar));
  }

  /** Every format constant on the record resolves, except the RIFF-carried ones. */
  @Test
  void testEmbeddedAssetConstantsResolve() {
    assertEquals("image/png", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_PNG));
    assertEquals("image/jpeg", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_JPEG));
    assertEquals("image/gif", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_GIF));
    assertEquals("application/pdf", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_PDF));
    assertEquals("application/zip", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_ZIP));
    assertEquals("image/tiff", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_TIFF));
    assertEquals("application/gzip", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_GZIP));
    assertEquals("application/x-7z-compressed",
        KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_SEVEN_ZIP));
    assertEquals("application/vnd.rar", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_RAR));
    assertEquals("audio/flac", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_FLAC));
    assertEquals("application/ogg", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_OGG));
    assertEquals("audio/midi", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_MIDI));
    assertEquals("application/vnd.sqlite3",
        KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_SQLITE));
    assertEquals("application/x-elf", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_ELF));
    assertEquals("application/vnd.microsoft.portable-executable",
        KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_PE));
    assertEquals("application/java-vm", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_CLASS));
    assertEquals("font/woff", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_WOFF));
    assertEquals("font/woff2", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_WOFF2));
    assertEquals("audio/mpeg", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_MP3));
    assertEquals("application/x-ole-storage",
        KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_OLE2));
    assertEquals("application/zstd", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_ZSTD));
    assertEquals("application/wasm", KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_WASM));
    // the RIFF-carried formats are resolved by the detector, not by the table
    assertNull(KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_WEBP));
    assertNull(KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_WAV));
    assertNull(KnownMagics.mediaTypeOf(EmbeddedAsset.FORMAT_AVI));
  }

  /** The table stays clear of prose: no magic under four bytes except the core set. */
  @Test
  void testShortMagicsAreLimitedToTheCoreSet() {
    for (final KnownMagics.Entry entry : KnownMagics.ENTRIES) {
      if (entry.magic().length < 4) {
        assertTrue(entry.format().equals(EmbeddedAsset.FORMAT_JPEG)
                || entry.format().equals(EmbeddedAsset.FORMAT_GZIP)
                || entry.format().equals(EmbeddedAsset.FORMAT_MP3),
            "unexpected short magic for " + entry.format());
      }
    }
  }
}
