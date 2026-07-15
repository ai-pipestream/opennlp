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

package opennlp.tools.tokenize.lattice;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.zip.GZIPInputStream;

import opennlp.tools.util.archive.TarStream;

/**
 * Fetches and unpacks a mecab-format dictionary archive into a local directory, so the
 * dictionary is acquired by the user at install time and never ships with this library.
 * The user supplies the archive location from the dictionary project of their choice
 * and thereby accepts that dictionary's license; nothing is bundled and no location is
 * built in.
 *
 * <p>The installer reads gzip-compressed tar archives, the format the common
 * distributions use, and extracts only the files a {@link MecabDictionary} reads:
 * {@code *.csv}, {@code *.def}, and {@code dicrc}. Entries are flattened to their base
 * names, which also means no archive path can escape the target directory.</p>
 *
 * @see opennlp.tools.util.ResourceInstaller ResourceInstaller, the general tool for
 *      user-supplied resources
 * @since 3.0.0
 */
public final class MecabDictionaryInstaller {

  private MecabDictionaryInstaller() {
    // static installer only
  }

  /**
   * Downloads a dictionary archive and unpacks it.
   *
   * @param archive The archive location, a gzip-compressed tar. Must not be
   *                {@code null}.
   * @param targetDirectory The directory to unpack into; created when absent. Must not
   *                        be {@code null}.
   * @return The number of dictionary files extracted.
   * @throws IOException Thrown if fetching, reading, or writing fails, or the archive
   *         contains no dictionary file.
   * @throws IllegalArgumentException Thrown if a parameter is {@code null}.
   */
  public static int install(URI archive, Path targetDirectory) throws IOException {
    if (archive == null || targetDirectory == null) {
      throw new IllegalArgumentException("archive and targetDirectory must not be null");
    }
    try (InputStream in = archive.toURL().openStream()) {
      return extract(in, targetDirectory);
    }
  }

  /**
   * Unpacks a dictionary archive stream.
   *
   * @param archiveStream The gzip-compressed tar content. Must not be {@code null}.
   *                      Not closed.
   * @param targetDirectory The directory to unpack into; created when absent. Must not
   *                        be {@code null}.
   * @return The number of dictionary files extracted.
   * @throws IOException Thrown if reading or writing fails, or the archive contains no
   *         dictionary file.
   * @throws IllegalArgumentException Thrown if a parameter is {@code null}.
   */
  public static int extract(InputStream archiveStream, Path targetDirectory)
      throws IOException {
    if (archiveStream == null || targetDirectory == null) {
      throw new IllegalArgumentException("stream and targetDirectory must not be null");
    }
    Files.createDirectories(targetDirectory);
    final TarStream entries = new TarStream(new GZIPInputStream(archiveStream));
    int extracted = 0;
    while (entries.next()) {
      if (!entries.isFile()) {
        continue;
      }
      final String baseName = baseName(entries.name());
      if (baseName.endsWith(".csv") || baseName.endsWith(".def")
          || "dicrc".equals(baseName)) {
        Files.copy(entries.entryStream(), targetDirectory.resolve(baseName),
            StandardCopyOption.REPLACE_EXISTING);
        extracted++;
      }
    }
    if (extracted == 0) {
      throw new IOException("the archive contains no dictionary file");
    }
    return extracted;
  }

  private static String baseName(String name) {
    final int slash = name.lastIndexOf('/');
    return slash < 0 ? name : name.substring(slash + 1);
  }
}
