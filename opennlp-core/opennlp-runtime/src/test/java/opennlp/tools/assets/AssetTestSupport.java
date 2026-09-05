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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Header fixtures for format detection, not complete image files. */
final class AssetTestSupport {

  /** Prevents construction of the fixture utility. */
  private AssetTestSupport() {
  }

  /**
   * Builds a 45-byte PNG prefix with an IHDR chunk and zero-filled trailing bytes.
   *
   * @param width The declared width.
   * @param height The declared height.
   * @return The header fixture, without a valid CRC or image data.
   */
  static byte[] png(int width, int height) {
    return ByteBuffer.allocate(45)
        .put(new byte[] {(byte) 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A})
        .putInt(13)
        .put(new byte[] {'I', 'H', 'D', 'R'})
        .putInt(width).putInt(height)
        .put(new byte[] {8, 6, 0, 0, 0}).array();
  }

  /**
   * Builds a 30-byte GIF prefix with zero-filled trailing bytes.
   *
   * @param width The declared logical screen width.
   * @param height The declared logical screen height.
   * @return The header fixture, without image data.
   */
  static byte[] gif(int width, int height) {
    return ByteBuffer.allocate(30).order(ByteOrder.LITTLE_ENDIAN)
        .put(new byte[] {'G', 'I', 'F', '8', '9', 'a'})
        .putShort((short) width).putShort((short) height).array();
  }
}
