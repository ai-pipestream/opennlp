/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
package example;

import java.io.Console;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.nio.file.DirectoryStream;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.util.Formatter;
import java.util.Locale;
import java.util.Scanner;
import java.util.function.BiFunction;
import java.util.function.Function;

final class JdkRegexFacades {
  Object[] calls(Path path, PrintStream stream, PrintWriter writer, Console console) throws IOException {
    DirectoryStream<Path> paths = Files.newDirectoryStream(path, "*.bin");
    PathMatcher matcher = path.getFileSystem().getPathMatcher("glob:*.bin");
    Scanner scanner = new Scanner("token");
    Formatter formatter = new Formatter().format("%s", "value");
    String first = String.format("%s", "value");
    String localized = String.format(Locale.ROOT, "%s", "value");
    String instance = "%s".formatted("value");
    stream.printf("%s", "value");
    stream.printf(Locale.ROOT, "%s", "value");
    stream.format("%s", "value");
    stream.format(Locale.ROOT, "%s", "value");
    writer.printf("%s", "value");
    writer.printf(Locale.ROOT, "%s", "value");
    writer.format("%s", "value");
    writer.format(Locale.ROOT, "%s", "value");
    console.printf("%s", "value");
    console.format("%s", "value");
    return new Object[] {paths, matcher, scanner, formatter, first, localized, instance};
  }

  Object[] references(FileSystem fileSystem, PrintStream stream) {
    Function<String, PathMatcher> boundMatcher = fileSystem::getPathMatcher;
    BiFunction<FileSystem, String, PathMatcher> unboundMatcher = FileSystem::getPathMatcher;
    FormatFunction staticFormat = String::format;
    StreamFormatFunction boundPrintf = stream::printf;
    UnboundStreamFormatFunction unboundPrintf = PrintStream::printf;
    return new Object[] {boundMatcher, unboundMatcher, staticFormat, boundPrintf, unboundPrintf};
  }

  interface FormatFunction {
    String apply(String format, Object... args);
  }

  interface StreamFormatFunction {
    PrintStream apply(String format, Object... args);
  }

  interface UnboundStreamFormatFunction {
    PrintStream apply(PrintStream stream, String format, Object... args);
  }
}
