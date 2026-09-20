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
package opennlp.tools.models.simple;

import java.util.stream.Stream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import opennlp.tools.models.AbstractClassPathFinderTest;
import opennlp.tools.models.ClassPathModelFinder;

public class SimpleClassPathModelFinderTest extends AbstractClassPathFinderTest {

  private record TestArguments(String classPath, String[] expected) {
    static TestArguments of(String classPath, String[] expected) {
      return new TestArguments(classPath, expected);
    }
  }


  @Override
  protected ClassPathModelFinder getModelFinder() {
    return new SimpleClassPathModelFinder();
  }

  @Override
  protected ClassPathModelFinder getModelFinder(String pattern) {
    return new SimpleClassPathModelFinder(pattern);
  }

  private static Stream<TestArguments> unixClassPaths() {
    return Stream.of(
        TestArguments.of("", new String[0]),
        TestArguments.of(":", new String[0]),
        TestArguments.of("::", new String[0]),
        TestArguments.of("a.jar", new String[] {"a.jar"}),
        TestArguments.of("a.jar:b.jar", new String[] {"a.jar", "b.jar"}),
        // empty entries are skipped wherever they appear
        TestArguments.of(":a.jar", new String[] {"a.jar"}),
        TestArguments.of("a.jar:", new String[] {"a.jar"}),
        TestArguments.of("a.jar::", new String[] {"a.jar"}),
        TestArguments.of("a.jar::b.jar", new String[] {"a.jar", "b.jar"}),
        TestArguments.of("::a.jar:", new String[] {"a.jar"}),
        TestArguments.of(" :a.jar", new String[] {" ", "a.jar"}),
        TestArguments.of("/usr/lib/a.jar:/opt/b.jar", new String[] {"/usr/lib/a.jar", "/opt/b.jar"}),
        TestArguments.of("C:\\lib\\a.jar;C:\\lib\\b.jar",
            new String[] {"C", "\\lib\\a.jar;C", "\\lib\\b.jar"}),
        TestArguments.of("C:/lib/a.jar", new String[] {"C", "/lib/a.jar"}),
        TestArguments.of("\uD801\uDC12.jar:b.jar", new String[] {"\uD801\uDC12.jar", "b.jar"}),
        TestArguments.of("\t:a.jar", new String[] {"\t", "a.jar"}),
        TestArguments.of("my%20lib/a.jar:my lib/b.jar", new String[] {"my%20lib/a.jar", "my lib/b.jar"}),
        TestArguments.of("a.jar!/x:b.jar", new String[] {"a.jar!/x", "b.jar"}),
        TestArguments.of("/lib/*:/opt/*.jar", new String[] {"/lib/*", "/opt/*.jar"}),
        TestArguments.of("a.jar\nb.jar", new String[] {"a.jar\nb.jar"}),
        TestArguments.of("a.jar\n:b.jar", new String[] {"a.jar\n", "b.jar"}));
  }

  @Test
  void testSplitClassPathUnix() {
    Assertions.assertAll(unixClassPaths().map(arguments -> () -> Assertions.assertArrayEquals(
        arguments.expected(), SimpleClassPathModelFinder.splitClassPath(arguments.classPath(), false))));
  }

  private static Stream<TestArguments> windowsClassPaths() {
    return Stream.of(
        TestArguments.of("", new String[0]),
        TestArguments.of(";", new String[0]),
        TestArguments.of(";;", new String[0]),
        TestArguments.of("a.jar", new String[] {"a.jar"}),
        TestArguments.of("a.jar;b.jar", new String[] {"a.jar", "b.jar"}),
        // empty entries are skipped wherever they appear
        TestArguments.of(";a.jar", new String[] {"a.jar"}),
        TestArguments.of("a.jar;", new String[] {"a.jar"}),
        TestArguments.of("a.jar;;", new String[] {"a.jar"}),
        TestArguments.of("a.jar;;b.jar", new String[] {"a.jar", "b.jar"}),
        TestArguments.of(";;a.jar;", new String[] {"a.jar"}),
        TestArguments.of("C:\\lib\\a.jar;C:\\lib\\b.jar", new String[] {"C:\\lib\\a.jar", "C:\\lib\\b.jar"}),
        TestArguments.of("/usr/lib/a.jar:/opt/b.jar", new String[] {"/usr/lib/a.jar:/opt/b.jar"}),
        TestArguments.of("C:\\a.jar;D:\\b.jar", new String[] {"C:\\a.jar", "D:\\b.jar"}),
        TestArguments.of("C:/a.jar;d:/b.jar", new String[] {"C:/a.jar", "d:/b.jar"}),
        TestArguments.of("\\\\server\\share\\a.jar;b.jar",
            new String[] {"\\\\server\\share\\a.jar", "b.jar"}),
        TestArguments.of("C:\\my lib\\a.jar; ;b.jar", new String[] {"C:\\my lib\\a.jar", " ", "b.jar"}),
        TestArguments.of("C:\\my%20lib\\a.jar", new String[] {"C:\\my%20lib\\a.jar"}),
        TestArguments.of("a.jar!/x;b.jar", new String[] {"a.jar!/x", "b.jar"}),
        TestArguments.of("C:\\lib\\*;D:\\*.jar", new String[] {"C:\\lib\\*", "D:\\*.jar"}),
        TestArguments.of("\uD801\uDC12.jar;b.jar", new String[] {"\uD801\uDC12.jar", "b.jar"}));
  }

  @Test
  void testSplitClassPathWindows() {
    Assertions.assertAll(windowsClassPaths().map(arguments -> () -> Assertions.assertArrayEquals(
        arguments.expected(), SimpleClassPathModelFinder.splitClassPath(arguments.classPath(), true))));
  }

  @Test
  void testSplitClassPathRejectsNull() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> SimpleClassPathModelFinder.splitClassPath(null, false));
  }
}
