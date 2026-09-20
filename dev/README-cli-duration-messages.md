<!--
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to You under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

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
