# tiny-brain

::: tiny-dnn based C++ deep learning applications :::

[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://raw.githubusercontent.com/blackccpie/tiny-brain/master/LICENSE)
-----------------

After spending more than a year working on a from scratch deep learning C++ stack ([neurocl](http://github.com/blackccpie/neurocl)), I decided to switch my experiments to use a more established C++ framework, that is [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn).

tiny-brain additional image processing is implemented in header only [tinymage](https://github.com/blackccpie/tiny-brain/blob/master/tiny_brain/tinymage.h) class, which is based on std::vector private inheritance, and allows some basic image processing (thresholding, resizing, rotating, cropping, computing row/column sum images...).
Some more task specific processing is implemented in header only classes [tinydigit](https://github.com/blackccpie/tiny-brain/blob/master/tiny_brain/tinydigit.h) and [tinysign](https://github.com/blackccpie/tiny-brain/blob/master/tiny_brain/tinysign.h), respectively dedicated to digits recognition and digits sign extraction (from a rich scene).

tiny-brain is **C++14** compliant.

| **`Linux`** |
|-------------|
|[![Build Status](https://travis-ci.org/blackccpie/tiny-brain.svg?branch=master)](https://travis-ci.org/blackccpie/tiny-brain)|
-----------------

****Cloning****

tiny-brain includes tiny-dnn as a _git submodule_, therefore use the following command line for recursive cloning:

```
git clone --recursive https://github.com/blackccpie/tiny-brain.git
```

****Demos****

Full-webassemby mnist digit recognition demo can be viewed online here:

[Digits localization and recognition on white background](http://blackccpie.free.fr/ocr/)

[Digits sign localization, extraction, warping and recognition  (in progress...3/4 correct guesses)](http://blackccpie.free.fr/sign/)

****Status****

Development is on hiatus, while facing some emscripten compilation issue related to [#855](https://github.com/tiny-dnn/tiny-dnn/pull/855) and functional issues related to [#857](https://github.com/tiny-dnn/tiny-dnn/issues/857). Hope these two will be fixed soon.
