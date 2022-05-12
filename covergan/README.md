## Contents

* `audio/`: default folder with music tracks (`.flac` or `.mp3` format) for CoverGAN training.
* `captions/`: a network that predicts aesthetically matching colors and positions for the captions (artist and track
  names).
* `checkpoint/`: default folder where checkpoints and other intermediate files while training CoverGAN and Captioner
  Networks will be stored.
* `clean_covers/`: default folder with covers on which captures were removed.
* `docs/`: folder with instructions on how to start training or testing models.
* `examples/`: folder with simple music tracks, their generated covers, and with examples of original and clean datasets.
* `fonts/`: folder with downloaded from [Google Fonts](https://fonts.google.com/) fonts.
* `gen_samples/`: default folder where generated covers will be saved.
* `original_covers/`: default folder with original covers.
* `outer/`: the primary GAN that generates vector graphics descriptions from audio files and user-specified emotions.
* `protosvg/`: bundled generated Python code for the ProtoSVG format (see the Dependencies section) and a thin Python
  client.
* `test_audio/`: folder with music tracks to test learning.
* `utils/`: parts of code implementing various independent functionality and separated for convenient reuse.
* `weights/`: folder where the best models were saved.
* `captioner_train.py`: an entry point to trigger the Captioner network training.
* `covergan_train.py`: an entry point to trigger the CoverGAN training.
* `emotions.json`: file with emotion markup for train dataset.
* `eval.py`: an entry point to trigger the primary flow as a command line tool.
* `service.py`: the primary code flow for album cover generation.

## Dependencies

### First-Party

* ProtoSVG: a small library built around a simplified SVG subset.
  This library is primarily used by the main code flow to convert internal cover representations into standard SVG
  definitions and render them to raster images. The ProtoSVG gRPC server needs to be started by external process
  management means. The client code is bundled in this repository, complete with a check-in of the auto-generated code
  for convenience.

### Third-party

* The machine learning models rely on the popular [PyTorch](https://pytorch.org) framework.
* Differentiable vector graphics rendering is provided by [diffvg](https://github.com/BachiLi/diffvg), which needs to be
  built from source.
* Audio feature extraction is based on [Essentia](https://github.com/MTG/essentia), prebuilt pip packages are available.
* Other Python library dependencies include Pillow, Matplotlib, SciPy, and [Kornia](https://kornia.github.io).
