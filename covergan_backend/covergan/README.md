## Contents

* `captions/`: a network that predicts aesthetically matching colors and positions for the captions (artist and track
  names).
* `colorer/`: a network that predicts palettes for music covers.
* `docs/`: folder with instructions on how to start training or testing models.
* `examples/`: folder with simple music tracks, their generated covers, and with examples of original and clean
  datasets.
* `fonts/`: folder with downloaded from [Google Fonts](https://fonts.google.com/) fonts.
* `outer/`: the primary GAN that generates vector graphics descriptions from audio files and user-specified emotions.
* `utils/`: parts of code implementing various independent functionality and separated for convenient reuse.
* `weights/`: folder where the best models were saved.
* `captioner_train.py`: an entry point to trigger the Captioner network training.
* `covergan_train.py`: an entry point to trigger the CoverGAN training.
* `eval.py`: an entry point to trigger the primary flow as a command line tool.
* `service.py`: the primary code flow for album cover generation.

## Default structure of dataset folder:

* `audio/`: default folder with music tracks (`.flac` or `.mp3` format) for CoverGAN training.
* `checkpoint/`: default folder where checkpoints and other intermediate files while training CoverGAN and Captioner
  Networks will be stored.
* `clean_covers/`: default folder with covers on which captures were removed.
* `original_covers/`: default folder with original covers.
* `plots/`: the folder where the intermediate plots while training will be saved
* `emotions.json`: file with emotion markup for train dataset.

## Dependencies

* The machine learning models rely on the popular [PyTorch](https://pytorch.org) framework.
* Differentiable vector graphics rendering is provided by [diffvg](https://github.com/BachiLi/diffvg), which needs to be
  built from source.
* Audio feature extraction is based on [Essentia](https://github.com/MTG/essentia), prebuilt pip packages are available.
* Other Python library dependencies include Pillow, Matplotlib, SciPy, and [Kornia](https://kornia.github.io).

## Dataset

The full dataset contains of:

* Audio tracks
* Original covers
* Cleaned covers
* Fonts
* Marked up emotions
* Marked up rectangles for captioner model training

The dataset can be downloaded from [here](https://drive.google.com/file/d/1_NKlS79y29_he9P3xTLd7SgYbOstCkmO/view?usp=sharing)

## Training using Docker with GPU

* Build image running `docker_build.sh`
* See [these](/docs) docs for more details about specified options while training networks.
* Specify training command in `covergan_training_command.sh`
* Start container running `docker_run.sh`
