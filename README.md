# CoverGAN

<div align="center">
  <img src="./examples/gen1_capt1/Boney%20M.%20-%20Rasputin.PNG" alt="img1" width="256"/>
  <img src="./examples/gen1_capt1/Charlie%20Puth%20-%20Girlfriend.PNG" alt="img2" width="256"/>
  <img src="./examples/gen2_capt2/BOYO%20-%20Dance%20Alone.png" alt="img3" width="256"/>
</div>

**CoverGAN** is a set of tools and machine learning models designed to generate good-looking album covers based on users'
audio tracks and emotions. Resulting covers are generated in vector graphics format (SVG).

Available emotions:

* Anger
* Comfortable
* Fear
* Funny
* Happy
* Inspirational
* Joy
* Lonely
* Nostalgic
* Passionate
* Quiet
* Relaxed
* Romantic
* Sadness
* Serious
* Soulful
* Surprise
* Sweet
* Wary

The service is available on http://81.3.154.178:5001/covergan.

## Service functionality

* Generation of music covers by analyzing music and emotions
* Several GAN models
* SVG format
* Possibility of rasterization
* Insertion of readable captions
* A large number of different fonts
* Insertion of different color filters
* SVG editor
* Convenient change of colors
* Style transfer from provided image
* Saving images in any resolution

## Weights

* The pretrained weights can be downloaded
  from [here](https://drive.google.com/file/d/1ArU0TziLBOxhphG4KBshUxPBBECErxu1/view?usp=sharing)
* These weights should be placed into `./weights` folder

## Training

* See [this README](./README.md) for training details.

## Testing using Docker

In this service two types of generator are available:

* The first one creates the covers with abstract lines
* The second one draws closed forms.

It is also possible to use one of two algorithms for applying inscriptions to the cover:

* The first algorithm uses the captioner model
* The second is a deterministic algorithm which searches for a suitable location

The service uses pretrained weights. See [this](README.md#Weights) section.

### Building

* Specify PyTorch version to install in [`Dockerfile`](./Dockerfile).

* Build the image running `docker_build_covergan_service.sh` file

### Running

* Start the container running `docker_run_covergan_service.sh` file

### Testing

Go to `http://localhost:5001` in the browser and enjoy!

## Local testing

### Install dependencies

* Install suitable PyTorch version: `pip install torch torchvision torchaudio`
* Install [DiffVG](https://github.com/BachiLi/diffvg)
* Install dependencies from [this](./requirements.txt) file

### Running

* Run

```sh
python3 ./eval.py \
  --audio_file="test.mp3" \
  --emotions=joy,relaxed \
  --track_artist="Cool Band" \
  --track_name="New Song"
```

* The resulting `.svg` covers by default will be saved to [`./gen_samples`](./covergan/gen_samples) folder.

## Examples of generated covers

See [this](./examples) examples folder.

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

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png

[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg