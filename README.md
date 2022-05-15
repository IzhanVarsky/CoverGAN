# CoverGAN

<div align="center">
  <img src="/covergan/examples/gen1_capt1/Boney%20M.%20-%20Rasputin.PNG" alt="img1" width="256"/>
  <img src="/covergan/examples/gen1_capt1/Charlie%20Puth%20-%20Girlfriend.PNG" alt="img2" width="256"/>
  <img src="/covergan/examples/gen2_capt2/BOYO%20-%20Dance%20Alone.png" alt="img3" width="256"/>
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

The service is available on http://109.188.135.85:5001.

## Weights

* The pretrained weights can be downloaded
  from [here](https://drive.google.com/file/d/1ArU0TziLBOxhphG4KBshUxPBBECErxu1/view?usp=sharing)
* These weights should be placed into `/covergan/weights` folder

## Training

* See [this README](./covergan/README.md) for training details.

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
* Install dependencies from [this](/covergan/requirements.txt) file

### Compile ProtoSVG server

In [`protosvg`](./protosvg) folder:

* `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` (type `yes`, after `1`)
* `source $HOME/.cargo/env`
* `rustup component add rustfmt`
* `cargo install --locked --path .`

### Running

Run as background process the ProtoSVG server by command `protosvg`

In [`covergan`](./covergan) folder:

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

See [this](./covergan/examples) examples folder.

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png

[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg