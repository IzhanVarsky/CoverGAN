# CoverGAN

CoverGAN is a set of tools and machine learning models designed to generate good-looking album covers based on users'
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

CoverGAN can be run on a machine without a GPU, however, if a GPU is available, the model will use it.

This tutorial is for Linux.

## Dependencies (for local test and training)

Install PyTorch. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.

Install other python requirements:

* `pip install -r requirements.txt`

Install DiffVG:

* Make sure that CMake is installed. See [https://cmake.org/install/](https://cmake.org/install/) for CMake install
  instructions.
* `git clone --recursive https://github.com/BachiLi/diffvg`
* `cd diffvg && python setup.py install`
* `cd .. && rm -rf diffvg`

## Testing using Docker

### Building

* Specify PyTorch version to install in [`Dockerfile`](./Dockerfile) .

* Build the image:

```sh
docker build -t covergan .
```

### Running

* Run the container:

With CUDA enabled:

```sh
docker run --rm --network="host" --gpus 1 covergan
```

Only CPU using:

```sh
docker run --rm --network="host" covergan
```

### Testing

Below is an example command that can be used to trigger the generation endpoint:

```sh
curl --progress-bar \
    -F "audio_file=@/home/user/audio.flac" \
    "http://localhost:5001/generate?track_artist=Cool%20Band&track_name=Song&emotion=joy" \
    -o ./output.json
```

## Local testing

### Compile ProtoSVG server

In [`protosvg`](./protosvg) folder:

* `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` (type `yes`, after `1`)
* `source $HOME/.cargo/env`
* `rustup component add rustfmt`
* `cargo install --locked --path .`

* Run as background process the ProtoSVG server by command `protosvg`

### Running

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

## Local training

Main directory is [`covergan`](./covergan).

### CoverGAN network training

* Put audio tracks (`.flac` or `.mp3` format) to default [`./audio`](./covergan/audio) folder.
* Put original covers (`.jpg` format) to default [`./clean_covers`](./covergan/clean_covers) folder.
* See this [help document](./covergan/docs/covergan_train_help.txt) for more details about specified options.
* Run `./covergan_train.py` file with specified options.

Example:
`python3 ./covergan_train.py --emotions emotions.json`

### Captioner network training

* Put original covers (`.jpg` format) in [`./original_covers`](./covergan/original_covers) folder.
* If the covers title and author name have been already saved to `data.json` file (which for each cover stores the
  coordinates of the captures and their text color), it should be stored
  at [`./checkpoint/caption_dataset/data.json`](./covergan/checkpoint/caption_dataset/data.json).
* Or else put clean (with captures removed) covers (`.jpg` format) to [`./clean_covers`](./covergan/clean_covers)
  folder.
* See this [help document](./covergan/docs/captioner_train_help.txt) for more details about specified options.
* Run `./captioner_train.py` file with specified options.

Example:
`python3 ./captioner_train.py --clean_covers ./clean_covers`

## Examples of generated covers

See [this](./covergan/examples/tracks_with_generated_covers) folder with simple music tracks and their generated covers.

