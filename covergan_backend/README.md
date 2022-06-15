# CoverGAN backend server

## Weights

* The pretrained weights can be downloaded
  from [here](https://drive.google.com/file/d/1ArU0TziLBOxhphG4KBshUxPBBECErxu1/view?usp=sharing)
* These weights should be placed into `./covergan/weights` folder

## Running server
The server is auto-updated when files are changed.

### Building
Run `docker_build_covergan_service.sh`
### Running
Run `docker_run_covergan_service.sh`

## Training and testing

* See [this README](./covergan/README.md) for training and testing details.

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

## Examples of generated covers

See [this](./covergan/examples) examples folder.