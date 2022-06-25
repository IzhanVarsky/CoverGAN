# CoverGAN

<div align="center">
  <img src="./covergan_backend/covergan/examples/gen1_capt1/Boney%20M.%20-%20Rasputin.PNG" alt="img1" width="256"/>
  <img src="./covergan_backend/covergan/examples/gen1_capt1/Charlie%20Puth%20-%20Girlfriend.PNG" alt="img2" width="256"/>
  <img src="./covergan_backend/covergan/examples/gen2_capt2/BOYO%20-%20Dance%20Alone.png" alt="img3" width="256"/>
</div>

**CoverGAN** is a set of tools and machine learning models designed to generate good-looking album covers based on
users'
audio tracks and emotions. Resulting covers are generated in vector graphics format (SVG).

Service is available at http://109.188.135.85:5001/.

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

## Examples of generated covers

See [this](/covergan_backend/covergan/examples) examples folder.

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png

[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg