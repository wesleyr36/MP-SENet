This source code is modified from MP-SENet.

## Pre-requisites
1. Python >= 3.6.
2. Clone this repository.
3. Install python requirements. These aren't the full requirements for some reason so they'll be updated
4. A dataset, you can usually make this yourself, I may provide some dataset creation scrips in the future.

## Training
```
python train.py --config {config_name}.json
```
Checkpoints and a copy of the config file are currently saved in the `cp_mpsenet-1024-declip` directory by default.<br>
You can change the path by adding the `--checkpoint_path` option.
The code by default scans your system for all availbale GPUs to train on but if you're using your personal machine and you happen to have 2 GPUs (let's say an RTX 2070 and a Quadro P4000) and want to still be able use your machine whilst training during the day, you can use the `--solo_GPU` flag followed by the ID of the GPU i.e. `--solo_GPU 1` for just the Quadro P4000 to train on just that GPU.

## Inference
```
python inference.py --checkpoint_file [generator checkpoint file path]
```
You can also use the pretrained checkpoint file I've provided `cp_mpsenet-1024-{task}/g_{steps}` which will be updated when I can be assed.<br>
Generated wav files are saved in `generated_files` by default.<br>
You can change the path by adding `--output_dir` option.

I've modified the code to implement chunking to allow for audio files of any kind of length to be processed.
Currently by default that chunking splits it into 3s chunks with 2s of overlap between chunks to avoid clicks, the is technically modifiable based on how much VRAM you have but, there is a small issue with it being a number of samples off. I've bodged it for now but that bodge only works for the current settings, changing it *will* cause phasing issues.

## Citation
```
@inproceedings{lu2023mp,
  title={{MP-SENet}: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra},
  author={Lu, Ye-Xin and Ai, Yang and Ling, Zhen-Hua},
  booktitle={Proc. Interspeech},
  pages={3834--3838},
  year={2023}
}
```
