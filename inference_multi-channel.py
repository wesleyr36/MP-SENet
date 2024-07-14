from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
from re import S
import torch
import librosa
from env import AttrDict
from datasets.dataset import mag_pha_stft, mag_pha_istft
from models.generator import MPNet
import soundfile as sf
import numpy as np

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def split_audio_with_overlap(audio, sample_rate, segment_length=3, overlap=2):
    segment_length_samples = segment_length * sample_rate
    overlap_samples = overlap * sample_rate
    
    segments = []
    start = 0
    while start < len(audio):
        end = min(start + segment_length_samples, len(audio))
        segment = audio[start:end]
        segments.append(segment)
        start += segment_length_samples - overlap_samples

        if end == len(audio):
            break
    
    return segments

def process_segment(segment, model, device, h):
    print(segment.shape)
    segment = torch.FloatTensor(segment).to(device)
    norm_factor = torch.sqrt(len(segment) / torch.sum(segment ** 2.0)).to(device)
    segment = (segment * norm_factor).unsqueeze(0)

    noisy_amp, noisy_pha, noisy_com = mag_pha_stft(segment, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
    audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
    audio_g = audio_g / norm_factor
    
    return audio_g.squeeze().cpu().numpy()

def crossfade_segments(segments, overlap_samples):
    if len(segments) == 1:
        return segments[0]
    
    crossfaded_audio = segments[0]
    for i in range(1, len(segments)):
        start_overlap = crossfaded_audio[-overlap_samples:]
        end_overlap = segments[i][:overlap_samples]

        # Ensure the overlapping segments have the same length
        min_length = min(len(start_overlap), len(end_overlap))
        start_overlap = start_overlap[:min_length]
        end_overlap = end_overlap[:min_length]

        # Crossfade
        crossfade = np.linspace(1, 0, min_length) * start_overlap + np.linspace(0, 1, min_length) * end_overlap
        
        crossfaded_audio = np.concatenate((crossfaded_audio[:-min_length], crossfade, segments[i][min_length:]))
    
    return crossfaded_audio

def process_audio(noisy_wav, a, h):
    # Split audio into segments with overlap
    segments = split_audio_with_overlap(noisy_wav, h.sampling_rate)

    # Process each segment
    processed_segments = [process_segment(seg, model, device, h) for seg in segments]

    # Concatenate processed segments
    processed_audio = crossfade_segments(processed_segments, overlap_samples)
    
    return processed_audio

def inference(a):
    model = MPNet(h).to(device)

    state_dict = load_checkpoint(a.checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])

    with open(a.input_test_file, 'r', encoding='utf-8') as fi:
        test_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    os.makedirs(a.output_dir, exist_ok=True)

    model.eval()

    overlap_samples = 2*44062 #2*43998 #2*44062

    with torch.no_grad():
        for i, index in enumerate(test_indexes):
            print(index)
            if noisy_wav.ndim == 1 or a.multi_channel == False:
                noisy_wav, _ = librosa.load(os.path.join(a.input_noisy_wavs_dir, index + '.wav'), h.sampling_rate, mono=False)
                # Split audio into segments with overlap
                segments = split_audio_with_overlap(noisy_wav, h.sampling_rate)

                # Process each segment
                processed_segments = [process_segment(seg, model, device, h) for seg in segments]

                # Concatenate processed segments
                processed_audio = crossfade_segments(processed_segments, overlap_samples)
            elif noisy_wav.ndim == 2 and a.multi_channel == True:
                segments_L = split_audio_with_overlap(noisy_wav[0], h.sampling_rate)
                segments_R = split_audio_with_overlap(noisy_wav[1], h.sampling_rate)

                processed_segments_L = []
                processed_segments_R = []

                # Process each segment
                for seg in range(segments_L):
                    processed_segments_L = processed_segments_L.append(process_segment(segments_L[seg], model, device, h))
                    processed_segments_R = processed_segments_R.append(process_segment(segments_R[seg], model, device, h))
                
                # Concatenate processed segments
                processed_audio_L = crossfade_segments(processed_segments_L, overlap_samples)
                processed_audio_R = crossfade_segments(processed_segments_R, overlap_samples)
                processed_audio = np.stack((processed_audio_L, processed_audio_R), axis=0)
            elif noisy_wav.ndim > 2 and a.multi_channel == True:
                segments_channels = []
                for channel in range(noisy_wav.ndim):
                    segments = []
                    segments = split_audio_with_overlap(noisy_wav[channel], h.sampling_rate)
                    segments_channels = segments_channels.append(segments)

                processed_segments_channels = []
                for channel in range(noisy_wav.ndim):
                    processed_segments = []
                    for seg in range(segments_channels[0]):
                        processed_segments = processed_segments_L.append(process_segment(segments_channels[channel][seg], model, device, h))
                        segments_channel s= segments_channels.append(processed_segments)
            output_file = os.path.join(a.output_dir, index + '.wav')

            sf.write(output_file, processed_audio, h.sampling_rate, 'float')
                


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_clean_wavs_dir', default='VoiceBank+DEMAND/wavs_clean')
    parser.add_argument('--input_noisy_wavs_dir', default='VoiceBank+DEMAND/wavs_noisy')
    parser.add_argument('--input_test_file', default='VoiceBank+DEMAND/test.txt')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--multi_channel', default=False)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

