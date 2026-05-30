import av
import numpy as np
from essentia.standard import *
import os
import pandas as pd


TARGET_SR = 44100


def analyze_audio(fpath):
    container = av.open(fpath)
    audio_stream = container.streams.audio[0]
    orig_sr = audio_stream.codec_context.sample_rate

    samples = []
    for frame in container.decode(audio=0):
        s = frame.to_ndarray()
        samples.append(s)
    container.close()

    audio_data = np.concatenate(samples, axis=1)
    mono = audio_data.mean(axis=0).astype(np.float32)
    mono = mono / (np.max(np.abs(mono)) + 1e-10)

    if orig_sr != TARGET_SR:
        resample = Resample(inputSampleRate=orig_sr, outputSampleRate=TARGET_SR, quality=4)
        audio = resample(mono)
    else:
        audio = mono

    bpm, _, _, _, _ = RhythmExtractor2013(method="multifeature")(audio)
    danceability, _ = Danceability()(audio)
    rms = RMS()(audio)
    energy = float(rms ** 2)
    loudness = float(20 * np.log10(rms + 1e-10))
    ebu_momentary, ebu_shortterm, ebu_integrated, dr = LoudnessEBUR128(hopSize=1024/TARGET_SR, startAtZero=True)(audio)

    return {
        'Tempo (BPM)': round(bpm, 1),
        'Danceability': round(danceability, 4),
        'Energy (RMS²)': round(energy, 6),
        # 'Spectral Energy': round(energy2, 6),
        'Loudness (dBFS)': round(loudness, 2),
        'Integrated Loudness (dBFS)': round(ebu_integrated, 2),
    }


def analyze_directory(audio_dir):
    results = []
    for fname in sorted(os.listdir(audio_dir)):
        if not fname.endswith('.m4a'):
            continue
        fpath = os.path.join(audio_dir, fname)
        row = analyze_audio(fpath)
        row['File'] = fname
        results.append(row)
    return pd.DataFrame(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze audio features using essentia')
    parser.add_argument('audio_dir', help='Directory containing .m4a files')
    args = parser.parse_args()
    print(args.audio_dir)
    df = analyze_directory(args.audio_dir)
    print(df.to_string(index=False))
