# -*- coding: utf-8 -*-
"""
基于 Essentia 的本地音频特征提取。

@author: Fisher E. Yu
@date: 2026-03-26
"""
import essentia.standard as es
import numpy as np
import os
import subprocess
import tempfile

TARGET_SR = 44100

def _to_wav(file_path: str) -> str:
    """
    将音频文件转换为.wav 格式，返回临时文件路径。

    :param file_path: 音频文件路径
    :return: 临时文件路径
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ('.wav', '.aiff', '.flac'):
        return file_path
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    cmd = [
        'ffmpeg', '-y', '-i', file_path,
        '-ar', str(TARGET_SR),
        '-ac', '2',
        '-f', 'wav', tmp.name,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp.name

def analyze_music_features(file_path: str):
    """
    提取音乐特征（BPM、舞动感、能量、响度、情感特征）。支持单声道/立体声的自动处理。

    :param file_path: 音频文件路径
    :return: 包含特征的字典（bpm, danceability, energy, integrated_loudness (LUFS), valence, arousal）  
    """
    # Essentia ‌原生不支持.m4a 格式，需将其转换成.wav 格式。
    wav_path = _to_wav(file_path) 
    is_tmp = wav_path != file_path

    try:
        loader = es.AudioLoader(filename=wav_path)
        audio, sr_orig, num_channels, _, _, _ = loader()
        # 重采样到目标采样率
        if sr_orig != TARGET_SR:
            resampler = es.Resample(inputSampleRate=sr_orig, outputSampleRate=TARGET_SR)
            if num_channels == 1:
                audio = resampler(audio[:, 0]).reshape(-1, 1)
            else:
                resampled_channels = []
                for ch in range(num_channels):
                    resampled_channels.append(resampler(audio[:, ch]))
                audio = np.column_stack(resampled_channels)
            sr = TARGET_SR
        else:
            sr = sr_orig
        # 转换为单声道
        if num_channels >= 2:
            mono_audio = np.mean(audio, axis=1)
        else:
            mono_audio = audio[:, 0]

        # 提取节奏特征
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, _ = rhythm_extractor(mono_audio)

        # 提取 danceability
        dance_algo = es.Danceability(sampleRate=sr)
        danceability, _ = dance_algo(mono_audio)
        # essentia 的 danceability 正常值范围在 0 到 3 之间，归一化到 0 到 1 之间。
        danceability = danceability/3.0

        # 提取 energy
        # !! WARNING: Spotify 的 energy 与 essentia 的 energy 不是同一概念 !!
        # energy_algo = es.Energy()
        # energy = energy_algo(mono_audio)

        # 提取 loudness
        loudness_algo = es.LoudnessEBUR128(sampleRate=sr)
        if num_channels >= 2:
            loudness_input = audio[:, :2]
        else:
            loudness_input = mono_audio
        _, _, integrated_loudness, _ = loudness_algo(loudness_input)

        # 提取 valence 和 arousal
        # 参考 https://essentia.upf.edu/models.html#arousal-valence-deam
        # 将音频重采样到 16KHz，因为模型通常在 16KHz 下训练
        resampler = es.Resample(inputSampleRate=sr, outputSampleRate=16000)
        audio_16KHz = resampler(mono_audio)
        # 载入预训练的 TensorFlow 模型
        embedding_model = es.TensorflowPredictMusiCNN(graphFilename="model/msd-musicnn-1.pb", output="model/dense/BiasAdd")
        embeddings = embedding_model(audio_16KHz)
        # 预测 valence 和 arousal
        model = es.TensorflowPredict2D(graphFilename="model/deam-msd-musicnn-2.pb", output="model/Identity")
        predictions = model(embeddings)
        predictions = np.mean(predictions.squeeze(), axis=0)
        # 归一化 (1, 9) -> (0, 1)
        predictions = (predictions - 1) / 8
        valence = predictions[0]
        arousal = predictions[1]

        return {
            "bpm": round(bpm, 0),
            "danceability": round(danceability, 4),
            # "energy": round(energy, 6),
            "loudness": round(integrated_loudness, 2),
            "valence": round(float(valence), 4),
            "arousal": round(float(arousal), 4),
        }
    finally:
        if is_tmp:
            os.unlink(wav_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python audio_features.py <音频文件>")
        sys.exit(1)
    result = analyze_music_features(sys.argv[1])
    for k, v in result.items():
        print(f"{k}: {v}")
