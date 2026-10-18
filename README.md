# 改善注意力缺陷的功能性音乐特征分析

本项目针对注意力缺陷多动障碍（Attention-Deficit/Hyperactivity Disorder, ADHD）人群，从音频特征工程的角度出发，研究"专注音乐（Focus Music）"与古典音乐、流行音乐之间的差异，并基于研究结论，使用 AI 音乐生成模型（Suno、ACE-Step）生成符合 ADHD 用户需求的功能性音乐。

## 项目背景

ADHD 是一种全球高发的神经发育障碍，其核心症状包括持续性注意力不集中、冲动控制受损以及工作记忆缺陷。近年来研究表明，听觉刺激（尤其是功能性音乐）可以作为非药物干预手段帮助 ADHD 人群改善专注度。

本项目希望回答两个问题：

1. **专注音乐到底"长什么样"？** 通过对 Spotify 上 Focus Music、Classical Music、Pop Music 三类音乐的高级音频特征进行统计分析，提取专注音乐的关键特征指纹。报告详见 [https://fisheryv.github.io/focus-music-analysis/focus-music-analysis/audio_feature_analysis.html](https://fisheryv.github.io/focus-music-analysis/audio_feature_analysis.html)
2. **AI 能否生成合格的专注音乐？** 基于上述特征指纹构建提示词，对比 Suno 与 ACE-Step 两类音乐生成模型在不同提示词策略下的生成效果。报告详见 [https://fisheryv.github.io/focus-music-analysis/healing_music_generation.html](https://fisheryv.github.io/focus-music-analysis/healing_music_generation.html)

## 目录结构

```
ADHD-music-analysis/
├── audio/                              # AI 生成的音频样本
│   ├── ace-step/                       # ACE-Step 生成结果
│   └── suno/                           # Suno 生成结果
├── dataset/                            # 从 Spotify 抓取的音频特征数据集
│   ├── Classical Music.csv
│   ├── Focus Music.csv
│   └── Pop Music.csv
├── img/                                # 分析与对比图
├── model/                              # Essentia 预训练模型（MusiCNN / DEAM）
│   ├── deam-msd-musicnn-2.pb
│   └── msd-musicnn-1.pb
├── stems/                              # 分轨素材（用于"有鼓 vs 无鼓"对比实验）
├── audio_feature_analysis.ipynb        # Spotify 音频特征统计分析
├── audio_feature_analysis.html         # 音频特征统计分析报告 HTML 版本
├── healing_music_generation.ipynb      # AI 疗愈音频生成体验报告
├── healing_music_generation.html       # AI 疗愈音频生成体验报告 HTML 版本
├── audio_features.py                   # 基于 Essentia 的本地音频特征提取
├── audio_features_old.py               # 旧版特征提取实现（PyAV）
├── spotify_data_retrieve.py            # Spotify + RapidAPI 音频特征抓取脚本
├── plot.py                             # AI 生成音乐 vs 目标特征 对比绘图
├── requirements.txt
└── .env.example
```

## 主要内容

### 一、Spotify 音频特征分析

[audio_feature_analysis.ipynb](audio_feature_analysis.ipynb) 中通过 Spotify API 获取了 3220 首音乐的高级音频特征（`energy`、`tempo`、`loudness`、`speechiness`、`valence`、`instrumentalness`、`liveness`、`danceability`、`acousticness` 等），并对 Focus Music、Classical Music、Pop Music 三类音乐进行了对比分析。

主要结论：

- Focus Music 在 `energy`、`loudness`、`instrumentalness`、`acousticness` 上的分布与 Pop Music 几乎"上下颠倒"——表现为**低能量、低响度、高器乐性、高原生性**。
- Focus Music 与 Classical Music 在上述四个维度高度重叠，但 `valence` 显著低于古典音乐（中位数仅 ~0.0655 vs ~0.34），说明 Focus Music 在情感表达上刻意"冷淡、抽离"。
- Focus Music 的 `danceability` 是所有类别中最低的（中位数 ~0.22），缺乏强烈的节拍驱动感。

由此可勾勒出适用于 ADHD 用户的理想 Focus Music 特征：

| 特征 | 期望区间 |
|---|---|
| `energy` | < 0.2 |
| `loudness` | < -20 dB |
| `instrumentalness` | > 0.8 |
| `acousticness` | > 0.8 |
| `valence` | < 0.15 |
| `danceability` | < 0.30 |
| `tempo` | 90 – 110 BPM |

### 二、AI 疗愈音频生成体验

[healing_music_generation.ipynb](healing_music_generation.ipynb) 基于上述特征指纹，分别使用 **Suno v5.5 Pro**（商业 SOTA）和 **ACE-Step V1.5 XL Turbo**（开源 SOTA）生成 Focus Music，并从指标性提示词到语义性提示词、从 Simple 模式到 Advanced 模式做了多轮对照实验，最终通过 [plot.py](plot.py) 输出可视化对比图（见 `img/compare0X.png`）。

## 核心脚本说明

### `spotify_data_retrieve.py`

从 Spotify 抓取三类音乐特征并保存为 CSV：

- **Focus Music**：通过专辑搜索 `ADHD Focus`，再获取专辑下全部曲目；
- **Classical Music** / **Pop Music**：直接通过曲目搜索关键词获取。

> ⚠️ Spotify Web API 已下线 `audio-features` 接口，本脚本通过 [RapidAPI 上的 Spotify Extended Audio Features API](https://rapidapi.com/) 获取高级音频特征。

### `audio_features.py`

基于 [Essentia](https://essentia.upf.edu/) 的本地音频特征提取脚本，用于分析 Suno / ACE-Step 生成的音频。提取以下特征：

- `bpm`（RhythmExtractor2013）
- `danceability`（归一化到 0–1）
- `loudness`（EBU R128 integrated loudness, LUFS）
- `valence` / `arousal`（基于 MusiCNN + DEAM 预训练模型）

非 `.wav/.aiff/.flac` 格式（如 `.m4a`）会先通过 `ffmpeg` 转码为 44.1 kHz `.wav`。

用法：

```bash
python audio_features.py path/to/song.m4a
```

### `plot.py`

绘制"目标特征 vs AI 生成音乐特征"的多组对比柱状图，输出到 `img/compare0X.png`。

## 环境配置

### 依赖

```bash
pip install -r requirements.txt
```

[requirements.txt](requirements.txt) 中包含：`spotipy`、`pandas`、`seaborn`、`matplotlib`、`scikit-learn`、`numpy`、`scipy`、`xgboost`、`essentia-tensorflow`。

> Essentia 的 TensorFlow 版本（`essentia-tensorflow`）目前仅在 Linux / macOS 上提供官方 wheel；Windows 用户建议在 WSL 中运行 `audio_features.py`。`audio_features.py` 还依赖系统中可用的 `ffmpeg` 命令（用于 `.m4a` → `.wav` 转码）。

### 模型文件

`audio_features.py` 使用 Essentia 提供的两个预训练模型，已包含在 `model/` 目录：

- `msd-musicnn-1.pb` — MusiCNN embedding 模型
- `deam-msd-musicnn-2.pb` — 基于 DEAM 数据集的 valence / arousal 预测模型


## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API 密钥
cp .env.example .env
# 编辑 .env 填入密钥

# 3. （可选）重新抓取 Spotify 数据
python spotify_data_retrieve.py

# 4. 打开分析 Notebook
jupyter notebook audio_feature_analysis.ipynb

# 5. 对 AI 生成音频提取特征
python audio_features.py "audio/suno/Moss Between Pages by Fisher Yu_1.m4a"

# 6. 绘制对比图
python plot.py
```

## 作者

于儿（Fisher E. Yu）&lt;yue.fisher2025@gdhfi.com&gt;
