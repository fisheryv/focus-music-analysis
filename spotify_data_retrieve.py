# -*- coding: utf-8 -*-
"""
从 Spotify 获取音乐特征数据。

具体而言，本脚本获取了三类音乐的特征：
1. Focus Music：通过关键词搜索专辑，然后获取专辑下的所有歌曲。
2. Classical Music：直接通过关键词`Classical Music`获取歌曲。
3. Pop Music：直接通过关键词`Pop Music`获取歌曲。

获取到的音乐特征以 csv 格式保存到 /dataset 目录下。

@author: Fisher E. Yu
@date: 2026-03-22
"""
import spotipy
import os
import json
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import pandas as pd
import time

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
RAPID_API_KEY = os.getenv("RAPID_API_KEY")
RAPID_API_HOST = "spotify-extended-audio-features-api.p.rapidapi.com"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
))

def extract_audio_features(audio_features: dict) -> dict:
    """
    从 RapidAPI 提供的音频特征中提取需要的字段
    :param audio_features: RapidAPI 提供的音频特征字典
    :return: 包含需要的音频特征的字典
    """
    return {
        "id": audio_features.get("id", ""),
        "energy": audio_features.get("energy", -1),
        "tempo": audio_features.get("tempo", -1),
        "duration_ms": audio_features.get("duration_ms", -1),
        "key": audio_features.get("key", -1),
        "loudness": audio_features.get("loudness", -1),
        "mode": audio_features.get("mode", -1),
        "speechiness": audio_features.get("speechiness", -1),
        "valence": audio_features.get("valence", -1),
        "instrumentalness": audio_features.get("instrumentalness", -1),
        "liveness": audio_features.get("liveness", -1),
        "danceability": audio_features.get("danceability", -1),
        "acousticness": audio_features.get("acousticness", -1),
        "time_signature": audio_features.get("time_signature", -1),
    }

def get_audio_features(track_id: str) -> list[dict]:
    """
    获取歌曲的特征

    WARNING: 由于 Spotify Web API 已经废除了 audio_features 接口，所以这里使用 RapidAPI 提供的接口。
    :param track_id: Spotify 歌曲 ID 列表，逗号分隔
    :return: 歌曲的特征列表
    :raises: requests.HTTPError
    """
    url = f"https://{RAPID_API_HOST}/v1/audio-features"
    params = {"ids": track_id}
    headers = {
        "x-rapidapi-key": RAPID_API_KEY,
        "x-rapidapi-host": RAPID_API_HOST,
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    result = response.json()
    audio_features = result.get("audio_features", [])
    return audio_features


def search(key_word: str, type: str, limit: int = 10, offset: int = 0, max_pages: int | None = None):
    """
    搜索 Spotify 上的音乐

    :param key_word: 搜索关键词
    :param type: 搜索类型，可选值为 "track"、"album"、"artist"、"playlist"
    :param limit: 每页返回的歌曲数量
    :param offset: 偏移量，用于分页
    :param max_pages: 最大页数，用于限制返回的歌曲数量
    :return: 搜索结果的迭代器
    """
    page = 0
    while True:
        if max_pages is not None and page >= max_pages:
            break

        results = sp.search(q=key_word, type=type, limit=limit, offset=offset)
        data = results.get(f"{type}s", {})
        items = data.get("items", [])

        if not items:
            break
        
        for item in items:
            if item and item.get('id', ""):
                yield item

        offset += limit
        page += 1

        if not data.get("next", ""):
            break
        

def get_album_tracks(album_id: str,  limit: int = 50, offset: int = 0, max_pages: int | None = None):
    """
    获取专辑中的所有歌曲

    :param album_id: 专辑ID
    :param limit: 每页返回的歌曲数量
    :param offset: 偏移量，用于分页
    :param max_pages: 最大页数，用于限制返回的歌曲数量
    :return: 专辑中的所有歌曲的迭代器
    """
    page = 0
    while True:
        if max_pages is not None and page >= max_pages:
            break
        
        result = sp.album_tracks(album_id)
        items = result.get("items", []) 
        
        if not items:
            break
        
        for item in items:
            if item and item.get('id', ""):
                yield item
        
        if result.get("next") is None:
            break
        
        offset += limit
        page += 1


def save_to_json(data: list[dict], filepath: str):
    """
    保存数据到 JSON 文件

    :param data: 要保存的数据列表
    :param filepath: JSON 文件路径
    """
    dir_path = os.path.dirname(filepath)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(data)} 条数据到 {filepath}")


def save_to_csv(data: list[dict], filepath: str):
    """
    保存数据到 CSV 文件

    :param data: 要保存的数据列表
    :param filepath: CSV 文件路径
    """
    COLUMNS = [
        "id", "energy", "tempo", "duration_ms", "key", "loudness", 
        "mode", "speechiness", "valence", "instrumentalness", "liveness",
        "danceability", "acousticness", "time_signature"
    ]
    data = [d for d in data if d is not None]
    df = pd.DataFrame(data, columns=COLUMNS)
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    df.to_csv(filepath, index=False)
    print(f"已保存 {len(data)} 条数据到 {filepath}")

def get_all_audio_features(track_ids: list[str]) -> list[dict]:
    """
    获取歌曲id列表中所有歌曲的特征

    :param track_ids: 歌曲ID列表
    :return: 所有歌曲的特征列表
    """
    all_features = []
    for i in range(0, len(track_ids), 5):
        ids = ",".join(track_ids[i:i+5])
        print(ids)
        audio_features = get_audio_features(ids)
        for feat in audio_features:
            if feat is not None:
                all_features.append(extract_audio_features(feat))
        time.sleep(1)  # 避免 API 调用频率限制

    return all_features
        

def fetch_music_features_from_album(key_word:str):
    """
    获取专辑搜索结果下的所有歌曲的特征，保存到 JSON 或 CSV 文件中

    :param key_word: 专辑搜索关键词
    """
    id_set = set[str]()
    for album in search(key_word, 'album'):
        print(f"{album['id']} \t {album['name']}")
        for track in get_album_tracks(album['id']):
            id_set.add(track['id'])
    all_features = get_all_audio_features(list(id_set))
    if all_features:
        # save_to_json(all_features, f"dataset/{key_word}.json")
        save_to_csv(all_features, f"dataset/{key_word}.csv")


def fetch_music_features_from_track(key_word:str):
    """
    获取歌曲搜索结果下的所有歌曲的特征，保存到 JSON 或 CSV 文件中

    :param key_word: 歌曲搜索关键词
    """
    id_set = list[str]()
    for track in search(key_word, 'track'):
        print(f"{track['id']} \t {track['name']}")
        id_set.append(track['id'])
    all_features = get_all_audio_features(list(id_set))
    if all_features:
        # save_to_json(all_features, f"dataset/{key_word}.json")
        save_to_csv(all_features, f"dataset/{key_word}.csv")


if __name__ == "__main__":
    # 获取 "ADHD Focus" 音乐特征
    fetch_music_features_from_album("ADHD Focus")

    # 获取 "Classical Music" 音乐特征
    fetch_music_features_from_track("Classical Music")

    # 获取 "Pop Music" 音乐特征
    fetch_music_features_from_track("Pop Music")
