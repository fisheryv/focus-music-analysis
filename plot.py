import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 指标性提示词，Simple 模式
data_group_1 = {
    'music': ['Target','Suno #1', 'Suno #2', 'ACE-Step #1', 'ACE-Step #2'],
    'bpm': [100, 98, 98, 124, 125],
    'danceability': [0.3, 0.3957, 0.3806, 0.5653, 0.4938],
    'loudness': [-20, -14.50, -14.40, -15.28, -16.54],
    'valence': [0.15, 0.3451, 0.4084, 0.5866, 0.6151],
    'arousal': [0, 0.4271, 0.4045, 0.5521, 0.5297]
}

# 指标性提示词，Advanced 模式
data_group_2 = {
    'music': ['Target', 'Suno #3', 'Suno #4', 'ACE-Step #3', 'ACE-Step #4'],
    'bpm': [100, 108, 110, 100, 100],
    'danceability': [0.3, 0.4127, 0.3862, 0.4007, 0.5454],
    'loudness': [-20, -14.32, -13.22, -13.49, -11.92],
    'valence': [0.15, 0.5150, 0.5690, 0.6586, 0.6701],
    'arousal': [0, 0.5018, 0.5776, 0.6311, 0.6608]
}

# 语义性提示词 v1.0，Simple 模式
data_group_3 = {
    'music': ['Target', 'Suno #5', 'Suno #6', 'ACE-Step #5', 'ACE-Step #6'],
    'bpm': [100, 100, 100, 100, 100],
    'danceability': [0.3, 0.3165, 0.3102, 0.4847, 0.3928],
    'loudness': [-20, -14.47, -15.05, -17.36, -16.98],
    'valence': [0.15, 0.2894, 0.3038, 0.4638, 0.4892],
    'arousal': [0, 0.3117, 0.347, 0.356, 0.3758]
}

# 语义性提示词 v2，Advanced 模式
data_group_4 = {
    'music': ['Target', 'Suno #7', 'Suno #8', 'ACE-Step #7', 'ACE-Step #8'],
    'bpm': [100, 95, 96, 96, 96],
    'danceability': [0.3, 0.3544, 0.3595, 0.3581, 0.3484],
    'loudness': [-20, -14.23, -13.80, -14.54, -16.43],
    'valence': [0.15, 0.2875, 0.2822, 0.3339, 0.2964],
    'arousal': [0, 0.2898, 0.3205, 0.3056, 0.2775]
}

# 无鼓版 vs 有鼓版
data_group_5 = {
    'music': ['Target', 'Suno #7', 'Suno #7 drumless', 'Suno #8', 'Suno #8 drumless'],
    'bpm': [100, 95, 95, 96, 96],
    'danceability': [0.3, 0.3544, 0.2965, 0.3595, 0.2879],
    'loudness': [-20, -14.23, -18.0, -13.80, -18.3],
    'valence': [0.15, 0.2875, 0.2853, 0.2822, 0.2937],
    'arousal': [0, 0.2898, 0.242, 0.3205, 0.2584]
}

def plot(data:dict, save_path:str):
    df = pd.DataFrame(data)

    # 特征列表（也是子图的顺序）
    features = ['bpm', 'danceability', 'valence', 'arousal',  'loudness']

    # 创建并排子图：1行5列，共享Y轴
    fig, axes = plt.subplots(1, len(features), figsize=(14, 4), sharey=True)

    # 为每个特征绘制水平条形图
    for i, feat in enumerate(features):
        # 按特征值降序排序，让条形从大到小排列（可选）
        # plot_df = df[['misic', feat]].sort_values(by=feat, ascending=True)
        
        # 绘制水平条形图（x=特征值, y=模型名）
        sns.barplot(data=df, x=feat, y='music', hue='music', legend=False, ax=axes[i], palette='Blues_d')
        
        # 设置子图标题（特征名）
        axes[i].set_title(feat)
        
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        if i != 0:
            # 同时隐藏其他子图的Y轴刻度标签（但保留刻度线，保证对齐）
            axes[i].tick_params(axis='y', labelleft=False)
        
        # 设置X轴标签
        axes[i].set_xlabel(feat)
        
        # 可选：在条形末端显示数值
        for container in axes[i].containers:
            if i in range(1,4):
                axes[i].bar_label(container, fmt='%.4f', padding=3)
            else:
                axes[i].bar_label(container, fmt='%.2f', padding=3)

    # 调整布局，防止标签重叠
    plt.tight_layout()
    plt.savefig(save_path, dpi=300,
                bbox_inches="tight", facecolor="white")

if __name__ == '__main__':
    for i, data in enumerate([data_group_1, data_group_2, data_group_3, data_group_4, data_group_5]):
        plot(data, f'img/compare0{i+1}.png')