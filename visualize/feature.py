import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import manifold
from sklearn.preprocessing import normalize
from math import pi
import os.path as osp
from functools import reduce

kwargs = {'levels': np.arange(0, 5.5, 0.5)}

def composePath(*pathArr):
    return reduce((lambda prePath, cur: osp.join(prePath, cur)), pathArr, '')

# dimensionality reduction
def reduction_features(user_emb):
    udx = np.random.choice(len(user_emb), 2000)
    selected_user_emb = user_emb[udx]
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=12345)
    user_emb_2d = tsne.fit_transform(selected_user_emb)
    user_emb_2d = normalize(user_emb_2d, axis=1, norm='l2')
    return user_emb_2d


# print user-item feature plots input: user
def plot_features(embs, name):
    reduction_f = reduction_features(embs)
    f, axs = plt.subplots(nrows=2, figsize=(20,5), gridspec_kw={'height_ratios': [3, 1]})
    kwargs = {'levels': np.arange(0, 5.5, 0.5)}
    # sns.set_palette('viridis')
    sns.kdeplot(data=reduction_f, bw=0.05, shade=True, color="blue", cmap="GnBu", legend=True, ax=axs[0], **kwargs)
    axs[0].set_title(name, fontsize=9, fontweight="bold")
    x = [p[0] for p in reduction_f]
    y = [p[1] for p in reduction_f]
    angles = np.arctan2(y, x)
    sns.kdeplot(data=angles, bw=0.15, shade=True,
                legend=True, ax=axs[1], color='green')
    axs[0].tick_params(axis='x', labelsize=8)
    axs[0].tick_params(axis='y', labelsize=8)
    axs[0].patch.set_facecolor('white')
    axs[0].collections[0].set_alpha(0)
    axs[0].set_xlim(-1.2, 1.2)
    axs[0].set_ylim(-1.2, 1.2)
    axs[0].set_xlabel('Features', fontsize=9)
    axs[0].set_ylabel('Features', fontsize=9)

    axs[1].tick_params(axis='x', labelsize=8)
    axs[1].tick_params(axis='y', labelsize=8)
    axs[1].set_xlabel('Angles', fontsize=9)
    axs[1].set_ylim(0, 0.5)
    axs[1].set_xlim(-pi, pi)
    axs[1].set_ylabel('Density', fontsize=9)
    
    path = composePath('.', name + '.png')
    print('path==>' + path)
    plt.savefig(name)

emb = np.random.rand(100, 4096)

plot_features(emb, 't1')
