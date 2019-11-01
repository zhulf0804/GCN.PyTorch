import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import os
import numpy as np
from sklearn.manifold import TSNE


sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


# Random state.
RS = 20191101


def tsne_vis(x, y, labels, title, name='tsne', saved_dir='experiments'):
    '''
    t-sne visualization.
    :param x: (n, 2)
    :param y: (n, )
    :param labels: (class_num, )
    :param title: used for plt.title and saved name
    :param name: used for saved filename
    :return:
    '''
    x = TSNE(random_state=RS).fit_transform(x)
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", len(labels)))
    ax = plt.subplot(aspect='equal')
    ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[y.astype(np.int)])
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit.
    txts = []
    for i in range(len(labels)):
        # Position of each label.
        xtext, ytext = np.median(x[y == i, :], axis=0)
        txt = ax.text(xtext, ytext, labels[i], fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    plt.title(title)
    plt.savefig(os.path.join(saved_dir, "%s_%s.png" %(name, title)), dpi=120)
    plt.show()