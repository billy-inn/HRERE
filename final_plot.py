import numpy as np
import config
import os
from sklearn.metrics import precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot():
    font = {'size': 13}
    matplotlib.rc('font', **font)
    plt.clf()
    d = {
        "Base": "HRERE-base",
        "Naive": "HRERE-naive",
        "Full": "HRERE-full"
    }
    color = {
        "Base": "turquoise",
        "Naive": "red",
        "Full": "cornflowerblue"
    }
    shape = {
        "Base": "-.",
        "Naive": "--",
        "Full": "-"
    }
    width = {
        "Base": 4,
        "Naive": 4,
        "Full": 4,
    }

    for filename in ['Base', 'Naive', 'Full']:
        all_labels = np.load(os.path.join(config.PLOT_OUT_DIR, filename + "_labels.npy"))
        all_probs = np.load(os.path.join(config.PLOT_OUT_DIR, filename + "_probs.npy"))
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        plt.plot(recall[:], precision[:], color=color[filename], ls=shape[filename],
                 lw=width[filename], label=d[filename])

    all_labels = np.load(os.path.join(config.PLOT_OUT_DIR, "Weston_labels.npy"))
    all_probs = np.load(os.path.join(config.PLOT_OUT_DIR, "Weston_probs.npy"))
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    plt.plot(recall[:], precision[:], lw=2, color="navy", label="Weston", ls='-')

    filename = ['PCNN+ATT', 'CNN+ATT']
    color = ['teal', 'darkorange']
    shape = ['--', '-.']
    for i in range(len(filename)):
        precision = np.load(os.path.join(config.PLOT_DATA_DIR, filename[i] + '_precision.npy'))
        recall = np.load(os.path.join(config.PLOT_DATA_DIR, filename[i] + '_recall.npy'))
        plt.plot(recall, precision, color=color[i], lw=2, label=filename[i], ls=shape[i])

    plt.xlabel('Recall')
    plt.ylabel("Precision")
    plt.ylim([0.65, 1.0])
    plt.xlim([0.0, 0.2])
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(config.PLOT_FIG_DIR, 'comparison.png'))

def main():
    plot()

if __name__ == "__main__":
    main()
