import numpy as np
import matplotlib.pyplot as plt


def getscales(X,Y):
    distances = np.sqrt(np.power(X-4,2)+np.power(Y-4,2))
    scalefactionor = 2.0/(1.0+np.exp(-2*distances))-0.99;
    return scalefactionor


def virtualizing(data, action, label, pred, sample_size, save=False):
    sample_index = np.random.choice(len(data), sample_size)
    fig9 = plt.figure(figsize=(12, 9), dpi=100)
    ax9 = fig9.add_subplot(111)
    plt.xlim(0, 8)
    plt.ylim(0, 8)

    delta = 0.01
    x = y = np.arange(0, 8.01, delta)
    X, Y = np.meshgrid(x, y)
    Z = getscales(X, Y)
    cp = plt.contour(X, Y, Z, extent=(0, 8, 0, 8))
    plt.clabel(cp, inline=True,
               fontsize=10)

    plt.plot([data[0, 0], label[0, 0]], [data[0, 1], label[0, 1]], 'r-', lw=1.5, label="True Transition")
    plt.plot([data[0, 0], data[0, 0] + action[0, 0]], [data[0, 1], data[0, 1] + action[0, 1]], 'g:', lw=1.5,
             label="Proposed Transition")
    plt.plot([data[0, 0], pred[0, 0]], [data[0, 1], pred[0, 1]], 'b-.', lw=1.5, label="Network Estimated Transition")
    plt.legend(loc='upper center', shadow=True)
    for i in sample_index:
        plt.plot([data[i, 0], label[i, 0]], [data[i, 1], label[i, 1]], 'r-', lw=1.5)
        plt.plot([data[i, 0], data[i, 0] + action[i, 0]], [data[i, 1], data[i, 1] + action[i, 1]], 'g:', lw=1.5)
        plt.plot([data[i, 0], pred[i, 0]], [data[i, 1], pred[i, 1]], 'b-.', lw=1.5)

    if save:
        plt.savefig('Comparison.png')
    else:
        plt.show()