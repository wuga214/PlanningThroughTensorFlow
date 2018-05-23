import matplotlib.pyplot as plt


def showhistory(history, save=False, path=''):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if save:
        plt.savefig(path+'train_curve.png')
    else:
        plt.show()