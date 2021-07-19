import matplotlib.pyplot as plt
import os

def plot_loss(loss, dir, fontsize=None):
    if fontsize:
        plt.rcParams.update({'font.size': fontsize})
    plt.figure(figsize=(10,5))
    plt.plot(loss['epoch'], loss['loss_train'], 'b-', label='Training Loss')
    plt.plot(loss['epoch'], loss['loss_val'], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir, 'loss.pdf'))
    plt.close()