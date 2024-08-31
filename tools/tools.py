import torch
from matplotlib import pyplot as plt


class EarlyStopping:

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss


# 定义可视化函数
def visualize_losses(train_losses, val_losses, val_accuracies, learning_rates, gradient_norms, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # 将 Train Loss 和 Val Loss 画在同一个图上
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(val_losses, label='Val Loss')
    axs[0, 0].set_title('Train and Val Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Val Accuracy
    axs[0, 1].plot(val_accuracies, label='Val Accuracy')
    axs[0, 1].set_title('Val Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()

    # Learning Rate
    axs[1, 0].plot(learning_rates, label='Learning Rate')
    axs[1, 0].set_title('Learning Rate')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Learning Rate')
    axs[1, 0].legend()

    # Gradient Norm
    axs[1, 1].plot(gradient_norms, label='Gradient Norm')
    axs[1, 1].set_title('Gradient Norm')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Gradient Norm')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
