import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.interactive(False)


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()
    

def show_dataset(dataset, n=6):
    imgs = [dataset[i][0] for i in range(n)]
    grid = make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()


def show_dl_combined(dl, n=3):
    batch = None
    for batch in dl:
        break

    imgs = batch[0][0][:n*n]
   
    grid = make_grid(imgs, nrow =n, padding=20)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()
    
    
def show_dl(dl, n=3):
    batch = None
    for batch in dl:
        break
    imgs = batch[0][:n*n]
    grid = make_grid(imgs, nrow =n, padding=20)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()

def plot_dl(dl, labels_map=None ,n=3):
    figure = plt.figure()
    cols, rows = n, n
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dl), size=(1,)).item()
        img, label = next(iter(dl))
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.permute(1,2,0), cmap="gray") 
    plt.show()