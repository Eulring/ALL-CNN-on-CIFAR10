import torch
import numpy as np


def ZCA_W(x):
# Input Matrix should be the shape of (N x 32 x 32 x 3)
    x = torch.Tensor(x)
    x = x.numpy()
    x = np.transpose(x, (0, 3, 1, 2))
    x = torch.tensor(x)
    
    s0, s1, s2, s3 = x.size(0), x.size(1), x.size(2), x.size(3)

    X = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
    X = X - X.mean(dim=0)
    sigma = torch.mm(X.t(), X) / X.size(0)
    u, s, _ = np.linalg.svd(sigma.numpy())
    epsilon = 1e-5

    ss = torch.Tensor(np.diag(1. / np.sqrt(s + epsilon)))
    u = torch.Tensor(u)
    zca_matrix = torch.mm(torch.mm(u, ss), u.t())

    print(zca_matrix.shape)

    nx = torch.mm(zca_matrix, X.t()).t()
    
    nx = nx.view(s0, s1, s2, s3)
    
    nx = nx.numpy()
    nx = np.transpose(nx, (0, 2, 3, 1))
    #nx = torch.tensor(nx)
    
    
    return nx


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
