import torch

import torchelie.nn as tnn
from torchelie.data_learning import ParameterizedImg


class FeatureVisRecipe:
    def __init__(self, model, layer, device='cpu'):
        self.device = device
        self.model = tnn.WithSavedActivations(model, names=[layer]).to(device)
        self.model.eval()

    def fit(self, nb_iters=500):
        canvas = ParameterizedImg(1, 3, 224, 224).to(self.device)
        canvas.requires_grad=True

        opt = torch.optim.Adam(canvas.parameters(), lr=1e-3, betas=(0.95, 0.9995))

        for iters in range(nb_iters):
            opt.zero_grad()
            _, acts = self.model(canvas())
            loss = -acts[0]
            loss.backward()
            opt.step()

        return canvas().detach().cpu()[0]
