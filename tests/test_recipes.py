import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from torchelie.recipes.classification import CrossEntropyClassification
from torchelie.recipes.deepdream import DeepDreamRecipe
from torchelie.recipes.feature_vis import FeatureVisRecipe
from torchelie.recipes.neural_style import NeuralStyleRecipe
from torchelie.recipes.trainandtest import TrainAndCall


class FakeData:
    def __len__(self):
        return 10

    def __getitem__(self, i):
        cls = 0 if i < 5 else 1
        return torch.randn(10) + cls * 3, cls


def test_classification():
    trainloader = DataLoader(FakeData(), 4, shuffle=True)
    testloader = DataLoader(FakeData(), 4, shuffle=True)

    model = nn.Linear(10, 2)

    clf_recipe = CrossEntropyClassification(model)
    clf_recipe(trainloader, testloader, 1)


def test_deepdream():
    model = nn.Sequential(nn.Conv2d(3, 6, 3))
    dd = DeepDreamRecipe(model, '0', lr=1)
    dd(1, ToPILImage()(torch.randn(3, 8, 8)))


def test_featurevis():
    model = nn.Sequential(nn.Conv2d(3, 6, 3))
    dd = FeatureVisRecipe(model, '0', 8, lr=1)
    dd(1, 0)


def test_neuralstyle():
    stylizer = NeuralStyleRecipe()

    content = ToPILImage()(torch.randn(3, 32, 32))
    style_img = ToPILImage()(torch.randn(3, 32, 32))

    result = stylizer(1, content, style_img, 1, ['conv1_1'])

def test_trainandcall():
    class Test(nn.Module):
        def __init__(self):
            super(Test, self).__init__()
            self.model = nn.Linear(10, 2)

        def make_optimizer(self):
            return torch.optim.Adam(self.model.parameters(), lr=1e-3)

        def train_step(self, batch, opt):
            x, y = batch
            opt.zero_grad()
            out = self.model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
            loss.backward()
            opt.step()
            return {'loss': loss}

        def after_train(self):
            print('Yup.')
            return {}

    trainloader = DataLoader(FakeData(), 4, shuffle=True)
    trainer = TrainAndCall(Test())
    trainer(trainloader)
