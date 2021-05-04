import torch
import torchelie as tch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from torchelie.recipes import CrossEntropyClassification
from torchelie.recipes import DeepDream
from torchelie.recipes import FeatureVis
from torchelie.recipes import NeuralStyle
from torchelie.recipes import TrainAndCall
import torchelie.callbacks as tcb


class FakeData:

    def __len__(self):
        return 10

    def __getitem__(self, i):
        cls = 0 if i < 5 else 1
        return torch.randn(10) + cls * 3, cls


class FakeImg:

    def __len__(self):
        return 10

    def __getitem__(self, i):
        cls = 0 if i < 5 else 1
        return torch.randn(1, 4, 4) + cls * 3, cls


def test_classification():
    trainloader = DataLoader(FakeImg(), 4, shuffle=True)
    testloader = DataLoader(FakeImg(), 4, shuffle=True)

    model = nn.Sequential(tch.nn.Reshape(-1), nn.Linear(16, 2))

    clf_recipe = CrossEntropyClassification(model, trainloader, testloader,
                                            ['foo', 'bar'])
    clf_recipe.run(1)


def test_deepdream():
    model = nn.Sequential(nn.Conv2d(3, 6, 3))
    dd = DeepDream(model, '0')
    dd.fit(ToPILImage()(torch.randn(3, 128, 128)), 1)


def test_featurevis():
    model = nn.Sequential(nn.Conv2d(3, 6, 3))
    dd = FeatureVis(model, '0', 229, lr=1)
    dd.fit(1, 0)


def test_neuralstyle():
    stylizer = NeuralStyle()

    content = ToPILImage()(torch.randn(3, 64, 64))
    style_img = ToPILImage()(torch.randn(3, 64, 64))

    stylizer.fit(1,
                 content,
                 style_img,
                 1,
                 second_scale_ratio=1,
                 content_layers=['conv1_1'])


def test_trainandcall():
    model = nn.Linear(10, 2)

    def train_step(batch):
        x, y = batch
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        return {'loss': loss}

    def after_train():
        print('Yup.')
        return {}

    trainloader = DataLoader(FakeData(), 4, shuffle=True)
    trainer = TrainAndCall(model, train_step, after_train, trainloader)
    trainer.callbacks.add_callbacks(
        [tcb.Optimizer(torch.optim.Adam(model.parameters(), lr=1e-3))])

    trainer.run(1)


def test_callbacks():
    from torchelie.recipes import Recipe

    def train(b):
        x, y = b
        return {'pred': torch.randn(y.shape[0])}

    m = nn.Linear(2, 2)

    r = Recipe(train, DataLoader(FakeImg(), 4))
    r.callbacks.add_callbacks([
        tcb.Counter(),
        tcb.AccAvg(),
        tcb.Checkpoint('/tmp/m', m),
        tcb.ClassificationInspector(1, ['1', '2']),
        tcb.ConfusionMatrix(['one', 'two']),
        tcb.ImageGradientVis(),
        tcb.MetricsTable(),
    ])
    r.run(1)
