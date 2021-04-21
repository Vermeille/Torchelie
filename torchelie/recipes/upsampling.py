import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchelie as tch
import torchelie.models.unet as U
from torchelie.recipes import TrainAndCall


def make_net(in_sz=256, n_down=4, max_ch=128, base_ch=8):
    layers = []
    for _ in range(n_down):
        layers.append(min(base_ch, max_ch))
        assert in_sz // 2 * 2 == in_sz
        in_sz = in_sz // 2
        base_ch *= 2

    model = U.UNetBone(layers, 3, 3, skip=False)
    model.out_conv.add_module('sig', nn.Sigmoid())
    return model


def cutblur(base, patcher):
    a = max(base.shape[2], base.shape[3])
    s = random.randrange(int(a / 6), int(a / 4))

    y = random.randrange(0, base.shape[2] - 1 - s)
    x = random.randrange(0, base.shape[3] - 1 - s)

    base[:, :, y:y + s, x:x + s] = patcher[:, :, y:y + s, x:x + s]
    return base


def train(rank, world_size):
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as TF
    net = make_net(base_ch=16, max_ch=512, n_down=6)

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            tch.utils.kaiming(m, dynamic=True)

    if False:
        ckpt = torch.load('model/ckpt_25000.pth', map_location='cpu')
        net.load_state_dict(ckpt['model'])
        del ckpt
    net = nn.parallel.DistributedDataParallel(net.to(rank),
                                              device_ids=[rank],
                                              output_device=rank)

    ploss = tch.loss.PerceptualLoss(['conv1_2', 'conv2_2', 'conv3_4'],
                                    rescale=True,
                                    remove_unused_layers=True,
                                    use_avg_pool=False).to(rank)

    ploss2 = tch.loss.PerceptualLoss(['conv1_2', 'conv2_2'],
                                    rescale=False,
                                    remove_unused_layers=True,
                                    use_avg_pool=False).to(rank)

    def train_fun(batch):
        x, _ = batch
        down_mode = random.choice(['nearest', 'bilinear', 'bicubic'])
        factor = random.choice([3, 4, 5, 8])
        x_down = F.interpolate(F.interpolate(
            x,
            scale_factor=1 / factor,
            mode=down_mode,
            align_corners=True if down_mode != 'nearest' else None),
                               size=x.shape[2:],
                               mode='bicubic',
                               align_corners=True)
        x_down = cutblur(x_down, x)
        out = net(x_down * 2 - 1)
        out_g = out.detach()
        out_g.requires_grad_(True)
        out_g.retain_grad

        lossl1 = F.l1_loss(out_g, x)
        lossl1.backward()

        lossp1 = ploss(out_g, x)
        lossp1.backward()

        lossp2 = ploss2(out_g, x)
        lossp2.backward()

        out.backward(out_g.grad)

        loss = lossl1 + lossp1 + lossp2
        return {
            'loss': loss.item(),
            'down': x_down[:2],
            'up': out[:2],
            '224': F.interpolate(out[:2],
                                 scale_factor=(224 / min(out.shape[2:])))
        }

    def test_fun():
        return {}

    tfm = TF.Compose([
        tch.transforms.ResizeNoCrop(600),
        #tch.transforms.AdaptPad((256, 256), padding_mode='edge'),
        #TF.Resize(336),
        TF.CenterCrop((336, 600)),
        TF.RandomHorizontalFlip(),
        TF.ToTensor()
    ])
    ds = tch.datasets.NoexceptDataset(
        tch.datasets.FastImageFolder('/hdd/data/resources/thumbs/480p/', transform=tfm))
    dl = torch.utils.data.DataLoader(ds,
                                     num_workers=4,
                                     shuffle=True,
                                     pin_memory=True,
                                     drop_last=True,
                                     batch_size=4)
    recipe = TrainAndCall(net,
                          train_fun,
                          test_fun,
                          dl,
                          visdom_env='main' if rank == 0 else None,
                          checkpoint='model' if rank == 0 else None,
                          log_every=50)
    recipe.callbacks.add_callbacks([
        tch.callbacks.Log('down', 'down'),
        tch.callbacks.Log('up', 'up'),
        tch.callbacks.Log('224', '224'),
        tch.callbacks.WindowedMetricAvg('loss'),
        tch.callbacks.Optimizer(
            tch.optim.RAdamW(net.parameters(),
                             lr=1e-3,
                             weight_decay=1e-5,
                             stable=True)),
    ])
    recipe.to(rank)
    recipe.run(50)


if __name__ == '__main__':
    tch.utils.parallel_run(train)
