"""
this script equalizes the mean and std of the convolutions of a VGG network.
It is used to make the VGG network used in the Gatys et al. style transfer
paper to be compatible with the VGG network used in the Johnson et al. paper.
It equalizes the contribution of each layer to the loss.

./stylevgg <vgg model> <imagenet path>

<vgg model> must be a constructor in torchelie.models
"""
import torch
from torchelie.datasets import FastImageFolder
import torchvision.transforms as TF
import torchelie.models as tchm
import torchelie as tch
import sys

torch.autograd.set_grad_enabled(False)

imagenet_path = sys.argv[2]

model = sys.argv[1]
m = tchm.__dict__[model](1000, pretrained='classification/imagenet')
del m.classifier
m.cuda()
m.eval()

ds = FastImageFolder(imagenet_path,
                     transform=TF.Compose([
                         TF.Resize(256),
                         TF.CenterCrop(224),
                         TF.ToTensor(),
                         tch.nn.ImageNetInputNorm()
                     ]))

batches = [
    b[0] for _, b in zip(
        range(200), torch.utils.data.DataLoader(
            ds, batch_size=320, shuffle=True))
]

batch = batches[0].cuda()


def flatvgg():
    layers = []

    def _rec(m):
        if len(list(m.children())) == 0:
            layers.append(m)
        else:
            for mm in m.children():
                _rec(mm)

    _rec(m.features)
    return torch.nn.Sequential(*layers)


idxs = [
    i for i, nm in enumerate(dict(m.features.named_children()).keys())
    if 'conv' in nm
]
flat = flatvgg()

flatidxs = [i for i, l in enumerate(flat) if isinstance(l, torch.nn.Conv2d)]
print(flatidxs)
#flatidxs.append(len(flat))
print(dict(m.features.named_children()).keys())

print('before')
for i in idxs:
    with torch.cuda.amp.autocast():
        out = m.features[:i + 1](batch)
    mean = out.cpu().float().mean(dim=(0, 2, 3))
    del out
    print(mean.mean(), mean.std())

prev_mean = torch.tensor([1, 1, 1]).cuda()
for i in range(len(flatidxs)):
    print('computing', i)
    ms = []
    for b in batches:
        with torch.cuda.amp.autocast():
            out = flat[:flatidxs[i] + 2](b.cuda())
        mean = out.cpu().float().mean(dim=(0, 2, 3))
        del out
        ms.append(mean)
    mean = torch.stack(ms, dim=0).mean(0).cuda()
    flat[flatidxs[i]].weight.data *= (prev_mean[None, :, None, None]
                                      / mean[:, None, None, None])
    flat[flatidxs[i]].bias.data /= mean
    prev_mean = mean

print('after')
for i in idxs:
    with torch.cuda.amp.autocast():
        out = m.features[:i + 1](batch)
    mean = out.cpu().float().mean(dim=(0, 2, 3))
    del out
    print(mean.mean(), mean.std())

ref = tchm.__dict__[model](
    1000, pretrained='classification/imagenet').features.cuda()(batch[:128])
print((m.features(batch[:128])
       - ref / prev_mean[None, :, None, None]).abs().mean().item())
torch.save(m.state_dict(), f'{model}.pth')
