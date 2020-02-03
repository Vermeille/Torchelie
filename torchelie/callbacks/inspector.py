from io import BytesIO
import base64 as b64

import torch
import torch.nn.functional as F
from torchelie.utils import as_multiclass_shape

import numpy as np
from PIL import Image


def img2html(img, opts=None):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()

    opts = {} if opts is None else opts
    if isinstance(img, np.ndarray):
        nchannels = img.shape[0] if img.ndim == 3 else 1
        if nchannels == 1:
            img = np.squeeze(img)
            img = img[np.newaxis, :, :].repeat(3, axis=0)

        if 'float' in str(img.dtype):
            if img.max() <= 1:
                img = img * 255.
            img = np.uint8(img)

        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img)

    opts['width'] = opts.get('width', img.width)
    opts['height'] = opts.get('height', img.height)

    buf = BytesIO()
    image_type = 'png'
    imsave_args = {}
    if 'jpgquality' in opts:
        image_type = 'jpeg'
        imsave_args['quality'] = opts['jpgquality']

    img.save(buf, format=image_type.upper(), **imsave_args)

    b64encoded = b64.b64encode(buf.getvalue()).decode('utf-8')

    return '<img src="data:image/{};base64,{}"/>'.format(
        image_type, b64encoded)


class ClassificationInspector:
    def __init__(self, topk, labels, center_value=0):
        self.labels = labels
        self.center_value = center_value
        self.topk = topk
        self.reset()

    def reset(self):
        self.best = []
        self.worst = []
        self.confused = []

    def analyze(self, batch, pred, true, pred_label=None, paths=None):
        pred = as_multiclass_shape(pred, as_probs=True)
        for_label = pred.gather(1, true.unsqueeze(1))
        if pred_label is None:
            pred_label = pred.argmax(dim=1)
        if paths is None:
            paths = [None] * len(batch)
        this_data = list(zip(batch, for_label, true, pred_label == true,
                             paths, pred_label))

        self.best += this_data
        self.best.sort(key=lambda x: -x[1])
        self.best = self.best[:self.topk]

        self.worst += this_data
        self.worst.sort(key=lambda x: x[1])
        self.worst = self.worst[:self.topk]

        self.confused += this_data
        self.confused.sort(key=lambda x: abs(self.center_value - x[1]))
        self.confused = self.confused[:self.topk]

    def _report(self, dat):
        def prob_as_bar(cos):
            return '<div style="width:{}%;background-color:green;height:5px"></div>'.format(
                int(cos * 100))

        html = ['<div style="display:flex;flex-wrap:wrap">']
        for img, p, cls, correct, path, pred_label in dat:
            img = img - img.min()
            img /= img.max()
            html.append(
                ('<div onclick="javascript:prompt(\'path\', \'{}\')">'
                 '<div style="padding:3px;width:{}px">{}{}{}{} ({})</div>'
                 '</div>').format(
                     path, dat[0][0].shape[2], img2html(img),
                     prob_as_bar(p.item()), '✓' if correct.item() else '✗',
                     self.labels[cls.item()].replace('_',
                                                     ' ').replace('-', ' '),
                     self.labels[pred_label.item()].replace('_',
                                                     ' ').replace('-', ' ')))
        html.append('</div>')
        return ''.join(html)

    def show(self):
        html = [
            '<h1>Best predictions</h1>',
            self._report(self.best), '<h1>Worst predictions</h1>',
            self._report(self.worst), '<h1>Confusions</h1>',
            self._report(self.confused)
        ]
        return ''.join(html)


