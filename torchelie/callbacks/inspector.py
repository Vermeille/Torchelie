from io import BytesIO
import base64 as b64
from typing import Union, List, Optional

import torch
from torchelie.utils import as_multiclass_shape, experimental

import numpy as np
from PIL import Image


def img2html(img: Union[torch.Tensor, np.ndarray], opts: dict = None) -> str:
    """
    Convert an image to a b64 html inline image.

    Possible options: width (int, pixels), height (int, pixels) or jpgquality
    (int, percentage)
    """
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
    """
    Visdom HTML display of classification results, with best, worst, and mode
    indecisive results.
    """
    def __init__(self,
                 topk: int,
                 labels: List[str],
                 center_value: float = 0) -> None:
        self.labels = labels
        self.center_value = center_value
        self.topk = topk
        self.reset()
        self.m = float('inf')
        self.M = float('-inf')

    def reset(self):
        self.best = []
        self.worst = []
        self.confused = []

    def analyze(self,
                batch: torch.Tensor,
                pred: torch.Tensor,
                true: torch.Tensor,
                pred_label: Optional[torch.Tensor] = None,
                paths: Optional[List[str]] = None) -> None:
        pred = as_multiclass_shape(pred, as_probs=True)
        for_label = pred.gather(1, true.unsqueeze(1))
        if pred_label is None:
            pred_label = pred.argmax(dim=1)
        best_pred = pred.gather(1, pred_label.unsqueeze(1))
        if paths is None:
            paths = [None] * len(batch)
        this_data = list(
            zip(batch, for_label, true, pred_label == true, paths, pred_label,
                best_pred))

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
        def prob_as_bar(cos, is_correct):
            return '<div style="width:{percents}%;background-color:{color};height:5px"></div>'.format(
                percents=int(cos * 100),
                color='green' if is_correct else 'red')

        html = ['<div style="display:flex;flex-wrap:wrap">']
        for img, p, cls, correct, path, pred_label, best_pred in dat:
            self.m = min(self.m, img.min())
            img = img - self.m
            self.M = max(self.M, img.max())
            img /= self.M

            tlabel = self.labels[cls.item()]
            tlabel = tlabel.replace('_', ' ').replace('-', ' ')

            plabel = self.labels[pred_label.item()]
            plabel = plabel.replace('_', ' ').replace('-', ' ')
            html.append(
                ('<div onclick="javascript:prompt(\'path\', \'{path}\')">'
                 '<div style="padding:3px;width:{px_sz}px">'
                 '{img}{bar}{checkmark}{true_label} (pred: {pred_label})'
                 '</div>'
                 '</div>').format(path=path,
                                  px_sz=dat[0][0].shape[2],
                                  img=img2html(img),
                                  bar=prob_as_bar(best_pred.item(),
                                                  correct.item()),
                                  checkmark='✓' if correct.item() else '✗',
                                  true_label=tlabel,
                                  pred_label=plabel))
        html.append('</div>')
        return ''.join(html)

    def show(self) -> str:
        """
        Get the HTML inspector view.
        """
        html = [
            '<h1>Best predictions</h1>',
            self._report(self.best), '<h1>Worst predictions</h1>',
            self._report(self.worst), '<h1>Confusions</h1>',
            self._report(self.confused)
        ]
        return ''.join(html)


class SegmentationInspector(ClassificationInspector):
    @experimental
    def __init__(self, topk, labels, center_value=0):
        super().__init__(topk, labels, center_value=0)

    def analyze(self,
                batch: torch.Tensor,
                pred: torch.Tensor,
                true: torch.Tensor,
                pred_label: Optional[torch.Tensor] = None,
                paths: Optional[List[str]] = None) -> None:
        pred = torch.sigmoid(pred)
        for_label = torch.median(
            (pred * true + (1 - pred) * (1 - true)).reshape(pred.shape[0],
                                                            -1), -1)[0]
        if pred_label is None:
            pred_label = (pred > 0.5).int()
        if paths is None:
            paths = [None] * len(batch)
        this_data = list(
            zip(batch, for_label,
                true.mean(tuple(range(1, true.dim()))) > 0.5,
                (pred_label == true).float().mean(
                    tuple(range(1, pred_label.dim()))) > 0.5, paths))

        self.best += this_data
        self.best.sort(key=lambda x: -x[1])
        self.best = self.best[:self.topk]

        self.worst += this_data
        self.worst.sort(key=lambda x: x[1])
        self.worst = self.worst[:self.topk]

        self.confused += this_data
        self.confused.sort(key=lambda x: abs(self.center_value - x[1]))
        self.confused = self.confused[:self.topk]
