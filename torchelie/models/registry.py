from typing import Optional
import torch
import torchelie.utils as tu


class Registry:

    def __init__(self):
        self.sources = ['https://s3.eu-west-3.amazonaws.com/torchelie.models']

    def from_source(self, src: str, model: str) -> dict:
        uri = f'{src}/{model}'
        if uri.lower().startswith('http'):
            return torch.hub.load_state_dict_from_url(uri, map_location='cpu',
                    file_name=model.replace('/', '.'))
        else:
            return torch.load(uri, map_location='cpu')

    def fetch(self, model: str) -> dict:
        for source in reversed(self.sources):
            try:
                return self.from_source(source, model)
            except Exception as e:
                print(f'{model} not found in source {source}, next', str(e))
        raise Exception(f'No source contains pretrained model {model}')

    def pretrained_decorator(self, f):
        """
        FIXME: early optimization is the root of all evil but this is clearly
        technical debt
        """

        def _f(*args, pretrained: Optional[str], **kwargs):
            model = f(*args, **kwargs)
            if pretrained:
                ckpt = self.fetch(f'{pretrained}/{f.__name__}.pth')
                tu.load_state_dict_forgiving(model, ckpt)
            return model

        return _f


registry = Registry()
pretrained = registry.pretrained_decorator
