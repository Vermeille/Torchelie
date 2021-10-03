from typing import Optional
import torch
import torchelie.utils as tu


class Registry:

    def __init__(self):
        self.sources = ['https://s3.eu-west-3.amazonaws.com/torchelie.models']
        self.known_models = {}

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

    def register_decorator(self, f):
        def _f(*args, pretrained: Optional[str] = None, **kwargs):
            model = f(*args, **kwargs)
            if pretrained:
                ckpt = self.fetch(f'{pretrained}/{f.__name__}.pth')
                tu.load_state_dict_forgiving(model, ckpt)
            return model

        self.known_models[f.__name__] = _f

        return _f

    def get_model(self, name, *args, **kwargs):
        return self.known_models[name](*args, ** kwargs)


registry = Registry()
register = registry.register_decorator
get_model = registry.get_model

__all__ = ['Registry', 'register', 'get_model']
