"""
SUPER ALPHA
"""

import random
import os

from typing import List
import PIL


class ImageLogger:
    """
    Logs images and decisions up to a capacity per class
    """
    def __init__(self, root:str, classes:List[str], capacity:int=200):
        """
        Initialize all classes

        Args:
            self: (todo): write your description
            root: (str): write your description
            classes: (todo): write your description
            capacity: (int): write your description
        """
        self.root = root
        self.classes = classes
        self.capacity = capacity
        for kls in classes:
            os.makedirs(root + '/' + kls, exist_ok=True)

    def __call__(self, images, klass:List[int]) -> None:
        """
        Call klass.

        Args:
            self: (todo): write your description
            images: (list): write your description
            klass: (todo): write your description
        """
        for i, k in zip(images, klass):
            k_name = self.classes[k]
            n = random.randint(-int(self.capacity*0.25), self.capacity)
            if n < 0:
                continue
            path = f'{self.root}/{k_name}/{n}.jpg'
            if isinstance(i, PIL.Image.Image):
                i.save(path)
