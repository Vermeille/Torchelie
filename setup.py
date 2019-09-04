from setuptools import setup, find_packages

setup(
        name='Torchelie',
        version='0.1dev',
        packages=find_packages(),
        license='Creative Commons Attribution-Noncommercial-Share Alike license',
        install_requires=[
            'visdom>=0.1.8',
            'crayons>=0.2',
            'torchvision>=0.4',
            'numpy>=1.16',
            'Pillow>=6',
            'onnx>=1.5',
        ]
)
