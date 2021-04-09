from setuptools import setup, find_packages

setup(
        name='Torchelie',
        version='0.1dev',
        packages=find_packages(),
        classifiers=[
            "License :: OSI Approved :: MIT License",
        ],
        install_requires=[
            'visdom>=0.1.8',
            'crayons>=0.2',
            'torchvision>=0.4',
            'numpy>=1.16',
            'Pillow>=6',
        ]
)
