from setuptools import setup, find_packages

setup(name='Torchelie',
      version='0.1dev',
      python_requires='>=3.10',
      extras_require={'visdom': ['visdom>=0.3.0', 'matplotlib>=3.5']},
      packages=find_packages(),
      classifiers=[
          "License :: OSI Approved :: MIT License",
      ],
      install_requires=[
          'trackio>=0.40,<1',
          'crayons>=0.2',
          'torchvision>=0.13',
          'torch>=2',
          'numpy>=1.16',
          'Pillow>=6',
      ])
