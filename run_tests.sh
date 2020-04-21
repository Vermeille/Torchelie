#!/bin/bash

set -e

if false; then
    RUN="--device cuda --iters 1000 --visdom-env torch-test"
else
    RUN="--device cpu --iters 1"
fi

pytest tests
python3 -m torchelie.recipes.deepdream \
    --input tests/dream_me.jpg \
    --out tests/dreamed.png \
    $RUN

python3 -m torchelie.recipes.neural_style \
    --content tests/dream_me.jpg \
    --style tests/style.jpg \
    --out tests/styled.png \
    --size 512 \
    --ratio 300 \
    $RUN

python3 -m torchelie.recipes.feature_vis \
    --model resnet \
    --layer layer4 \
    --neuron 0 \
    $RUN

python3 examples/mnist.py
python3 examples/conditional.py
python3 examples/gan.py
python3 examples/pixelcnn.py

