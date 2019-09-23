"""
Recipes are a way to provide off the shelf algorithms that ce be either used
directly from the command line or easily imported in a python script if more
flexibility is needed.

A recipe should, as much a possible, be agnostic of the data and the underlying
model so that it can be used as a way to quickly try an algorithm on new data
or be easily experimented on by changing the model
"""
from torchelie.recipes.classification import Classification
from torchelie.recipes.deepdream import DeepDreamRecipe
from torchelie.recipes.feature_vis import FeatureVisRecipe
from torchelie.recipes.neural_style import NeuralStyleRecipe
from torchelie.recipes.trainandtest import TrainAndTest
from torchelie.recipes.trainandcallbase import TrainAndCallBase
from torchelie.recipes.trainandcall import TrainAndCall
