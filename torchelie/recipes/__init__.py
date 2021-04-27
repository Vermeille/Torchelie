"""
Recipes are a way to provide off the shelf algorithms that ce be either used
directly from the command line or easily imported in a python script if more
flexibility is needed.

A recipe should, as much a possible, be agnostic of the data and the underlying
model so that it can be used as a way to quickly try an algorithm on new data
or be easily experimented on by changing the model
"""
from torchelie.recipes.recipebase import Recipe
from torchelie.recipes.classification import Classification
from torchelie.recipes.classification import CrossEntropyClassification
from torchelie.recipes.classification import MixupClassification
from torchelie.recipes.deepdream import DeepDream
from torchelie.recipes.feature_vis import FeatureVis
from torchelie.recipes.neural_style import NeuralStyle
from torchelie.recipes.trainandtest import TrainAndTest
from torchelie.recipes.trainandcall import TrainAndCall
from torchelie.recipes.gan import GANRecipe
