Recipes
=======

.. automodule:: torchelie.recipes

.. autoclass:: torchelie.recipe.recipebase.Recipe
   :members:
   :inherited-members:

.. autoclass:: torchelie.recipe.trainandcall.TrainAndCall
   :members:

.. autoclass:: torchelie.recipe.trainandTest.TrainAndTest
   :members:

.. autoclass:: torchelie.recipe.classification.Classification
   :members:

.. autoclass:: torchelie.recipe.classification.CrossEntropyClassification
   :members:

.. autoclass:: torchelie.recipe.gan.GANRecipe
   :members:

(unstable)

Model Training
~~~~~~~~~~~~~~

.. autoclass:: torchelie.recipes.trainandtest.TrainAndTest
   :members:

.. autoclass:: torchelie.recipes.classification.Classification
   :members:

.. autoclass:: torchelie.recipes.classification.CrossEntropyClassification
   :members:

Deep Dream
~~~~~~~~~~

.. image:: _static/dream_example.jpg

.. automodule:: torchelie.recipes.deepdream

.. autoclass:: torchelie.recipes.deepdream.DeepDream
   :members:
   :special-members: __call__

Feature visualization
~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/vis_example.jpg

.. automodule:: torchelie.recipes.feature_vis

.. autoclass:: torchelie.recipes.feature_vis.FeatureVis
   :members:
   :special-members: __call__

Neural Style
~~~~~~~~~~~~

.. image:: _static/style_example.png

.. automodule:: torchelie.recipes.neural_style

.. autoclass:: torchelie.recipes.neural_style.NeuralStyle
   :members:
   :special-members: __call__

Deep Image Prior
~~~~~~~~~~~~~~~~

.. automodule:: torchelie.recipes.image_prior
