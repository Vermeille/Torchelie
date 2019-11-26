Recipes
=======

.. automodule:: torchelie.recipes

.. autoclass:: torchelie.recipes.recipebase.Recipe
   :members:
   :inherited-members:

.. autofunction:: torchelie.recipes.trainandcall.TrainAndCall

.. autofunction:: torchelie.recipes.trainandTest.TrainAndTest

.. autofunction:: torchelie.recipes.classification.Classification

.. autofunction:: torchelie.recipes.classification.CrossEntropyClassification

.. autofunction:: torchelie.recipes.gan.GANRecipe

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
