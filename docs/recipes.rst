Recipes
=======

.. automodule:: torchelie.recipes

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

.. autoclass:: torchelie.recipes.deepdream.DeepDreamRecipe
   :members:
   :special-members: __call__

Feature visualization
~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/vis_example.jpg

.. automodule:: torchelie.recipes.feature_vis

.. autoclass:: torchelie.recipes.feature_vis.FeatureVisRecipe
   :members:
   :special-members: __call__

Neural Style
~~~~~~~~~~~~

.. image:: _static/style_example.png

.. automodule:: torchelie.recipes.neural_style

.. autoclass:: torchelie.recipes.neural_style.NeuralStyleRecipe
   :members:
   :special-members: __call__
