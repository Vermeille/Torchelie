from .trainandtest import TrainAndTest


def TrainAndCall(model,
                 train_fun,
                 test_fun,
                 train_loader,
                 test_every=100,
                 visdom_env='main',
                 checkpoint='model',
                 log_every=10):
    """
    Train a model and evaluate it with a custom function. The model is
    automatically checkpointed, VisdomLogger and StdoutLogger callbacks are
    also already provided. The gradients are disabled and the model is
    automatically set to evaluation mode for the evaluation procedure.

    Args:
        model (nn.Model): a model
        train_fun (Callabble): a function that takes a batch as a single
            argument, performs a training step and return a dict of values to
            populate the recipe's state.
        test_fun (Callable): a function taking no argument that performs
            something to evaluate your model and returns a dict to populate the
            state.
        train_loader (DataLoader): Training set dataloader
        test_every (int): testing frequency, in number of iterations (default:
            100)
        visdom_env (str): name of the visdom environment to use, or None for
            not using Visdom (default: None)
        checkpoint (str): checkpointing path or None for no checkpointing
        log_every (int): logging frequency, in number of iterations (default:
            10)

    Returns:
        a configured Recipe
    """

    def test_fun_wrap(_):
        return test_fun()

    return TrainAndTest(model,
                        train_fun,
                        test_fun_wrap,
                        train_loader=train_loader,
                        test_loader=range(1),
                        test_every=test_every,
                        visdom_env=visdom_env,
                        checkpoint=checkpoint,
                        log_every=log_every)
