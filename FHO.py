import numpy as np
from optimizer import Optimizer
from Detection import TriGuardNet
from save_load import load


def fit_func_70(x):
    x_train = load('x_train_70')
    x_test = load('x_test_70')
    y_train = load('y_train_70')
    y_test = load('y_test_70')

    epochs = int(x[0])
    learning_rate = x[2]
    batch_size = int(x[1])

    pred, met, history = TriGuardNet(x_train, y_train, x_test, y_test, epochs, batch_size, learning_rate)

    fit = 1 / met[0]

    return fit


def fit_func_80(x):
    x_train = load('x_train_80')
    x_test = load('x_test_80')
    y_train = load('y_train_80')
    y_test = load('y_test_80')

    epochs = int(x[0])
    learning_rate = x[2]
    batch_size = int(x[1])

    pred, met, history = TriGuardNet(x_train, y_train, x_test, y_test, epochs, batch_size, learning_rate)

    fit = 1 / met[0]

    return fit


# Fire Hawk Optimization
class OriginalFHO(Optimizer):
    """
    The original version of: Fire Hawk Optimization (FHO)
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.n_firehawks = np.random.choice(range(1, int(self.pop_size/5)+1))       # Maximum number of FireHawks
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

