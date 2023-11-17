import numpy as np

from classes.parameters import ParameterGenerator

# random number generator
rng = np.random.RandomState(1)

# create a parameter generator
param_gen = ParameterGenerator()

# generate a parameter set
for i in range(10):
    param = param_gen.generate(rng)
    print(param)

