import numpy as np
x = np.random.rand(2)
print(x) 

np.random.seed(42)
y = np.random.rand(2)
print(y)
#you can see how the value of x changes every time the code runs, but y doesn't change