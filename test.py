import numpy as np
x = np.random.rand(2,5)
print(x) 

np.random.seed(42)
y = np.random.randn(2,5)
print(y)
#you can see how the value of x changes every time the code runs, but y doesn't change