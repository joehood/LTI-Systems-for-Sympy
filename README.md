# Symbolic LTI System Tools for Python (ltisym) 

A package for linear, time invariant control systems for symbolic python. 

## Installation

There is a python installer for this package. You can download the *.zip* file in the `dist` folder (which may not be up-to-date) or clone the whole repository using
```
(coming soon)
```
After extracting, cd to the direcory and run
```
python setup.py install
```

### Creating and converting between models
You can create a State Space and Transfer Function models symbolically, and convert bewteen them:
```python
from sympy import *
from lti_systems.models import StateSpaceModel as ss
from lti_systems.models import TransferFunctionModel as tf

var('s a0 a1 a2 b0 b1 b2')

sfunc = (a2*s**2 + a1*s + a0)/(s**3 + b2*s**2 + b1*s + b0)

sys1 = tf(Matrix([sfunc]))

sys2 = ss(sys1)

print("A = {}".format(sys2.represent[0]))
print("B = {}".format(sys2.represent[1]))
print("C = {}".format(sys2.represent[2]))
print("D = {}".format(sys2.represent[3]))
```

```code
result:

A = Matrix([[-b2, -b1, -b0], [1, 0, 0], [0, 1, 0]])
B = Matrix([[1], [0], [0]])
C = Matrix([[a2, a1, a0]])
D = Matrix([[0]])
```
