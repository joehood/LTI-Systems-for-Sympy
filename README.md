# Symbolic LTI System Tools for Python (ltisym) 

A module for symbolic manipulation of linear, time-invariant control systems. 


### Example Usage


```python
import sympy
import ltisym

# define a transfer function and convert to state space:

sympy.var('s a0 a1 a2 b0 b1 b2')

tf = ltisym.TransferFunction(
       (a2*s**2 + a1*s + a0) / 
    (s**3 + b2*s**2 + b1*s + b0)
)

ss = ltisym.StateSpace(tf)

print(tf)
print(ss)

# define a state space and convert to transfer function:

var('a11 a12 a21 a22 b1 b2 c1 c2 d1')

a = Matrix([[a11, a12],
            [a21, a22]])

b = Matrix([[b1],
            [b2]])

c = Matrix([[c1, c2]])

d = Matrix([[d1]])

ss = StateSpace(a, b, c, d)

tf = TransferFunction(ss)
            
print(ss)
print(tf)
```

#### Results:

```shell
tf to ss test
-------------

G(s)=

[                  2   ]
[  a0 + a1*s + a2*s    ]
[----------------------]
[                2    3]
[b0 + b1*s + b2*s  + s ]


A=

[-b2  -b1  -b0]
[             ]
[ 1    0    0 ]
[             ]
[ 0    1    0 ]

B=

[1]
[ ]
[0]
[ ]
[0]

C=

[a2  a1  a0]

D=

[0]


ss to tf test
-------------

A=

[a11  a12]
[        ]
[a21  a22]

B=

[b1]
[  ]
[b2]

C=

[c1  c2]

D=

[0]


G(s)=

[-(b1*(a21*c2 - c1*(a22 - s)) + b2*(a12*c1 - c2*(a11 - s))) ]
[-----------------------------------------------------------]
[               a12*a21 - (a11 - s)*(a22 - s)               ]
```


