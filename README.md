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

ss = ltisym.StateSpace(a, b, c, d)

tf = ltisym.TransferFunction(ss)
            
print(ss)
print(tf)


# Compound system from block diagram components (cascade example):

"""
IEEE AC8B Exciter (simplified)
                
            .--------.                                               
vref     .->|  Kpr   |----.                                        
  |      |  '--------'    |                              
+ v      |              + v       .--------.      .--------. 
 ,-.  e  |  .--------. + ,-.  pid |   Ka   |      |   Ke   | 
( S )----+->| Kir/s  |->( S )---->| ------ |----->| ------ |---> vfd
 `-'     |  '--------'   `-'      | 1+s*Ta |  vr  | 1+s*Te | 
- ^      |  .--------.  + ^       '--------'      '--------' 
  |      |  | s*Kdr  |    |          (x2)            (x3)
 vt      '->| ------ |----'                        
            | 1+s*Tr |                             
            '--------'                             
             (x0, x1)                                                  
"""

sympy.var('s Kpr Kir Kdr Tdr Ka Ta Ke Te')

# define component transfer functions:

tf1 = ltisym.TransferFunction(Kpr + Kir/s + Kdr*s / (1 + s*Tdr))  
tf2 = ltisym.TransferFunction(Ka / (1 + s*Ta))                    
tf3 = ltisym.TransferFunction(Ke / (1 + s*Te))                    

# convert to state space models:

ss1 = ltisym.StateSpace(tf1)
ss2 = ltisym.StateSpace(tf2)
ss3 = ltisym.StateSpace(tf3)

# connect systems using the StateSpace.cascade function:

ss_ac8b = ss1.cascade(ss2).cascade(ss3)

# create transfer function version of the whole system:

tf_ac8b = ltisym.TransferFunction(ss_ac8b)

# display models:

print(tf_ac8b)
print(ss_ac8b)

```

#### Results:

```shell
tf to ss test
-------------

G(s) =

                  2   
  a0 + a1*s + a2*s    
----------------------
                2    3
b0 + b1*s + b2*s  + s 


A =

[-b2  -b1  -b0]
[             ]
[ 1    0    0 ]
[             ]
[ 0    1    0 ]

B =

[1]
[ ]
[0]
[ ]
[0]

C= [a2  a1  a0]

D= [0]


ss to tf test
-------------

A =

[a11  a12]
[        ]
[a21  a22]

B =

[b1]
[  ]
[b2]

C = [c1  c2]

D = [0]


G(s) =

-(b1*(a21*c2 - c1*(a22 - s)) + b2*(a12*c1 - c2*(a11 - s))) 
-----------------------------------------------------------
               a12*a21 - (a11 - s)*(a22 - s)               


cascade test
------------

A =

⎡    -1                     ⎤
⎢    ───        0    0    0 ⎥
⎢    Tdr                    ⎥
⎢                           ⎥
⎢     1         0    0    0 ⎥
⎢                           ⎥
⎢  Kdr         Kir  -1      ⎥
⎢- ──── + Kir  ───  ───   0 ⎥
⎢     2        Tdr   Ta     ⎥
⎢  Tdr                      ⎥
⎢                           ⎥
⎢                   Ka   -1 ⎥
⎢     0         0   ──   ───⎥
⎣                   Ta    Te⎦

B =

⎡    1    ⎤
⎢         ⎥
⎢    0    ⎥
⎢         ⎥
⎢Kdr      ⎥
⎢─── + Kpr⎥
⎢Tdr      ⎥
⎢         ⎥
⎣    0    ⎦

C =

⎡         Ke⎤
⎢0  0  0  ──⎥
⎣         Te⎦

D = [0]


```


