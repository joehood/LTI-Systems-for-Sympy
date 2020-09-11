"""ltisym test functions.
"""

from sympy import Matrix, var
import ltisym

from ltisym import StateSpace
from ltisym import TransferFunction

def test_tf2ss():

    print("tf to ss test\n-------------")

    var('s a0 a1 a2 b0 b1 b2')

    tf = TransferFunction(
        (a2*s**2 + a1*s + a0) / 
     (s**3 + b2*s**2 + b1*s + b0)
    )

    ss = StateSpace(tf)

    print(tf)
    print(ss)

                                               
def test_ss2tf():

    print("ss to tf test\n-------------")

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

if __name__ == "__main__":

    test_tf2ss()
    test_ss2tf()



