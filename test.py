"""ltisym test functions.
"""

import sympy
import ltisym

sympy.init_printing()

def test_tf():

    print("2nd order tf test\n")

    sympy.var('s K z omega_0')

    tf = ltisym.TransferFunction(K / (s**2 + 2*z*omega_0*s + omega_0**2))

    print(tf)

    ss = ltisym.StateSpace(tf)

    print(ss)

def test_tf2ss():

    print("tf to ss test\n")

    sympy.var('s a0 a1 a2 b0 b1 b2')

    tf = ltisym.TransferFunction(
        (a2*s**2 + a1*s + a0) / 
     (s**3 + b2*s**2 + b1*s + b0)
    )

    ss = ltisym.StateSpace(tf)

    print(tf)
    print(ss)

                                               
def test_ss2tf():

    print("ss to tf test\n")

    sympy.var('a11 a12 a21 a22 b1 b2 c1 c2 d1')

    a = sympy.Matrix([[a11, a12],
                      [a21, a22]])

    b = sympy.Matrix([[b1],
                      [b2]])

    c = sympy.Matrix([[c1, c2]])

    d = sympy.Matrix([[d1]])

    ss = ltisym.StateSpace(a, b, c, d)

    tf = ltisym.TransferFunction(ss)
            
    print(ss)
    print(tf)

if __name__ == "__main__":

    test_tf()
    test_tf2ss()
    test_ss2tf()



