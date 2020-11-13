"""ltisym test functions.
"""

import sympy
import ltisym

sympy.init_printing()

def test1():

    print("2nd order tf test\n")

    sympy.var('s K z omega_0')

    tf = ltisym.TransferFunction(K / (s**2 + 2*z*omega_0*s + omega_0**2))

    print(tf)

    ss = ltisym.StateSpace(tf)

    print(ss)

def test2():

    print("tf to ss test\n")

    sympy.var('s a0 a1 a2 b0 b1 b2')

    tf = ltisym.TransferFunction(
        (a2*s**2 + a1*s + a0) / 
     (s**3 + b2*s**2 + b1*s + b0)
    )

    ss = ltisym.StateSpace(tf)

    print(tf)
    print(ss)

                                               
def test3():

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


def test3():

    """IEEE AC8B Exciter (simplified)
                   
                .--------.                                                
     vref   .-->|  Kpr   |-----.                                        
      |     |   '--------'     |                              
    + v     |                + v       .--------.      .--------. 
     ,-.    |   .--------.  + ,-.      |   Ka   |      |   Ke   | 
    ( Σ )---+-->| Kir/s  |-->( Σ )---->| ------ |----->| ------ |---> vfd
     `-'    |   '--------'    `-'      | 1+s*Ta |      | 1+s*Te | 
    - ^     |   .--------.   + ^       '--------'      '--------' 
      |     |   | s*Kdr  |     |          (x3)            (vfd)
     vterm  '-->| ------ |-----'                        
                | 1+s*Tr |                              
                '--------'                              
                 (x1, x2)                                     
                           
    """

    # create symbolic variables:
    sympy.var('s Kpr Kir Kdr Tdr Ka Ta Ke Te')

    # define symbolic component transfer functions:
    tf1 = ltisym.TransferFunction(Kpr + Kir/s + Kdr*s / (1 + s*Tdr))  # pid
    tf2 = ltisym.TransferFunction(Ka / (1 + s*Ta))                    # lag 1
    tf3 = ltisym.TransferFunction(Ke / (1 + s*Te))                    # lag 2

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


if __name__ == "__main__":

    #test1()
    #test2()
    #test3()
    test3()



