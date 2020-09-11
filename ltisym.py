"""Symbolic manipulation and numerical evaluation of linear time-invariant (LTI) systems.
"""

import sympy as sp
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad


# ========================= LINALG UTIL FUNCTIONS ============================


def matrix_degree(mat, sym):

    """returns the highest sp.degree of any entry in m with respect to s

    mat: sp.Matrix to get sp.degree from
    sym: sp.Symbol to get sp.degree from (sp.degree can be ambiguous with multiple coefficients in a expression)
    """

    return max(mat.applyfunc(lambda en: sp.degree(en, sym)))


def matrix_coeff(m, s):
    
    """returns the sp.Matrix valued coefficients N_i in m(x) = N_1 * x**(n-1) + N_2 * x**(n-2) + .. + N_deg(m)

    m : sp.Matrix
        sp.Matrix to get coefficient matrices from
    s :
        sp.Symbol to compute coefficient list (coefficients are ambiguous for expressins with multiple symbols)
    """

    m_deg = matrix_degree(m, s)
    res = [sp.zeros(m.shape[0], m.shape[1])] * (m_deg + 1)

    for r, row in enumerate(m.tolist()):
        for e, entry in enumerate(row):

            entry_coeff_list = sp.Poly(entry, s).all_coeffs()
            if sp.simplify(entry) == 0:
                coeff_deg = 0
            else:
                coeff_deg = sp.degree(entry, s)

            for c, coeff in enumerate(entry_coeff_list):
                res[c + m_deg - coeff_deg] += \
                    sp.SparseMatrix(m.shape[0], m.shape[1], {(r, e): 1}) * coeff
    return res


def fraction_list(m, only_denoms=False, only_numers=False):

    """list of fractions of m

    retuns a list of tuples of the numerators and denominators of all entries of m.
    the entries of m can be any sort of expressions.
    result[i*j + j][0/1] is the numerator/denominator of the sp.Matrix element m[i,j]

    Parameters
    ==========

    m : sp.Matrix
        the sp.Matrix we want the list of fraction from

    Flags
    =====

    only_denoms=False : Bool
        if True, sp.Function only returns a list of denominators, not tuples
    only_numers)False: Bool
        if True, sp.Function only returns a list of nmerators, not tuples

    """

    if (only_denoms is True) and (only_numers is True):
        raise ValueError(
            "at least one of only_denoms and only_numers must be False")

    if only_denoms is True:
        return map(lambda x: x.as_numer_denom()[1], m)
    if only_numers is True:
        return map(lambda x: x.as_numer_denom()[0], m)
    return map(lambda x: x.as_numer_denom(), m)


def is_proper(m, s, strict=False):

    """is_proper

    tests if the sp.degree of the numerator does not exceed the sp.degree of the denominator
    for all entries of a given sp.Matrix.

    Parameters
    ==========

    m : sp.Matrix
        sp.Matrix to test if proper

    Flags
    =====

    strict = False
        if rue, the sp.Function returns True only if the sp.degree of the denominator is always greater
        than the sp.degree of the numerator
    """

    if strict is False:
        return all(sp.degree(en.as_numer_denom()[0], s) <=
                   sp.degree(en.as_numer_denom()[1], s) for en in m)
    else:
        return all(sp.degree(en.as_numer_denom()[0], s) <
                   sp.degree(en.as_numer_denom()[1], s) for en in m)


# =========================== LTI SYSTEM MODELS ==============================


class StateSpace(object):

    """state space model (ssm) of a linear, time invariant control system

    Represents the standard state-space model with state sp.Matrix A, input sp.Matrix B, output sp.Matrix C, and
    transmission sp.Matrix D. This makes the linear controll system:
        (1) x'(t) = A * x(t) + B * u(t);    x in R^n , u in R^k
        (2) y(t)  = C * x(t) + D * u(t);    y in R^m
    where u(t)  is any input signal, y(t) the corresponding output, and x(t) the systems state.

    Parameters
    ==========

    arg : TransferFunction, List of Sympy-sp.Matrix
        tfm to construct the state space model from, or the Matrices A,B,C,D in a list

    See Also
    ========

    TranferFunctionModel: transfer sp.Function model of a lti system

    References
    ==========

    Joao P. Hespanha, Linear Systems Theory. 2009.
    """

    def __init__(self, a, b=None, c=None, d=None):

        if isinstance(a, TransferFunction):

            # call the private method for realization finding
            self.represent = self._find_realization(a.G, a.s)

            # create a block sp.Matrix [[A,B], [C,D]] for visual representation
            self.BlockRepresent = sp.BlockMatrix([[self.represent[0], self.represent[1]],
                                               [self.represent[2], self.represent[3]]])
            return None

        if isinstance(a, sp.Matrix):

            if b and c and d:
                
                self.represent = [a, b, c, d]

            if b and c:

                d = sp.zeros(c.shape[0], b.shape[1])

                self.represent = [a, b, c, d]

        else:

            # store the argument as representation of the system

            try:
                self.represent = arg[:4]

            except TypeError:
                raise TypeError("'repesentation' must be a list-like object")

            try:
                # assert that A,B,C,D have matching shapes
                if not ((self.represent[0].shape[0] == self.represent[1].shape[0]) and
                        (self.represent[0].shape[1] == self.represent[2].shape[1]) and
                        (self.represent[1].shape[1] == self.represent[3].shape[1]) and
                        (self.represent[2].shape[0] == self.represent[3].shape[0])):
                    raise sp.ShapeError("Shapes of A, B, C, D must fit")

                # create a block sp.Matrix [[A,B], [C,D]] for visual representation
                self.BlockRepresent = sp.BlockMatrix([[self.represent[0], self.represent[1]],
                                                   [self.represent[2], self.represent[3]]])
                return None

            except TypeError:
                raise TypeError("entries of 'representation' must be matrices")

            except AttributeError:
                raise TypeError("entries of 'representation' must be matrices")

            except IndexError:
                raise TypeError("'representation' must have at least 4 sp.Matrix-valued entries")

    def _find_realization(self, G, s):
        
        """ Represenatation [A, B, C, D] of the state space model

        Returns the representation in state space of a given transfer sp.Function

        Parameters
        ==========

        G: sp.Matrix
            sp.Matrix valued transfer sp.Function G(s) in laplace space
        s: sp.Symbol
            variable s, where G is dependent from

        See Also
        ========

        Utils : some quick tools for sp.Matrix polynomials

        References
        ==========

        Joao P. Hespanha, Linear Systems Theory. 2009.
        """

        A, B, C, D = 4 * [None]

        try:
            m, k = G.shape

        except AttributeError:
            raise TypeError("G must be a sp.Matrix")

        # test if G is proper
        if not is_proper(G, s, strict=False):
            raise ValueError("G must be proper!")

        # define D as the limit of G for s to infinity
        D = G.limit(s, sp.oo)

        # define G_sp as the (stricly proper) difference of G and D
        G_sp = sp.simplify(G - D)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # get the coefficients of the monic least common denominator of all entries of G_sp
        # compute a least common denominator using utl and sp.lcm
        lcd = sp.lcm(fraction_list(G_sp, only_denoms=True))

        # make it monic
        lcd = sp.simplify(lcd / sp.LC(lcd, s))

        # and get a coefficient list of its monic. The [1:] cuts the sp.LC away (thats a one)
        lcd_coeff = sp.Poly(lcd, s).all_coeffs()[1:]

        # get the sp.degree of the lcd
        lcd_deg = sp.degree(lcd, s)

        # get the sp.Matrix Valued Coeffs of G_sp in G_sp = 1/lcd * (N_1 * s**(n-1) + N_2 * s**(n-2) .. +N_n)
        G_sp_coeff = matrix_coeff(sp.simplify(G_sp * lcd), s)
        G_sp_coeff = [sp.zeros(m, k)] * (lcd_deg - len(G_sp_coeff)) + G_sp_coeff

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # now store A, B, C, D in terms of the coefficients of lcd and G_sp
        # define A
        A = (-1) * lcd_coeff[0] * sp.eye(k)

        for alpha in lcd_coeff[1:]:
            A = A.row_join((-1) * alpha * sp.eye(k))

        for i in xrange(lcd_deg - 1):
            if i == 0:
                tmp = sp.eye(k)
            else:
                tmp = sp.zeros(k)

            for j in range(lcd_deg)[1:]:
                if j == i:
                    tmp = tmp.row_join(sp.eye(k))
                else:
                    tmp = tmp.row_join(sp.zeros(k))
            if tmp is not None:
                A = A.col_join(tmp)

        # define B
        B = sp.eye(k)
        for i in xrange(lcd_deg - 1):
            B = B.col_join(sp.zeros(k))

        # define C
        C = G_sp_coeff[0]
        for i in range(lcd_deg)[1:]:
            C = C.row_join(G_sp_coeff[i])

        # return the state space representation
        return [sp.simplify(A), sp.simplify(B), sp.simplify(C), sp.simplify(D)]

    def evaluate(self, u, x0, t, t0=0, method=None, return_pretty=False, do_integrals=True, dps=5):
        
        """evaluate the system output for an input u

        The output of the system y for the output u if given by solving the state equation for x
        and than substituting that into the output equation

        Parameters
        ==========

        u  : one-column sp.Matrix
            The input vector in time-space
        x0 : one-column sp.Matrix
            the state of the system at time t0
        t  : sp.Symbol, tuple (t,[list of times])
            if t is only a sp.Symbol, the system is evaluated simbolycaly.
            if t is a tuple of a sp.Symbol and a list, the symstem is evaluated numericaly, at the given times in the list
        t0 = 0 : number
            the time t0 at which the state of the system is known

        method : Bool
            not supported yet, always uses diagonalizaton
        return_pretty : Bool
            if True, the funtion returns a tuple of equations showing the input, initial conditions, and the output
        do_integrals : Bool
            if True, the sp.Function tries to evaluate the integrals in the solution. if False, it returns an
            sp.Integral object instead. Only valid for symbolic solutions, ignored otherwise
        dps : integer
            the decimal precision of numericial integration

        References
        ==========

        Joao P. Hespanha, Linear Systems Theory. 2009.
        """

        try:
            # assert right shape of u
            if not u.shape[1] == 1:
                raise sp.ShapeError("u must not have more that one column, but has shape", u.shape)

            if not self.represent[3].shape[1] == u.shape[0]:
                raise sp.ShapeError("u must have length", self.represent[3].shape[1])

            # assert right shape of x0
            if not x0.shape[1] == 1:
                raise sp.ShapeError("x0 must not have more than one column, but has shape", x0.shape)

            if not self.represent[0].shape[1] == x0.shape[0]:
                raise sp.ShapeError("x0 must have length", self.represent[0].shape[1])

        except AttributeError:
            raise TypeError("u and x0 must be matrices!")

        # find out if t is sp.Symbol, tuple or given wrong and call subroutines
        # accordingly to that:

        sol = None

        try:

            # if t symobl, then calculate the solution symbolicaly
            if isinstance(t, sp.Symbol):
                sol = self._solve_symbolicaly(u, x0, t, t0, do_integrals=do_integrals)

            # if not, try if it is tuple, list or sth.
            elif isinstance(t[0], sp.Symbol):
                # if t[1] is a direct subclass of tuple or list
                if isinstance(t[1], (list, tuple)):

                    # use the private member sp.Function of the class to compute the numervial result
                    sol = self._solve_numerically(u, x0, t[0], t[1], t0, dps=dps)

                #  if its not, try to convert it
                else:
                    sol = self._solve_numerically(u, x0, t[0], list(t[1]), t0, dps=dps)

        # index error can occure if t is not list-like:

        except IndexError:
                IndexError("t must be sp.Symbol or have at least 2 entries")

        # if the conversion goes wrong, its (hopefully) a TypeError:

        except TypeError:
                TypeError("t[1] must be list, or list(t[1]) must work")

        # if that worked, return the prestored solution:

        if return_pretty is True:

            y = sp.Function('y')
            u_ = sp.Function('u')
            x_ = sp.Function('x')

            return (sp.Eq(u_(t), u), sp.Eq(x_(0), x0), sp.Eq(y(t), sol))

        else:
            return sol

    def _solve_numerically(self, u, x0, t, t_list, t0, dps=2):
        
        """ returns the numeric evaluation of the system for input u, know state x0 at time t0 and times t_list
        """

        result = []

        for t_i in t_list:

            # we use the arbitrary precision module mpmath for numercial evaluation of the sp.Matrix exponentials
            first = np.array(np.array(self.represent[2]), np.float).dot(
                expm(np.array(np.array((self.represent[0] * (t_i - t0)).evalf()), np.float))
            ).dot(
                np.array(np.array(x0), np.float)
            )

            second = np.array(np.array((self.represent[3] * u.subs(t, t_i)).evalf()), np.float)

            integrand = lambda tau: \
                np.array(np.array(self.represent[2]), np.float).dot(
                    expm(np.array(np.array((self.represent[0] * (t_i - tau)).evalf()), np.float))
                ).dot(
                    np.array(np.array(self.represent[1]), np.float)
                ).dot(
                    np.array(np.array(u.subs(t, tau).evalf()), np.float)
                )

            # the result must have the same shape as D:
            sp.Integral = sp.zeros(self.represent[2].rows, 1)

            # Loop through every entry and evaluate the sp.Integral using mpmath.quad()
            for row_idx in xrange(self.represent[2].rows):

                sp.Integral[row_idx, 0] = quad(lambda x: integrand(x)[row_idx, 0], t0, t_i)[0]

            result.append(sp.Matrix(first) + sp.Matrix(second) + sp.Integral)

        # return sum of results
        return result

    def _solve_symbolicaly(self, u, x0, t, t0, exp_method="diagonalize", do_integrals=True):
        
        """ returns the symbolic evaluation of the system for input u and known state x0 at time t0
        """

        # set the valid methods for the sp.Matrix exponential
        # TODO: Laplace Transform

        valid_methods = ("diagonalize")

        if exp_method not in valid_methods:
            raise ValueError("unknown method for sp.Matrix exponential:", exp_method)

        # define temporary symbols tau

        tau = sp.Symbol('tau', positive=True)
        x = sp.Symbol('x')

        # compute the two sp.Matrix exponentials that are used in the general solution
        # to avoid two eigenvalue problems, first solve for a general real x and substitude then
        
        expAx = sp.simplify(sp.exp(self.represent[0] * x))
        expA = sp.simplify(expAx.subs(x, t - t0))
        expAt = sp.simplify(expAx.subs(x, t - tau))

        # define the sp.Integral and heuristic simplification nowing that in the sp.Integral, tau < t always holds
        
        integrand = sp.simplify(self.represent[2] * expAt * self.represent[1] * u.subs(t, tau))
        integrand = sp.simplify(integrand.subs([(abs(t - tau), t - tau), (abs(tau - t), t - tau)]))
        sp.Integral = sp.zeros(integrand.shape[0], integrand.shape[1])

        for col_idx in xrange(integrand.cols):

            for row_idx in xrange(integrand.rows):
                try:
                    if not integrand[row_idx, col_idx] == 0:
                        if do_integrals is True:
                            sp.Integral[row_idx, col_idx] = sp.simplify(sp.integrate(integrand[row_idx, col_idx], (tau, t0, t)))
                        else:
                            sp.Integral[row_idx, col_idx] = sp.Integral(integrand, (tau, t0, t))
                except:
                    sp.Integral[row_idx, col_idx] = sp.Integral(integrand, (tau, t0, t))

        # return the general solution:
        return sp.simplify(self.represent[2] * expA * x0 + self.represent[3] * u + sp.Integral)

    def cotrollability_matrix(self):

        """ Returns the controllability sp.Matrix of the system:
            C = [B, A * B, A^2 * B, .. , A^(n-1), B]; A in R^(n x n), B in^R^(n x m)
        """

        res = self.represent[1]
        for i in xrange(self.represent[0].shape[0] - 1):
            res.col_join(self.represent[0] ** i * self.represent[1])
        return res

    def controllable_subspace(self):

        """ Returns a list of vectors that span the controllable subspace of the system.

        This subspace consists of the states x0 for which there exists an input u : [t0, t1] -> R^k, that
        transfers the state x(t0) = x0 to x(t1) = 0.

        The controllable subspace of an lti system is equal to the image of its controllability sp.Matrix.
        """

        return self.controllability_matrix().columnspace()

    def is_controllable(self):

        """ Returns True, if the system is controllable.

        A lti system is called 'controllable' if the controllable subspace of the system equals the
        whole state space R^n. This means, that every state x0 can be transfered to zero at any time.

        The package implements the Eigenvector test for controllability
        """

        for eigenvect_of_A_tr in self.represent[0].transpose().eigenvects():
            for idx in xrange(eigenvect_of_A_tr[1]):
                if (self.represent[1] * eigenvect_of_A_tr[2][idx]).is_zero:
                    return False
        return True

    def cascade(self, anotherSystem):

        """ Returns the cascade interconnection of the system and another system

        The casade interconnection of two systems P1 and P2 is the system for which
        u = u1, y = y2 and z = u2 = y1 so that:

               ----    z	 ----
        u --> | P1 | -----> | P2 | --> y
               ----          ----

        Parameters
        ==========

        anotherSystem : StateSpace
            StateSpace representation of the model you want to interconnect with
            the current model

        See Also
        ========

        parallel: parallel interconnection of two systems
        """

        if not isinstance(anotherSystem, StateSpace):
            raise TypeError("Argument must be of type StateSpace")

        # assert matching shapes:

        if not self.represent[2].shape[0] == anotherSystem.represent[1].shape[1]:
            raise sp.ShapeError("Dimensions of the input of the argument and the ouput of the System must match!")

        newA = self.represent[0].row_join(
            sp.zeros(self.represent[0].rows, anotherSystem.represent[0].cols)
        ).col_join(
            (anotherSystem.represent[1] * self.represent[2]).row_join(anotherSystem.represent[0])
        )

        newB = self.represent[1].col_join(anotherSystem.represent[1] * self.represent[3])
        newC = (anotherSystem.represent[3] * self.represent[2]).row_join(anotherSystem.represent[2])
        newD = anotherSystem.represent[3] * self.represent[3]

        return StateSpace([newA, newB, newC, newD])

    def parallel(self, anotherSystem):

        """ Returns the parallel interconnection of the system and another system

        The parallel interconnection of two systems P1 and P2 is the system for which
        u = u1 + u2 and y = y1 + y2 so that:

                  ----  y1
             --> | P1 |---
            |     ----    |+
        u --|             o ---> y
            |     ----    |+
             --> | P2 |---
                  ----  y2

        Parameters
        ==========

        anotherSystem : StateSpace
            StateSpace representation of the model you want to interconnect with
            the current model

        See Also
        ========

        cascade: cascade interconnection of two systems
        """

        if not isinstance(anotherSystem, StateSpace):
            raise TypeError("Argument must be of type StateSpace, not %r" % (type(anotherSystem)))
        
        # assert matching shapes:

        if not ((self.represent[1].shape[1] == anotherSystem.represent[1].shape[1]) and
                (self.represent[2].shape[0] == anotherSystem.represent[2].shape[0])):
            raise sp.ShapeError("Dimensions of inputs and outputs must match!")

        newA = self.represent[0].col_join(sp.zeros(self.represent[0].rows, anotherSystem.represent[0].cols)) \
                                .row_join(
                                    sp.zeros(anotherSystem.represent[0].rows, self.represent[0].cols)
                                    .col_join(anotherSystem.represent[0]))
        newB = self.represent[1].col_join(anotherSystem.represent[1])
        newC = self.represent[2].row_join(anotherSystem.represent[2])
        newD = self.represent[3] + anotherSystem.represent[3]

        return StateSpace([newA, newB, newC, newD])

    def __getattr__(self, name):

        # dont overwrite private or magic sp.Function attribute testing!
        if name[0] == '_':
            raise AttributeError("%r object has no attribute %r" %
                                 (self.__class__, name))

        try:
            def handler(*args, **kwargs):

                new_represent = []
                for r in self.represent:
                    methodToCall = getattr(r, name)
                    new_represent.append(methodToCall(*args, **kwargs))
                return StateSpace(new_represent)

        except AttributeError:
            raise AttributeError("%r object has no attribute %r" %
                                 (self.__class__, name))

        return handler

    def latex(self):

        return '$' + sp.latex(self.BlockRepresent) + '$'

    def pretty(self):

        a = sp.pretty(self.represent[0])
        b = sp.pretty(self.represent[1])
        c = sp.pretty(self.represent[2])
        d = sp.pretty(self.represent[3])

        return "A=\n\n{0}\n\nB=\n\n{1}\n\nC=\n\n{2}\n\nD=\n\n{3}\n\n".format(a, b, c, d)

    def pprint(self):

        print(self.pretty())

    def __repr__(self):

        return self.pretty()

    def __str__(self):

        return self.pretty()


class TransferFunction(object):

    """ Transfer sp.Function model of a linear, time invariant crontrol system

    Represents the transfere sp.Function model with a transfer sp.Function sp.Matrix G in laplace space.
    The input-output relation for the system in laplace space is then given by:
        y(s) = G(s) * u(s);     s in C
    where u(s) is the input of the system in laplace space and y(s) the corresponding output

    Parameters
    ==========

    arg : StateSpace, sp.Matrix
        the state space model to contruct the transfer sp.Function model from, or the transfer sp.Matrix G
    s = None : sp.Symbol
        the variable G is dependent from. only has to be set if arg is a non-constant sp.Matrix or StateSpace

    See Also
    ========

    TranferFunctionModel: transfer sp.Function model of a lti system
    Utils: mixed sp.Matrix and polynomial tools

    References
    ==========

    Joao P. Hespanha, Linear Systems Theory. 2009.
    """

    def __init__(self, arg, s=None):

        # check if a variable is given, if not create a new one as class-wide variable
        if s:
            self.s = s
        else:
            self.s = sp.var('s')

        # constructor from a given state space model

        if isinstance(arg, StateSpace):

            try:
                # define G as transfer sp.Function for the given state space model via the definition
                self.G = arg.represent[2] * \
                    (self.s * sp.eye(arg.represent[0].shape[0]) - arg.represent[0]).inv() * \
                    arg.represent[1] + arg.represent[3]

                # try to sp.simplify
                self.G = sp.simplify(self.G)

            except ValueError as err:
                raise ValueError(err.args, "sp.Matrix (s*I -A) must be invertible")
            
            except AttributeError:
                raise TypeError("Only explicit sp.Matrix Type supported for A,B,C,D (.inv() must work)")

        elif isinstance(arg, (sp.Matrix, sp.ImmutableMatrix, sp.MutableMatrix)):

            # set the given transfer sp.Function as self.G:

            self.G = sp.simplify(arg)

        elif isinstance(arg, sp.Basic):

            # handle the SISO case of just one s sp.Function:
            try:
                self.G = sp.simplify(sp.Matrix([arg]))
            except:
                raise TypeError("argument of unsupported type: {}".format(type(arg)))

        else:
            raise TypeError("argument of unsupported type: {}".format(type(arg)))

    def latex(self):

        return '$' + sp.latex(self.G) + '$'

    def evaluate(self, u, s):

        """ evaluate the result for input u

        The input u in laplace state depends on a complex variable s the result y is computed by
            y(s) = G(s) * u(s)

        Parameters
        ==========

        u : one-column sp.Matrix
            the input vector u in terms of complex variable s
        s : sp.Symbol
            the complex variable s u is dependent from.
        """

        # assert right shape of u:

        if not u.shape[1] == 1:
            raise sp.ShapeError("u must be a column vector, not a sp.Matrix")

        if not self.G.shape[1] == u.shape[0]:
            raise sp.ShapeError("u must have a length of ", self.G.shape[1])

        # return result:
        return self.G.subs(self.s, s) * u

    def pretty(self):

        g = sp.pretty(self.G)

        return "G(s)=\n\n{0}\n\n".format(g)

    def pprint(self):

        print(self.pretty())

    def __repr__(self):

        return self.pretty()

    def __str__(self):

        return self.pretty()

    def __getattr__(self, name):

        # dont overwrite private or magic sp.Function attribute testing!
        if name[0] == '_':
            raise AttributeError("%r object has no attribute %r" %
                                 (self.__class__, name))

        try:
            def handler(*args, **kwargs):
                methodToCall = getattr(self.G, name)
                return TransferFunction(methodToCall(*args, **kwargs))

        except AttributeError:
            raise AttributeError("%r object has no attribute %r" %
                                 (self.__class__, name))
        return handler
