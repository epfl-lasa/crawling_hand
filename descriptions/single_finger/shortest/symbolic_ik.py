from sympy import symbols, Eq, sin, cos, solve

# Define symbols
qt0, qt1, qt2, x, y, z = symbols('qt0 qt1 qt2 x y z')

# Define the equations as given in the image
eq1 = Eq(x, -0.0116*sin(qt0)*sin(qt1) + 0.0235*sin(qt0)*cos(qt1) + 0.039*sin(qt0)*cos(qt1 + qt2) 
          + 0.026*sin(qt0)*cos(qt1 + 2*qt2) - 0.0029*cos(qt0))

eq2 = Eq(y, 0.0029*sin(qt0) - 0.0116*sin(qt1)*cos(qt0) + 0.0235*cos(qt0)*cos(qt1) + 0.039*cos(qt0)*cos(qt1 + qt2) 
          + 0.026*cos(qt0)*cos(qt1 + 2*qt2) + 0.0725)

eq3 = Eq(z, -0.0235*sin(qt1) - 0.039*sin(qt1 + qt2) - 0.026*sin(qt1 + 2*qt2) - 0.0116*cos(qt1) + 0.0256)

# Solve the system of equations for qt0, qt1, and qt2
solutions = solve((eq1, eq2, eq3), (qt0, qt1, qt2), dict=True, rational=True)
print(solutions)