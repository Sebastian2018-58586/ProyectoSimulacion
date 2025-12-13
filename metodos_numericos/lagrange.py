# metodos_numericos/lagrange.py
import sympy as sp

def lagrange_interpolation(points):
    """
    Construye el polinomio interpolante de Lagrange

    Parámetros:
      points: lista de tuplas [(x0,y0), (x1,y1), ..., (xn,yn)]

    Retorna:
      Polinomio simbólico simplificado
    """

    x = sp.symbols('x')
    P = 0
    n = len(points)

    for i in range(n):
        xi, yi = points[i]
        Li = 1

        for j in range(n):
            if i != j:
                xj, _ = points[j]
                Li *= (x - xj) / (xi - xj)

        P += yi * Li

    return sp.simplify(P)


def input_points():
    """
    Pide al usuario el grado y los puntos
    """

    degree = int(input("Ingrese el grado del polinomio: "))
    points = []

    print(f"\nIngrese {degree + 1} puntos:")

    for i in range(degree + 1):
        x_i = float(input(f"x{i}: "))
        y_i = float(input(f"y{i}: "))
        points.append((x_i, y_i))

    return points
