from sympy import symbols, lambdify, diff
import matplotlib.pyplot as plt
import numpy as np

#     GRAFICAR LA FUNCIÓN Y LAS ITERACIONES (Newton-Raphson)
def graph_newton_raphson(f, result, x_range=None):
    """
    Grafica la función f(x) y los puntos generados por Newton-Raphson.

    Parámetros:
      f (sympy expression): función a graficar
      result (list): valores de x en cada iteración
      x_range (tuple): rango opcional (xmin, xmax)
    """
    # Variable simbólica
    x = symbols('x')

    # Convertir f(x) simbólica → función de Python
    f_py = lambdify(x, f)

    # Determinar rango automáticamente si no se da
    if x_range is None:
        x_min = min(result) - 1
        x_max = max(result) + 1
    else:
        x_min, x_max = x_range

    # Valores para graficar
    x_values = np.linspace(x_min, x_max, 500)
    y_values = f_py(x_values)

    # Crear figura
    plt.figure()
    plt.title("Raíces de una función - Método de Newton-Raphson")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    # Graficar la función
    plt.plot(x_values, y_values, label="f(x)")

    # Marcar los puntos de iteración
    for i, x_i in enumerate(result):
        plt.plot(x_i, f_py(x_i), 'ro')   # puntos rojos

    plt.legend()
    plt.grid(True)
    plt.show()



#     MÉTODO DE NEWTON - RAPHSON
def newton_raphson(f, x0, tol=1e-10):
    """
    Calcula la raíz de una función usando el método Newton-Raphson.

    Parámetros:
      f (sympy expression): función simbólica
      x0 (float): valor inicial
      tol (float): tolerancia del error

    Retorna:
      list: valores de x en cada iteración
    """

    x = symbols('x')

    # Derivada simbólica
    f_prime = diff(f, x)

    # Convertir f y f' a funciones Python
    f_py = lambdify(x, f)
    f_prime_py = lambdify(x, f_prime)

    # Comenzar iterando
    x_current = x0
    result = [x_current]

    # Bucle iterativo
    while abs(f_py(x_current)) > tol:
        x_current = x_current - f_py(x_current) / f_prime_py(x_current)
        result.append(x_current)

    return result