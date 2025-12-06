from sympy import symbols, lambdify, diff
import matplotlib.pyplot as plt
import numpy as np
import math


# ============================================================
#   POLINOMIO DE TAYLOR
# ============================================================

def taylor(f, x0, n):
    """
    Calcula el polinomio de Taylor de f en x0 de grado n.

    Parámetros:
        f  : función simbólica de SymPy
        x0 : punto de aproximación
        n  : grado del polinomio

    Retorna:
        t : polinomio de Taylor de grado n
    """
    x = symbols('x')
    t = 0

    # CORRECCIÓN: incluir el término i = n
    for i in range(n + 1):
        term = diff(f, x, i).subs(x, x0) / math.factorial(i) * (x - x0)**i
        t += term

    return t


# ============================================================
#   ERROR ABSOLUTO
# ============================================================

def absolute_error(f, t, x):
    """
    Calcula el error absoluto |f(x) - T(x)|

    Parámetros:
        f : función original simbólica
        t : polinomio de Taylor simbólico
        x : punto donde se evalúa el error

    Retorna:
        valor del error absoluto
    """
    x_sym = symbols('x')
    return abs(f.subs(x_sym, x) - t.subs(x_sym, x))


# ============================================================
#   ERROR RELATIVO
# ============================================================

def relative_error(f, t, x):
    """
    Calcula el error relativo |(f(x) - T(x)) / f(x)|

    Parámetros:
        f : función original simbólica
        t : polinomio de Taylor simbólico
        x : punto donde se evalúa el error

    Retorna:
        valor del error relativo
    """
    x_sym = symbols('x')
    real_value = f.subs(x_sym, x)

    if real_value == 0:
        return float('inf')  # evita división por 0

    return abs((real_value - t.subs(x_sym, x)) / real_value)


# ============================================================
#   GRÁFICA FUNCIÓN vs TAYLOR
# ============================================================

def graph_taylor(f, t, x0, x_range=None):
    """
    Gráfica la función original f y el polinomio de Taylor t.

    Parámetros:
        f       : función simbólica
        t       : polinomio de Taylor simbólico
        x0      : punto de aproximación
        x_range : tupla (xmin, xmax) opcional
    """

    x = symbols('x')

    # rango automático
    if x_range is None:
        x_range = (x0 - 5, x0 + 5)

    x_vals = np.linspace(x_range[0], x_range[1], 500)

    # convertir a funciones numéricas
    f_numeric = lambdify(x, f, 'numpy')
    t_numeric = lambdify(x, t, 'numpy')

    # evaluar
    f_vals = f_numeric(x_vals)
    t_vals = t_numeric(x_vals)

    # graficar
    plt.figure()
    plt.plot(x_vals, f_vals, label="Función original f(x)", linewidth=2)
    plt.plot(x_vals, t_vals, "--", label=f"Polinomio de Taylor (x0={x0})", linewidth=2)

    # punto de expansión
    plt.scatter([x0], [f_numeric(x0)], color='black', label="Punto x0")

    # estilo
    plt.title("Aproximación con el Polinomio de Taylor")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
