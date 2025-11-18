from sympy import symbols, lambdify, diff
import matplotlib.pyplot as plt
import numpy as np
import math

# graficar la función y el polinomio de taylor
def graph_taylor(f, t, x0, x_range=None):
  """
  Graficar la función y el polinomio de Taylor.

  Parámetros:
    f: función a aproximar.
    t: resultado de la aproximación.
    x0: punto de aproximación.
    x_range: tupla (xmin, xmax) para especificar el rango de x. Si es None, se usa un rango por defecto.
  """
  # Definir el rango de x alrededor de x0
  if x_range is None:
    x_range = (x0 - 5, x0 + 5)  # Rango por defecto si no se especifica
  
  x_vals = np.linspace(x_range[0], x_range[1], 500)
  
  # Convertir las funciones simbólicas a funciones numéricas
  x = symbols('x')
  f_numeric = lambdify(x, f, modules=['numpy'])
  t_numeric = lambdify(x, t, modules=['numpy'])
  
  # Evaluar la función original y el polinomio de Taylor
  try:
    f_vals = f_numeric(x_vals)
    t_vals = t_numeric(x_vals)
  except TypeError as e:
    print("Error al evaluar las funciones:", e)
    return
  
  # Crear la gráfica
  plt.figure()
  plt.plot(x_vals, f_vals, label='Función original $f(x)$', color='blue')
  plt.plot(x_vals, t_vals, label=f'Polinomio de Taylor en $x_0 = {x0}$', color='red', linestyle='--')
  
  # Destacar el punto de aproximación
  plt.scatter([x0], [f_numeric(x0)], color='black', label=f'Punto de aproximación $x_0={x0}$', zorder=5)
  
  # Etiquetas y leyenda
  plt.title('Aproximación con el Polinomio de Taylor')
  plt.xlabel('$x$')
  plt.ylabel('$y$')
  plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
  plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
  plt.grid(alpha=0.3)
  plt.legend()
  plt.show()


# polinomio de taylor
def taylor(f, x0, n):
  """
  Polinomio de Taylor de f en x0 de grado n

  Parámetros:
    f: función a aproximar
    x0: punto de aproximación
    n: grado del polinomio

  Retorna:
    t: polinomio de Taylor de f en x de grado n
  """

  x = symbols('x')
  t = 0

  for i in range(n):
    t += diff(f, x, i).subs(x, x0) / math.factorial(i) * (x - x0)**i

  return t


# función para calcular el error real
def absolute_error(f, t, x):
  """
  Calcular el error real de la aproximación con el polinomio de Taylor

  Parámetros:
    f: función original
    t: polinomio de Taylor
    x: punto en el que se evalúa el error

  Retorna:
    error: error real de la aproximación
  """
  
  x_sym = symbols('x')
  error = f.subs(x_sym, x) - t.subs(x_sym, x)
  
  return abs(error)

# función para calcular el error relativo
def relative_error(f, t, x):
  """
  Calcular el error relativo de la aproximación con el polinomio de Taylor

  Parámetros:
    f: función original
    t: polinomio de Taylor
    x: punto en el que se evalúa el error

  Retorna:
    error: error relativo de la aproximación
  """
  
  x_sym = symbols('x')
  error = (f.subs(x_sym, x) - t.subs(x_sym, x)) / f.subs(x_sym, x)
  
  return abs(error)