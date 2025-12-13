import sympy as sp
from metodos_numericos.polinomio_taylor import taylor, absolute_error, relative_error, graph_taylor
from metodos_numericos.newton_raphson import newton_raphson, graph_newton_raphson
from metodos_numericos.ecuaciones_lineales import jacobi_method
from metodos_numericos.ecuaciones_no_lineales import newton_raphson_n_variables, graph_nonlinear_equations
from metodos_numericos.diferencias_finitas import solve_finite_differences, graph_finite_differences
from metodos_numericos.biseccion import bisection_method, print_bisection_table
from metodos_numericos.lagrange import lagrange_interpolation, input_points



# funcion principal
def main():
  while True:
    print("1. Método polinomio de taylor")
    print("2: Método newton-raphson")
    print("3. Método diferencias finitas")
    print("4. Método sistemas de ecuaciones no lineales (jacobiano)")
    print("5. Método sistema de ecuaciones lineales (jacobi)")
    print("6. Método de bisección")
    print("7. Interpolación de Lagrange")
    print("8. Salir")

    option = input("Seleccione una opción: ")

    if option == "1":
      # datos de entrada
      f = input("Ingrese la función f(x): ")
      x0 = float(input("Ingrese el punto de aproximación x0: "))
      n = int(input("Ingrese el grado del polinomio de Taylor: "))
      x = float(input("Ingrese el punto en el que se evalúa el error x: "))
      x_range = int(input("Ingrese el rango de x: "))

      # rango de x
      x_range = (x0 - x_range, x0 + x_range)

      # calcular el polinomio de Taylor
      f = sp.sympify(f)
      t = taylor(f, x0, n)
      print(t)

      # imprimir el error absoluto y relativo
      print(f'Error absoluto: {absolute_error(f, t, x)}')
      print(f'Error relativo: {relative_error(f, t, x)}')

      # graficar la función y el polinomio de Taylor
      graph_taylor(f, t, x0, x_range)

    elif option == "2":
         # datos de entrada
      f = input("Ingrese la función f(x): ")
      x0 = float(input("Ingrese el punto de aproximación x0: "))
      tol = float(input("Ingrese la tolerancia: "))
      x_range = int(input("Ingrese el rango de x: "))

      # rango de x
      x_range = (x0 - x_range, x0 + x_range)

      # calcular las raíces de la función
      result = newton_raphson(f, x0, tol)

      # imprimir los resultados
      for i, x in enumerate(result):
        print(f'Raíz {i + 1}: {x}')

      # graficar la función y las raíces
      graph_newton_raphson(f, result, x_range)

    elif option == "3":
       # datos de entrada
      x = sp.symbols('x')
      px_input = input("Ingrese p(x): ")
      qx_input = input("Ingrese q(x): ")
      rx_input = input("Ingrese r(x): ")

      px = sp.lambdify(x, sp.sympify(px_input), 'numpy')
      qx = sp.lambdify(x, sp.sympify(qx_input), 'numpy')
      rx = sp.lambdify(x, sp.sympify(rx_input), 'numpy')

      a = float(input("Ingrese el límite inferior del intervalo: "))
      b = float(input("Ingrese el límite superior del intervalo: "))
      y_a = float(input("Ingrese la condición inicial y(a): "))
      y_b = float(input("Ingrese la condición inicial y(b): "))
      n = int(input("Ingrese el número de puntos: "))

      coefficients = {'px': px, 'qx': qx, 'rx': rx}
      points = [(a, y_a), (b, y_b)]

      # resolver la ecuación diferencial
      x_vals, y_vals = solve_finite_differences(coefficients, points, n)

      # imprimir los resultados
      for x, y in zip(x_vals, y_vals):
        print(f'y({x}) = {y}')

      # graficar la solución
      graph_finite_differences((x_vals, y_vals))

    elif option == "4":
      # datos de entrada
      variables = int(input("Ingrese el número de variables: "))
      f = []

      for i in range(variables):
        f.append(input(f"Ingrese la función f{(i+1)}: "))
      
      x0 = []

      for i in range(variables):
        x0.append(float(input(f"Ingrese el valor inicial {(i+1)}: ")))
      
      n = int(input("Ingrese el número de iteraciones: "))

      # calcular las raíces del sistema de ecuaciones no lineales
      result = newton_raphson_n_variables(f, x0, variables, n)

      # imprimir los resultados
      formatted_result = [[float(x) for x in y] for y in result]

      for i, x in enumerate(formatted_result):
        print(f'x{i} = {x}')

      if variables == 2:
        # graficar las funciones y los puntos
        graph_nonlinear_equations(f[0], f[1], result)

    elif option == "5":
       # datos de entrada
      A = input("Ingrese la matriz A: ")
      A = sp.Matrix(eval(A))
      b = input("Ingrese el vector b: ")
      b = sp.Matrix(eval(b))
      tol = float(input("Ingrese la tolerancia: "))
      n = int(input("Ingrese el número de iteraciones: "))

      # calcular las raíces del sistema de ecuaciones lineales
      result = jacobi_method(A, b, tol, n)
      
      # imprimir los resultados
      formatted_result = [[float(x) for x in y] for y in result]

      for i, x in enumerate(formatted_result):
        print(f'Iteración {i}: {x}')
        
    elif option == "6":
      f = input("Ingrese la función f(x): ")
      a = float(input("Ingrese el extremo izquierdo a: "))
      b = float(input("Ingrese el extremo derecho b: "))
      tol = float(input("Ingrese la tolerancia: "))

      try:
          table = bisection_method(f, a, b, tol)
          print_bisection_table(table)
          print(f"\nAproximación final: x ≈ {table[-1]['p']}")
      except ValueError as e:
          print("Error:", e)
          
    elif option == "7":
      points = input_points()
      P = lagrange_interpolation(points)
      print("\nPolinomio interpolante de Lagrange:")
      print(P)

    elif option == "8":
      break

    else:
      print("Opción inválida")
  
if __name__ == "__main__":
  # llamar las funciones
  main()