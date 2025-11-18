import numpy as np

def jacobi_method(A, b, tol=1e-4, max_iter=100):
    """
    Método de Jacobi para resolver sistemas lineales de la forma Ax = b.

    Este método calcula aproximaciones sucesivas del vector solución x,
    usando únicamente valores de la iteración anterior (característica
    principal del método de Jacobi).

    Parámetros:
        A (matrix o lista de listas):
            Matriz de coeficientes del sistema.

        b (vector o lista):
            Vector de términos independientes.

        tol (float):
            Tolerancia para el criterio de parada. El método se detiene cuando
            la diferencia entre dos aproximaciones consecutivas es menor a tol.

        max_iter (int):
            Número máximo de iteraciones permitidas.

    Retorna:
        list:
            Lista que contiene todas las aproximaciones generadas en cada
            iteración, incluidas la inicial (vector cero) y la aproximación final.
    """

    # Convertir A y b a arreglos NumPy para operaciones numéricas eficientes
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    n = len(b)

    # Aproximación inicial: vector de ceros
    x = np.zeros(n, dtype=float)
    results = [list(x)]  # Guardar la primera aproximación

    # Iteraciones del método
    for k in range(max_iter):
        x_new = np.copy(x)

        # Cálculo de cada componente de x^(k+1)
        for i in range(n):
            sum_ = 0
            for j in range(n):
                if j != i:
                    sum_ += A_np[i][j] * x[j]

            # Fórmula del método de Jacobi
            x_new[i] = (b_np[i] - sum_) / A_np[i][i]

        # Guardar la aproximación de esta iteración
        results.append(list(x_new))

        # Verificar convergencia usando norma infinito
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return results

        # Actualizamos x para la siguiente iteración
        x = np.copy(x_new)

    # Si se alcanza el máximo de iteraciones sin converger
    print("⚠ El método no convergió en el número máximo de iteraciones.")
    return results
