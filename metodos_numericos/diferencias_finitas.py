import numpy as np
import matplotlib.pyplot as plt


def graph_finite_differences(result):
    """
    Grafica la solución aproximada obtenida mediante el método de diferencias finitas.

    Parámetros:
        result (tuple): Tupla (x, y), donde:
                        - x: Arreglo con los puntos discretizados del dominio.
                        - y: Arreglo con los valores aproximados de la solución.
    """
    x, y = result

    plt.plot(x, y, 'o-', label='Solución aproximada')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solución de la ecuación diferencial por diferencias finitas')
    plt.grid(True)
    plt.legend()
    plt.show()



def solve_finite_differences(coefficients, points, n):
    """
    Resuelve una ecuación diferencial de segundo orden de la forma:

        y'' + p(x)·y' + q(x)·y = r(x)

    utilizando el método de Diferencias Finitas en un intervalo con condiciones de frontera.

    Parámetros:
        coefficients (dict): Diccionario con funciones:
            - 'px': Función p(x)  → coeficiente del término y'
            - 'qx': Función q(x)  → coeficiente del término y
            - 'rx': Función r(x)  → término independiente
        
        points (list of tuples): Parejas [(x1, y1), (x2, y2)], donde:
            - x1, y1 → condición de frontera en el extremo izquierdo
            - x2, y2 → condición de frontera en el extremo derecho
        
        n (int): Número de puntos interiores en la discretización.

    Retorna:
        tuple: (x, y)
            - x → arreglo con los puntos de discretización
            - y → solución aproximada en cada punto
    """

    # Extraer funciones coeficientes
    px = coefficients['px']
    qx = coefficients['qx']
    rx = coefficients['rx']

    # Condiciones de frontera
    (x1, y1), (x2, y2) = points

    # Tamaño de paso
    h = (x2 - x1) / (n + 1)

    # Discretización del dominio
    x = np.linspace(x1, x2, n + 2)

    # Matriz del sistema lineal
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Construcción de la matriz A y el vector b
    for i in range(1, n + 1):
        xi = x1 + i * h

        p = px(xi)
        q = qx(xi)
        r = rx(xi)

        # Subdiagonal
        if i > 1:
            A[i - 1, i - 2] = 1 + (h / 2) * p

        # Diagonal principal
        A[i - 1, i - 1] = -2 - h**2 * q

        # Superdiagonal
        if i < n:
            A[i - 1, i] = 1 - (h / 2) * p

        # Vector b
        b[i - 1] = r * h**2 * x[i]

    # Ajuste por condiciones de frontera
    b[0] = r * h**2 * x[1] - (1 + h / 2 * p) * y1
    b[-1] = r * h**2 * x[-2] - (1 - h / 2 * p) * y2

    # Resolver el sistema lineal
    y_inner = np.linalg.solve(A, b)

    # Unir solución interior con las condiciones de frontera
    y = np.concatenate(([y1], y_inner, [y2]))

    return x, y
