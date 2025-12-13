# metodos_numericos/biseccion.py
import sympy as sp

def bisection_method(f, a, b, tol=1e-4, max_iter=100):
    """
    Método de bisección con tabla de iteraciones

    Retorna una lista de diccionarios con:
    iter, an, bn, pn, fpn, error
    """

    x = sp.symbols('x')
    f_sym = sp.sympify(f)
    f_func = sp.lambdify(x, f_sym, 'math')

    if f_func(a) * f_func(b) >= 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos")

    table = []
    n = 0

    while n < max_iter:
        p = (a + b) / 2
        fp = f_func(p)
        error = abs(b - a) / 2

        table.append({
            "iter": n,
            "a": a,
            "b": b,
            "p": p,
            "fp": fp,
            "error": error
        })

        if error < tol:
            break

        if f_func(a) * fp < 0:
            b = p
        else:
            a = p

        n += 1

    return table


def print_bisection_table(table):
    print("\nMétodo de Bisección")
    print("-" * 85)
    print(f"{'n':<5}{'an':<15}{'bn':<15}{'pn':<15}{'f(pn)':<15}{'Error':<15}")
    print("-" * 85)

    for row in table:
        print(f"{row['iter']:<5}"
              f"{row['a']:<15.6f}"
              f"{row['b']:<15.6f}"
              f"{row['p']:<15.6f}"
              f"{row['fp']:<15.6f}"
              f"{row['error']:<15.6e}")
