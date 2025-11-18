import tkinter as tk
from tkinter import ttk, messagebox
import sympy as sp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np

# Importar métodos numéricos
from metodos_numericos.polinomio_taylor import taylor, absolute_error, relative_error
from metodos_numericos.newton_raphson import newton_raphson
from metodos_numericos.diferencias_finitas import solve_finite_differences
from metodos_numericos.ecuaciones_no_lineales import newton_raphson_n_variables, graph_nonlinear_equations
from metodos_numericos.ecuaciones_lineales import jacobi_method

class NumericalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulación Métodos Numéricos")
        self.geometry("1000x700")

        # Crear tabs
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(expand=1, fill="both")

        # Crear pestañas de cada método
        self.create_taylor_tab()
        self.create_newton_tab()
        self.create_finite_diff_tab()
        self.create_nonlinear_tab()
        self.create_jacobi_tab()

    # ------------------------- TAB TAYLOR -------------------------
    def create_taylor_tab(self):
        self.taylor_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.taylor_tab, text="Taylor")

        frame_inputs = ttk.LabelFrame(self.taylor_tab, text="Datos de Entrada")
        frame_inputs.pack(fill="x", padx=10, pady=10)

        labels = ["f(x):", "x0:", "Grado n:", "Punto x:", "Rango x:"]
        self.taylor_entries = []
        for i, text in enumerate(labels):
            ttk.Label(frame_inputs, text=text).grid(row=i, column=0, sticky="w", padx=5, pady=5)
            entry = ttk.Entry(frame_inputs, width=30)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.taylor_entries.append(entry)

        ttk.Button(frame_inputs, text="Calcular y Graficar", command=self.calculate_taylor).grid(row=5, column=0, columnspan=2, pady=10)

        self.taylor_result = tk.Text(self.taylor_tab, height=10)
        self.taylor_result.pack(fill="both", padx=10, pady=10)

    def calculate_taylor(self):
        try:
            self.taylor_result.delete("1.0", tk.END)

            f = sp.sympify(self.taylor_entries[0].get())
            x0 = float(self.taylor_entries[1].get())
            n = int(self.taylor_entries[2].get())
            x_val = float(self.taylor_entries[3].get())
            x_range_val = int(self.taylor_entries[4].get())
            x_range = (x0 - x_range_val, x0 + x_range_val)

            t_poly = taylor(f, x0, n)
            err_abs = absolute_error(f, t_poly, x_val)
            err_rel = relative_error(f, t_poly, x_val)

            self.taylor_result.insert(tk.END, f"Polinomio de Taylor: {t_poly}\n")
            self.taylor_result.insert(tk.END, f"Error absoluto: {err_abs}\n")
            self.taylor_result.insert(tk.END, f"Error relativo: {err_rel}\n")

            # Nueva ventana para la gráfica
            win = tk.Toplevel(self)
            win.title("Gráfica Taylor")
            win.geometry("800x600")
            fig, ax = plt.subplots()

            x_sym = sp.symbols("x")
            f_lamb = sp.lambdify(x_sym, f, "numpy")
            t_lamb = sp.lambdify(x_sym, t_poly, "numpy")
            xs = np.linspace(x_range[0], x_range[1], 500)

            ax.plot(xs, f_lamb(xs), label="f(x) original", color="blue", linewidth=2)
            ax.plot(xs, t_lamb(xs), label=f"Taylor grado {n}", color="orange", linestyle="--", linewidth=2)
            ax.scatter([x_val], [f_lamb(x_val)], color="green", s=100, label=f"Punto evaluación x={x_val}")
            ax.annotate(f"f(x)={f_lamb(x_val):.4f}\nT(x)={t_lamb(x_val):.4f}\nError abs={err_abs:.4f}",
                        xy=(x_val, f_lamb(x_val)), xytext=(x_val + 0.5, f_lamb(x_val)),
                        arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
            ax.set_title("Polinomio de Taylor vs Función Original", fontsize=14, fontweight='bold')
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("y", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            toolbar = NavigationToolbar2Tk(canvas, win)
            toolbar.update()
            canvas._tkcanvas.pack(fill="both", expand=True)

            for entry in self.taylor_entries:
                entry.delete(0, tk.END)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------------------- TAB NEWTON -------------------------
    def create_newton_tab(self):
        self.newton_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.newton_tab, text="Newton-Raphson")

        frame_inputs = ttk.LabelFrame(self.newton_tab, text="Datos de Entrada")
        frame_inputs.pack(fill="x", padx=10, pady=10)

        labels = ["f(x):", "x0:", "Tolerancia:", "Rango x:"]
        self.newton_entries = []
        for i, text in enumerate(labels):
            ttk.Label(frame_inputs, text=text).grid(row=i, column=0, sticky="w", padx=5, pady=5)
            entry = ttk.Entry(frame_inputs, width=30)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.newton_entries.append(entry)

        ttk.Button(frame_inputs, text="Calcular y Graficar", command=self.calculate_newton).grid(row=4, column=0, columnspan=2, pady=10)

        self.newton_result = tk.Text(self.newton_tab, height=10)
        self.newton_result.pack(fill="both", padx=10, pady=10)

    def calculate_newton(self):
        try:
            self.newton_result.delete("1.0", tk.END)

            f = self.newton_entries[0].get()
            x0 = float(self.newton_entries[1].get())
            tol = float(self.newton_entries[2].get())
            x_range_val = int(self.newton_entries[3].get())
            x_range = (x0 - x_range_val, x0 + x_range_val)

            roots = newton_raphson(f, x0, tol)
            for i, r in enumerate(roots):
                self.newton_result.insert(tk.END, f"Raíz {i+1}: {r}\n")

            win = tk.Toplevel(self)
            win.title("Gráfica Newton-Raphson")
            win.geometry("800x600")
            fig, ax = plt.subplots()

            x_sym = sp.symbols("x")
            f_lamb = sp.lambdify(x_sym, sp.sympify(f), "numpy")
            xs = np.linspace(x_range[0], x_range[1], 500)
            ax.plot(xs, f_lamb(xs), label="f(x)", color="blue", linewidth=2)
            ax.scatter(roots, [0]*len(roots), color="red", s=80, label="Raíces")
            for r in roots:
                ax.annotate(f"{r:.4f}", xy=(r, 0), xytext=(r, max(f_lamb(xs))*0.05),
                            arrowprops=dict(facecolor='black', shrink=0.05))
            ax.set_title("Newton-Raphson", fontsize=14, fontweight='bold')
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            toolbar = NavigationToolbar2Tk(canvas, win)
            toolbar.update()
            canvas._tkcanvas.pack(fill="both", expand=True)

            for entry in self.newton_entries:
                entry.delete(0, tk.END)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------------------- TAB DIFERENCIAS FINITAS -------------------------
    def create_finite_diff_tab(self):
        self.fd_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.fd_tab, text="Diferencias Finitas")

        frame_inputs = ttk.LabelFrame(self.fd_tab, text="Datos de Entrada")
        frame_inputs.pack(fill="x", padx=10, pady=10)

        labels = ["p(x):", "q(x):", "r(x):", "a:", "b:", "y(a):", "y(b):", "Número de puntos:"]
        self.fd_entries = []
        for i, text in enumerate(labels):
            ttk.Label(frame_inputs, text=text).grid(row=i, column=0, sticky="w", padx=5, pady=5)
            entry = ttk.Entry(frame_inputs, width=30)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.fd_entries.append(entry)

        ttk.Button(frame_inputs, text="Calcular y Graficar", command=self.calculate_fd).grid(row=len(labels), column=0, columnspan=2, pady=10)

        self.fd_result = tk.Text(self.fd_tab, height=10)
        self.fd_result.pack(fill="both", padx=10, pady=10)

    def calculate_fd(self):
        try:
            self.fd_result.delete("1.0", tk.END)

            x = sp.symbols("x")
            px = sp.lambdify(x, sp.sympify(self.fd_entries[0].get()), 'numpy')
            qx = sp.lambdify(x, sp.sympify(self.fd_entries[1].get()), 'numpy')
            rx = sp.lambdify(x, sp.sympify(self.fd_entries[2].get()), 'numpy')

            a = float(self.fd_entries[3].get())
            b = float(self.fd_entries[4].get())
            y_a = float(self.fd_entries[5].get())
            y_b = float(self.fd_entries[6].get())
            n = int(self.fd_entries[7].get())

            coefficients = {'px': px, 'qx': qx, 'rx': rx}
            points = [(a, y_a), (b, y_b)]

            x_vals, y_vals = solve_finite_differences(coefficients, points, n)
            for xv, yv in zip(x_vals, y_vals):
                self.fd_result.insert(tk.END, f"y({xv}) = {yv}\n")

            win = tk.Toplevel(self)
            win.title("Gráfica Diferencias Finitas")
            win.geometry("800x600")
            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals, label="Solución", color="purple", linewidth=2)
            ax.scatter([a, b], [y_a, y_b], color="red", s=80, label="Condiciones Iniciales")
            ax.set_title("Solución EDO por Diferencias Finitas", fontsize=14, fontweight='bold')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            toolbar = NavigationToolbar2Tk(canvas, win)
            toolbar.update()
            canvas._tkcanvas.pack(fill="both", expand=True)

            for entry in self.fd_entries:
                entry.delete(0, tk.END)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------------------- TAB SISTEMAS NO LINEALES -------------------------
    def create_nonlinear_tab(self):
        self.nl_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.nl_tab, text="Ecuaciones No Lineales")

        frame_inputs = ttk.LabelFrame(self.nl_tab, text="Datos de Entrada")
        frame_inputs.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame_inputs, text="Número de variables:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.nl_var_entry = ttk.Entry(frame_inputs)
        self.nl_var_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame_inputs, text="Número de iteraciones:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.nl_iter_entry = ttk.Entry(frame_inputs)
        self.nl_iter_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(frame_inputs, text="Ingresar funciones y valores", command=self.nl_input_funcs).grid(row=2, column=0, columnspan=2, pady=10)

        self.nl_result = tk.Text(self.nl_tab, height=15)
        self.nl_result.pack(fill="both", padx=10, pady=10)

    def nl_input_funcs(self):
        try:
            self.num_vars = int(self.nl_var_entry.get())
            self.num_iter = int(self.nl_iter_entry.get())
            self.nl_funcs = []
            self.nl_initials = []

            self.input_win = tk.Toplevel(self)
            self.input_win.title("Funciones y Valores Iniciales")

            for i in range(self.num_vars):
                ttk.Label(self.input_win, text=f"f{i+1}(x1..x{self.num_vars}):").grid(row=i, column=0, padx=5, pady=5)
                entry_f = ttk.Entry(self.input_win, width=30)
                entry_f.grid(row=i, column=1, padx=5, pady=5)
                self.nl_funcs.append(entry_f)

            for i in range(self.num_vars):
                ttk.Label(self.input_win, text=f"Valor inicial x{i+1}:").grid(row=i+self.num_vars, column=0, padx=5, pady=5)
                entry_x = ttk.Entry(self.input_win, width=30)
                entry_x.grid(row=i+self.num_vars, column=1, padx=5, pady=5)
                self.nl_initials.append(entry_x)

            ttk.Button(self.input_win, text="Calcular", command=self.calculate_nonlinear).grid(row=2*self.num_vars, column=0, columnspan=2, pady=10)

            self.nl_var_entry.delete(0, tk.END)
            self.nl_iter_entry.delete(0, tk.END)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def calculate_nonlinear(self):
        try:
            self.nl_result.delete("1.0", tk.END)
            f_list = [f.get() for f in self.nl_funcs]
            x0 = [float(x.get()) for x in self.nl_initials]

            result = newton_raphson_n_variables(f_list, x0, self.num_vars, self.num_iter)
            formatted_result = [[float(val) for val in row] for row in result]

            for i, vals in enumerate(formatted_result):
                self.nl_result.insert(tk.END, f"Iteración {i}: {vals}\n")

            if self.num_vars == 2:
                win = tk.Toplevel(self)
                win.title("Gráfica Sistema No Lineal")
                win.geometry("800x600")
                graph_nonlinear_equations(f_list[0], f_list[1], result)

            for entry in self.nl_funcs + self.nl_initials:
                entry.delete(0, tk.END)

            self.input_win.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------------------- TAB JACOBI -------------------------
    def create_jacobi_tab(self):
        self.jacobi_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.jacobi_tab, text="Jacobi")

        frame_inputs = ttk.LabelFrame(self.jacobi_tab, text="Datos de Entrada")
        frame_inputs.pack(fill="x", padx=10, pady=10)

        labels = ["Matriz A:", "Vector b:", "Tolerancia:", "Número de iteraciones:"]
        self.jacobi_entries = []
        for i, text in enumerate(labels):
            ttk.Label(frame_inputs, text=text).grid(row=i, column=0, sticky="w", padx=5, pady=5)
            entry = ttk.Entry(frame_inputs, width=50)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.jacobi_entries.append(entry)

        ttk.Button(frame_inputs, text="Calcular", command=self.calculate_jacobi).grid(row=len(labels), column=0, columnspan=2, pady=10)

        self.jacobi_result = tk.Text(self.jacobi_tab, height=15)
        self.jacobi_result.pack(fill="both", padx=10, pady=10)

    def calculate_jacobi(self):
        try:
            self.jacobi_result.delete("1.0", tk.END)
            A = sp.Matrix(eval(self.jacobi_entries[0].get()))
            b = sp.Matrix(eval(self.jacobi_entries[1].get()))
            tol = float(self.jacobi_entries[2].get())
            n = int(self.jacobi_entries[3].get())

            result = jacobi_method(A, b, tol, n)
            formatted_result = [[float(x) for x in row] for row in result]

            for i, vals in enumerate(formatted_result):
                self.jacobi_result.insert(tk.END, f"Iteración {i}: {vals}\n")

            for entry in self.jacobi_entries:
                entry.delete(0, tk.END)

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    app = NumericalApp()
    app.mainloop()

