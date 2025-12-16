# **Proyecto de Simulación – Métodos Numéricos**  
**Autores:** *Joan Sebastián Saavedra* & *Diego Gómez* & Michael Perez

---

## **Descripción del Proyecto**

Este proyecto implementa y visualiza diversos **métodos numéricos clásicos** utilizados en análisis matemático, aproximación de funciones, solución de ecuaciones y modelamiento de sistemas. Incluye herramientas para graficar resultados, calcular errores y resolver problemas tanto lineales como no lineales.

El propósito principal es ofrecer una simulación clara y visual del comportamiento de cada método, permitiendo comprender su funcionamiento paso a paso.

---

## **Métodos Implementados**

### **1. Polinomio de Taylor**
- Generación del polinomio de Taylor de grado *n* alrededor de un punto \(x_0\).
- Cálculo de errores absoluto y relativo.
- Gráfica comparativa entre la función original y su aproximación.

### **2. Método de Newton–Raphson**
- Cálculo de raíces de funciones no lineales.
- Registro de iteraciones.
- Graficación del comportamiento de la convergencia.

### **3. Método de Diferencias Finitas**
- Resolución de ecuaciones diferenciales de la forma:
  \[
  y'' = p(x)\, y' + q(x)\, y + r(x)
  \]
- Implementación con condiciones de frontera.
- Generación de tabla de soluciones aproximadas.

### **4. Sistemas de Ecuaciones No Lineales (Jacobiano)**
- Resolución mediante Newton Multivariable.
- Cálculo del jacobiano simbólico y su evaluación numérica.
- Trayectoria de los puntos iterados en un plano.

### **5. Método de Jacobi para Sistemas Lineales**
- Resolución de sistemas lineales de la forma \(Ax = b\).
- Iteraciones controladas por tolerancia o número máximo de iteraciones.
- Reporte de convergencia.

### **6. Método de biseccion**
-  El método de bisección es una técnica numérica para encontrar raíces de una ecuación 
f(x)=0. Consiste en dividir un intervalo donde la función cambia de signo hasta aproximarse a la raíz. Es un método simple y confiable.

### **7. Método de lagrange**
El método de Lagrange permite aproximar una función mediante un polinomio que pasa por un conjunto de puntos conocidos. Se utiliza para estimar valores intermedios cuando solo se dispone de datos discretos.

---

