import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import os

# Crear carpeta figures si no existe
figures_dir = "figures"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Configurar matplotlib para usar LaTeX con paquetes adicionales
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Parámetros principales
w = 1.0
a_val = 1.0    
v = 0.8      
x_eval = 0.5   
n = 5.0
m = 2000

# Parámetros normales
A = w * a_val
m2 = w * a_val**2 / 2
m1 = abs(v - w)

# Parámetros prime (diferentes de los normales)
A_prime = 1.1 * w * a_val  
m2_prime = 1.1* w * a_val**2 / 2 
m1_prime = abs(0.9*v - 1.1*w) 

# Parámetro t
t = 1.2  

print("Parametros normales:")
print(f"A = {A}")
print(f"m1 = {m1}")
print(f"m2 = {m2}")
print("\nParametros prime:")
print(f"A' = {A_prime}")
print(f"m1' = {m1_prime}")
print(f"m2' = {m2_prime}")
print(f"t = {t}")

# Discretización del espacio ω
Delta_omega = 2 * n / m
omega_values = np.linspace(-n, n, m)
eta = 5 * Delta_omega

print(f"\nDiscretización:")
print(f"n = {n}")
print(f"m = {m}")
print(f"Delta_omega = {Delta_omega:.6f}")
print(f"eta = {eta:.6f}")

# Calculo de d_+/-
discriminant_d = 1 - (4 * m1 * m2) / (A**2)
print(f"\nDiscriminante para d_+/-: {discriminant_d:.6f}")

if discriminant_d >= 0:
    d_plus_inv = (abs(A) / (2 * m2)) * (1 + np.sqrt(discriminant_d))
    d_minus_inv = (abs(A) / (2 * m2)) * (1 - np.sqrt(discriminant_d))
    print(f"d+^(-1) = {d_plus_inv:.6f}")
    print(f"d-^(-1) = {d_minus_inv:.6f}")
else:
    sqrt_disc = np.sqrt(abs(discriminant_d)) * 1j
    d_plus_inv = (abs(A) / (2 * m2)) * (1 + sqrt_disc)
    d_minus_inv = (abs(A) / (2 * m2)) * (1 - sqrt_disc)
    print(f"d+^(-1) = {d_plus_inv}")
    print(f"d-^(-1) = {d_minus_inv}")

# Calculo de xi_+/- (usando parametros prime)
discriminant_xi = 1 - (4 * m1_prime * m2_prime) / (A_prime**2)
print(f"\nDiscriminante para xi_+/-: {discriminant_xi:.6f}")

if discriminant_xi >= 0:
    xi_plus_inv = (abs(A_prime) / (2 * m2_prime)) * (1 + np.sqrt(discriminant_xi))
    xi_minus_inv = (abs(A_prime) / (2 * m2_prime)) * (1 - np.sqrt(discriminant_xi))
    print(f"xi+^(-1) = {xi_plus_inv:.6f}")
    print(f"xi-^(-1) = {xi_minus_inv:.6f}")
else:
    sqrt_disc_xi = np.sqrt(abs(discriminant_xi)) * 1j
    xi_plus_inv = (abs(A_prime) / (2 * m2_prime)) * (1 + sqrt_disc_xi)
    xi_minus_inv = (abs(A_prime) / (2 * m2_prime)) * (1 - sqrt_disc_xi)
    print(f"xi+^(-1) = {xi_plus_inv}")
    print(f"xi-^(-1) = {xi_minus_inv}")

#Calculo de las constantes de normalizacion c1 y c2
# c1 = sqrt(d+ + d-) / |d+ - d-|
# c2 = sqrt(xi+ + xi-) / |xi+ - xi-|

# Calcular d+ y d- (inversos de d_plus_inv y d_minus_inv)
d_plus = 1.0 / d_plus_inv
d_minus = 1.0 / d_minus_inv

# Calcular xi+ y xi- (inversos de xi_plus_inv y xi_minus_inv)
xi_plus = 1.0 / xi_plus_inv
xi_minus = 1.0 / xi_minus_inv

# Calcular c1 
c1 = np.sqrt(d_plus + d_minus) / abs(d_plus - d_minus)
c1_squared = c1**2

# Calcular c2 
c2 = np.sqrt(xi_plus + xi_minus) / abs(xi_plus - xi_minus)
c2_squared = c2**2

exp_term_d = np.exp(-x_eval * d_plus_inv) - np.exp(-x_eval * d_minus_inv)
exp_term_xi = np.exp(x_eval * xi_plus_inv) - np.exp(x_eval * xi_minus_inv)

print(f"\nConstantes de normalizacion:")
print(f"d+ = {d_plus:.6f}, d- = {d_minus:.6f}")
print(f"xi+ = {xi_plus:.6f}, xi- = {xi_minus:.6f}")
print(f"c1 = sqrt(d+ + d-) / |d+ - d-| = {c1:.6f}")
print(f"c2 = sqrt(xi+ + xi-) / |xi+ - xi-| = {c2:.6f}")
print(f"c1^2 = {c1_squared:.6f}")
print(f"c2^2 = {c2_squared:.6f}")
print(f"Termino exponencial d: e^(-x/d+) - e^(-x/d-) = {exp_term_d}")
print(f"Termino exponencial xi: e^(x/xi+) - e^(x/xi-) = {exp_term_xi}")

# Verificacion de las nuevas formulas
a_check = 2 * c1_squared * (abs(exp_term_d))**2
b_check = 16 * t**2 * c1_squared * c2_squared * (abs(exp_term_d))**2 * (abs(exp_term_xi))**2


a_x = 2 * c1_squared * (abs(exp_term_d))**2
b_x = 16 * t**2 * c1_squared * c2_squared * (abs(exp_term_d))**2 * (abs(exp_term_xi))**2

print(f"\nFunciones evaluadas en x = {x_eval}:")
print(f"a({x_eval}) = 2c1^2[e^(-x/d+) - e^(-x/d-)]^2 = {a_x:.10f}")
print(f"b({x_eval}) = 16t^2c1^2c2^2[e^(-x/d+) - e^(-x/d-)]^2[e^(x/xi+) - e^(x/xi-)]^2 = {b_x:.6f}")

# Extension analitica: omega -> omega + i*eta
omega_complex = omega_values + 1j * eta

# Calculo de G(omega) con extension analitica
numerator = a_x * omega_complex**3
denominator = omega_complex**4 - b_x * omega_complex**2 + 0.5 * b_x**2

G_complex = numerator / denominator

# Calculo de -1/pi * Im{G(omega)}
imaginary_part = np.imag(G_complex)
result = -imaginary_part / np.pi

# Visualización
plt.figure(figsize=(10, 8))

plt.plot(omega_values, result, 'r-', linewidth=1)
plt.title(r'$\rho(\omega)$')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\rho(\omega)$')
plt.grid(True, alpha=0.3)
plt.xlim(-n,n)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "ldos_plot.svg"), format="svg")
plt.savefig(os.path.join(figures_dir, "ldos_plot.jpg"), format="jpg", dpi=300)

# Funcion auxiliar para calcular constantes normalizadas dados los parametros
def calculate_normalized_constants(A_param, m1_param, m2_param, A_prime_param, m1_prime_param, m2_prime_param, x_eval_param):
    """
    Calcula las constantes c1 y c2 normalizadas para parametros dados
    """
    # Calcular d_+/-
    discriminant_d_aux = 1 - (4 * m1_param * m2_param) / (A_param**2)
    
    if discriminant_d_aux >= 0:
        d_plus_inv_aux = (abs(A_param) / (2 * m2_param)) * (1 + np.sqrt(discriminant_d_aux))
        d_minus_inv_aux = (abs(A_param) / (2 * m2_param)) * (1 - np.sqrt(discriminant_d_aux))
    else:
        sqrt_disc_aux = np.sqrt(abs(discriminant_d_aux)) * 1j
        d_plus_inv_aux = (abs(A_param) / (2 * m2_param)) * (1 + sqrt_disc_aux)
        d_minus_inv_aux = (abs(A_param) / (2 * m2_param)) * (1 - sqrt_disc_aux)
    
    # Calcular xi_+/-
    discriminant_xi_aux = 1 - (4 * m1_prime_param * m2_prime_param) / (A_prime_param**2)
    
    if discriminant_xi_aux >= 0:
        xi_plus_inv_aux = (abs(A_prime_param) / (2 * m2_prime_param)) * (1 + np.sqrt(discriminant_xi_aux))
        xi_minus_inv_aux = (abs(A_prime_param) / (2 * m2_prime_param)) * (1 - np.sqrt(discriminant_xi_aux))
    else:
        sqrt_disc_xi_aux = np.sqrt(abs(discriminant_xi_aux)) * 1j
        xi_plus_inv_aux = (abs(A_prime_param) / (2 * m2_prime_param)) * (1 + sqrt_disc_xi_aux)
        xi_minus_inv_aux = (abs(A_prime_param) / (2 * m2_prime_param)) * (1 - sqrt_disc_xi_aux)
    
    # Calcular d+, d-, xi+, xi- 
    d_plus_aux = 1.0 / d_plus_inv_aux
    d_minus_aux = 1.0 / d_minus_inv_aux
    xi_plus_aux = 1.0 / xi_plus_inv_aux
    xi_minus_aux = 1.0 / xi_minus_inv_aux
    
    # Calcular constantes usando las nuevas formulas
    c1_aux = np.sqrt(d_plus_aux + d_minus_aux) / abs(d_plus_aux - d_minus_aux)
    c2_aux = np.sqrt(xi_plus_aux + xi_minus_aux) / abs(xi_plus_aux - xi_minus_aux)
    c1_squared_aux = c1_aux**2
    c2_squared_aux = c2_aux**2
    
    # Calcular terminos exponenciales
    exp_term_d_aux = np.exp(-x_eval_param * d_plus_inv_aux) - np.exp(-x_eval_param * d_minus_inv_aux)
    exp_term_xi_aux = np.exp(x_eval_param * xi_plus_inv_aux) - np.exp(x_eval_param * xi_minus_inv_aux)
    
    return c1_squared_aux, c2_squared_aux, exp_term_d_aux, exp_term_xi_aux

# Variacion de t - GENERAR GRAFICAS SEPARADAS
t_values = [0.0, 0.05, 0.5, 1.0, 2.0]

for i, t_var in enumerate(t_values):
    # Las constantes c1 y c2 no dependen de t, asi que usamos las ya calculadas
    # Solo recalculamos b(x) con nuevo t
    b_x_var = 16 * t_var**2 * c1_squared * c2_squared * (abs(exp_term_d))**2 * (abs(exp_term_xi))**2
    
    print(f"\nt = {t_var}:")
    print(f"  a(x) = {a_x:.10f}")
    print(f"  b(x) = {b_x_var:.6f}")
    
    # Recalcular G(omega)
    denominator_var = omega_complex**4 - b_x_var * omega_complex**2 + 0.5 * b_x_var**2
    G_complex_var = (a_x * omega_complex**3) / denominator_var
    result_var = -np.imag(G_complex_var) / np.pi
    
    # Crear figura individual
    plt.figure(figsize=(10, 8))
    plt.plot(omega_values, result_var, linewidth=1, color='red')
    plt.title(rf"$\rho(\omega)$ con $t = {t_var}$")
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\rho(\omega)$')
    plt.grid(True, alpha=0.3)
    
    # Establecer xlim según el valor de t
    if t_var <= 0.05:  # Primeras dos gráficas (t=0.0 y t=0.05)
        plt.xlim(-1.2, 1.2)
    elif t_var <= 0.5:  # Gráfica intermedia (t=0.5)
        plt.xlim(-3.2, 3.2)
    else:  # Últimas gráficas (t=1.0 y t=2.0)
        plt.xlim(-n, n)
    
    plt.tight_layout()
    
    # Guardar figura individual en ambos formatos
    filename_base = f"ldos_t_{t_var:.2f}".replace(".", "_")
    plt.savefig(os.path.join(figures_dir, filename_base + ".svg"), format="svg")
    plt.savefig(os.path.join(figures_dir, filename_base + ".jpg"), format="jpg", dpi=300)


# Configuración del GIF 
t_min = 0.0
t_max = 3.0

t_segment1 = np.linspace(t_min, t_min + 0.15, 250, endpoint=False)
t_segment2 = np.linspace(t_min + 0.2, t_max, 100, endpoint=True)
t_values_anim = np.concatenate([t_segment1, t_segment2])

# Crear la figura
fig, ax = plt.subplots(figsize=(10, 6))
plt.title(rf"Evolucion de $\rho(\omega)$ con $t$ variando de ${t_min}$ a ${t_max}$ (normalizado)")
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\rho(\omega)$')
plt.grid(True, alpha=0.3)

# Línea inicial (vacía)
line, = ax.plot([], [], lw=1, color='red')
ax.set_xlim(np.min(omega_values), np.max(omega_values))

# Calcular ylim dinamicamente precalculando algunos valores
y_max_global = 0
y_min_global = 0
for t_val in np.linspace(t_min, t_max, 20):  # Muestrear algunos valores de t
    b_x_temp = 16 * t_val**2 * c1_squared * c2_squared * (abs(exp_term_d))**2 * (abs(exp_term_xi))**2
    denominator_temp = omega_complex**4 - b_x_temp * omega_complex**2 + 0.5 * b_x_temp**2
    G_complex_temp = (a_x * omega_complex**3) / denominator_temp
    result_temp = -np.imag(G_complex_temp) / np.pi
    y_max_global = max(y_max_global, np.max(result_temp))
    y_min_global = min(y_min_global, np.min(result_temp))

# Establecer ylim con un margen
margin = 0.1 * (y_max_global - y_min_global)
ax.set_ylim(y_min_global - margin, y_max_global + margin)

# Función de inicialización
def init():
    line.set_data([], [])
    return (line,)

# Función de animación
def animate(t_var):
  
    b_x_var = 16 * t_var**2 * c1_squared * c2_squared * (abs(exp_term_d))**2 * (abs(exp_term_xi))**2
    
    # Calcular G(omega)
    denominator_var = omega_complex**4 - b_x_var * omega_complex**2 + 0.5 * b_x_var**2
    G_complex_var = (a_x * omega_complex**3) / denominator_var
    result_var = -np.imag(G_complex_var) / np.pi
    
    # Actualizar los datos
    line.set_data(omega_values, result_var)
    
    # Actualizar ylim dinamicamente (opcional)
    y_min_current = np.min(result_var)
    y_max_current = np.max(result_var)
    margin = 0.1 * (y_max_current - y_min_current) if y_max_current != y_min_current else 0.1
    ax.set_ylim(y_min_current - margin, y_max_current + margin)
    
    # Actualizar título
    ax.set_title(rf"$\rho(\omega)$ con $t = {t_var:.2f}$")
    
    return (line,)

# Crear la animacion
anim = FuncAnimation(fig, animate, frames=t_values_anim,
                     init_func=init, blit=False, interval=100)

# Guardar el GIF
anim.save(os.path.join(figures_dir, 'ldos_evolution15.gif'), writer='pillow', fps=15, dpi=300)
anim.save(os.path.join(figures_dir, 'ldos_evolution10.gif'), writer='pillow', fps=10, dpi=300)
anim.save(os.path.join(figures_dir, 'ldos_evolution30.gif'), writer='pillow', fps=30, dpi=300)


