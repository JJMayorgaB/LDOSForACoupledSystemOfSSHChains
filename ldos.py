import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import os

# Create figures folder if it doesn't exist
figures_dir = "figures"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Configure matplotlib to use LaTeX with additional packages
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 30,
    'axes.labelsize': 32,
    'axes.titlesize': 34,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 30,
    'figure.titlesize': 36
})

# Main parameters
w = 1.0
a_val = 1.0    
v = 0.8      
x_eval = 0.5   
n = 5.0
m = 2000

# Normal parameters
A = w * a_val
m2 = w * a_val**2 / 2
m1 = abs(v - w)

# Prime parameters (different from normal ones)
A_prime = 1.1 * w * a_val  
m2_prime = 1.1* w * a_val**2 / 2 
m1_prime = abs(0.9*v - 1.1*w) 

# Parameter t
t = 1.2  

print("Normal parameters:")
print(f"A = {A}")
print(f"m1 = {m1}")
print(f"m2 = {m2}")
print("\nPrime parameters:")
print(f"A' = {A_prime}")
print(f"m1' = {m1_prime}")
print(f"m2' = {m2_prime}")
print(f"t = {t}")

# Discretization of ω space
Delta_omega = 2 * n / m
omega_values = np.linspace(-n, n, m)
eta = 5 * Delta_omega

print(f"\nDiscretization:")
print(f"n = {n}")
print(f"m = {m}")
print(f"Delta_omega = {Delta_omega:.6f}")
print(f"eta = {eta:.6f}")

# Calculation of d_+/-
discriminant_d = 1 - (4 * m1 * m2) / (A**2)
print(f"\nDiscriminant for d_+/-: {discriminant_d:.6f}")

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

# Calculation of xi_+/- (using prime parameters)
discriminant_xi = 1 - (4 * m1_prime * m2_prime) / (A_prime**2)
print(f"\nDiscriminant for xi_+/-: {discriminant_xi:.6f}")

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

# Calculation of normalization constants c1 and c2
# c1 = sqrt(d+ + d-) / |d+ - d-|
# c2 = sqrt(xi+ + xi-) / |xi+ - xi-|

# Calculate d+ and d- (inverses of d_plus_inv and d_minus_inv)
d_plus = 1.0 / d_plus_inv
d_minus = 1.0 / d_minus_inv

# Calculate xi+ and xi- (inverses of xi_plus_inv and xi_minus_inv)
xi_plus = 1.0 / xi_plus_inv
xi_minus = 1.0 / xi_minus_inv

# Calculate c1 
c1 = np.sqrt(d_plus + d_minus) / abs(d_plus - d_minus)
c1_squared = c1**2

# Calculate c2 
c2 = np.sqrt(xi_plus + xi_minus) / abs(xi_plus - xi_minus)
c2_squared = c2**2

exp_term_d = np.exp(-x_eval * d_plus_inv) - np.exp(-x_eval * d_minus_inv)
exp_term_xi = np.exp(x_eval * xi_plus_inv) - np.exp(x_eval * xi_minus_inv)

print(f"\nNormalization constants:")
print(f"d+ = {d_plus:.6f}, d- = {d_minus:.6f}")
print(f"xi+ = {xi_plus:.6f}, xi- = {xi_minus:.6f}")
print(f"c1 = sqrt(d+ + d-) / |d+ - d-| = {c1:.6f}")
print(f"c2 = sqrt(xi+ + xi-) / |xi+ - xi-| = {c2:.6f}")
print(f"c1^2 = {c1_squared:.6f}")
print(f"c2^2 = {c2_squared:.6f}")
print(f"Exponential term d: e^(-x/d+) - e^(-x/d-) = {exp_term_d}")
print(f"Exponential term xi: e^(x/xi+) - e^(x/xi-) = {exp_term_xi}")

# Verification of the new formulas
a_check = 2 * c1_squared * (abs(exp_term_d))**2
b_check = 16 * t**2 * c1_squared * c2_squared * (abs(exp_term_d))**2 * (abs(exp_term_xi))**2


a_x = 2 * c1_squared * (abs(exp_term_d))**2
b_x = 16 * t**2 * c1_squared * c2_squared * (abs(exp_term_d))**2 * (abs(exp_term_xi))**2

print(f"\nFunctions evaluated at x = {x_eval}:")
print(f"a({x_eval}) = 2c1^2[e^(-x/d+) - e^(-x/d-)]^2 = {a_x:.10f}")
print(f"b({x_eval}) = 16t^2c1^2c2^2[e^(-x/d+) - e^(-x/d-)]^2[e^(x/xi+) - e^(x/xi-)]^2 = {b_x:.6f}")

# Analytical continuation: omega -> omega + i*eta
omega_complex = omega_values + 1j * eta

# Calculation of G(omega) with analytical continuation
numerator = a_x * omega_complex**3
denominator = omega_complex**4 - b_x * omega_complex**2 + 0.5 * b_x**2

G_complex = numerator / denominator

# Calculation of -1/pi * Im{G(omega)}
imaginary_part = np.imag(G_complex)
result = -imaginary_part / np.pi

# Visualization
plt.figure(figsize=(10,8))

plt.plot(omega_values, result, 'r-', linewidth=1)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\rho(\omega)$')
plt.grid(True, alpha=0.3)
plt.xlim(-n,n)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "ldos_plot.svg"), format="svg")
plt.savefig(os.path.join(figures_dir, "ldos_plot.jpg"), format="jpg", dpi=300)

# Auxiliary function to calculate normalized constants given parameters
def calculate_normalized_constants(A_param, m1_param, m2_param, A_prime_param, m1_prime_param, m2_prime_param, x_eval_param):
    """
    Calculates normalized constants c1 and c2 for given parameters
    """
    # Calculate d_+/-
    discriminant_d_aux = 1 - (4 * m1_param * m2_param) / (A_param**2)
    
    if discriminant_d_aux >= 0:
        d_plus_inv_aux = (abs(A_param) / (2 * m2_param)) * (1 + np.sqrt(discriminant_d_aux))
        d_minus_inv_aux = (abs(A_param) / (2 * m2_param)) * (1 - np.sqrt(discriminant_d_aux))
    else:
        sqrt_disc_aux = np.sqrt(abs(discriminant_d_aux)) * 1j
        d_plus_inv_aux = (abs(A_param) / (2 * m2_param)) * (1 + sqrt_disc_aux)
        d_minus_inv_aux = (abs(A_param) / (2 * m2_param)) * (1 - sqrt_disc_aux)
    
    # Calculate xi_+/-
    discriminant_xi_aux = 1 - (4 * m1_prime_param * m2_prime_param) / (A_prime_param**2)
    
    if discriminant_xi_aux >= 0:
        xi_plus_inv_aux = (abs(A_prime_param) / (2 * m2_prime_param)) * (1 + np.sqrt(discriminant_xi_aux))
        xi_minus_inv_aux = (abs(A_prime_param) / (2 * m2_prime_param)) * (1 - np.sqrt(discriminant_xi_aux))
    else:
        sqrt_disc_xi_aux = np.sqrt(abs(discriminant_xi_aux)) * 1j
        xi_plus_inv_aux = (abs(A_prime_param) / (2 * m2_prime_param)) * (1 + sqrt_disc_xi_aux)
        xi_minus_inv_aux = (abs(A_prime_param) / (2 * m2_prime_param)) * (1 - sqrt_disc_xi_aux)
    
    # Calculate d+, d-, xi+, xi- 
    d_plus_aux = 1.0 / d_plus_inv_aux
    d_minus_aux = 1.0 / d_minus_inv_aux
    xi_plus_aux = 1.0 / xi_plus_inv_aux
    xi_minus_aux = 1.0 / xi_minus_inv_aux
    
    # Calculate constants using the new formulas
    c1_aux = np.sqrt(d_plus_aux + d_minus_aux) / abs(d_plus_aux - d_minus_aux)
    c2_aux = np.sqrt(xi_plus_aux + xi_minus_aux) / abs(xi_plus_aux - xi_minus_aux)
    c1_squared_aux = c1_aux**2
    c2_squared_aux = c2_aux**2
    
    # Calculate exponential terms
    exp_term_d_aux = np.exp(-x_eval_param * d_plus_inv_aux) - np.exp(-x_eval_param * d_minus_inv_aux)
    exp_term_xi_aux = np.exp(x_eval_param * xi_plus_inv_aux) - np.exp(x_eval_param * xi_minus_inv_aux)
    
    return c1_squared_aux, c2_squared_aux, exp_term_d_aux, exp_term_xi_aux

# Variation of t 
t_values = [0.0, 0.05, 0.5, 1.0, 2.0]

for i, t_var in enumerate(t_values):
    # Constants c1 and c2 don't depend on t, so we use the already calculated ones
    # Only recalculate b(x) with new t
    b_x_var = 16 * t_var**2 * c1_squared * c2_squared * (abs(exp_term_d))**2 * (abs(exp_term_xi))**2
    
    print(f"\nt = {t_var}:")
    print(f"  a(x) = {a_x:.10f}")
    print(f"  b(x) = {b_x_var:.6f}")
    
    # Recalculate G(omega)
    denominator_var = omega_complex**4 - b_x_var * omega_complex**2 + 0.5 * b_x_var**2
    G_complex_var = (a_x * omega_complex**3) / denominator_var
    result_var = -np.imag(G_complex_var) / np.pi
    
    # Create individual figure with correct figsize
    plt.figure(figsize=(10, 8))  # Changed from (70, 8) which was clearly an error
    plt.plot(omega_values, result_var, linewidth=1, color='red')
    plt.title(rf"$\rho(\omega)$ with $t = {t_var}$")
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\rho(\omega)$')
    plt.grid(True, alpha=0.3)
    
    # Set xlim according to t value
    if t_var <= 0.05:  # First two plots (t=0.0 and t=0.05)
        plt.xlim(-1.2, 1.2)
    elif t_var <= 0.5:  # Intermediate plot (t=0.5)
        plt.xlim(-3.2, 3.2)
    else:  # Last plots (t=1.0 and t=2.0)
        plt.xlim(-n, n)
    
    plt.tight_layout()
    
    # Save individual figure in both formats
    filename_base = f"ldos_t_{t_var:.2f}".replace(".", "_")
    plt.savefig(os.path.join(figures_dir, filename_base + ".svg"), format="svg")
    plt.savefig(os.path.join(figures_dir, filename_base + ".jpg"), format="jpg", dpi=300)

# GIF configuration 
t_min = 0.0
t_max = 3.0

t_segment1 = np.linspace(t_min, t_min + 0.15, 250, endpoint=False)
t_segment2 = np.linspace(t_min + 0.2, t_max, 100, endpoint=True)
t_values_anim = np.concatenate([t_segment1, t_segment2])

# Configure smaller font sizes specifically for GIF to reduce memory usage
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

# Create the figure
fig, ax = plt.subplots(figsize=(10, 6))
plt.title(rf"Evolution of $\rho(\omega)$ with $t$ varying from ${t_min}$ to ${t_max}$")
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\rho(\omega)$')
plt.grid(True, alpha=0.3)

# Initial line (empty)
line, = ax.plot([], [], lw=1, color='red')
ax.set_xlim(np.min(omega_values), np.max(omega_values))

# Calculate ylim dynamically by pre-calculating some values
y_max_global = 0
y_min_global = 0
for t_val in np.linspace(t_min, t_max, 20):  # Sample some t values
    b_x_temp = 16 * t_val**2 * c1_squared * c2_squared * (abs(exp_term_d))**2 * (abs(exp_term_xi))**2
    denominator_temp = omega_complex**4 - b_x_temp * omega_complex**2 + 0.5 * b_x_temp**2
    G_complex_temp = (a_x * omega_complex**3) / denominator_temp
    result_temp = -np.imag(G_complex_temp) / np.pi
    y_max_global = max(y_max_global, np.max(result_temp))
    y_min_global = min(y_min_global, np.min(result_temp))

# Set ylim with a margin
margin = 0.1 * (y_max_global - y_min_global)
ax.set_ylim(y_min_global - margin, y_max_global + margin)

# Initialization function
def init():
    line.set_data([], [])
    return (line,)

# Animation function
def animate(t_var):
  
    b_x_var = 16 * t_var**2 * c1_squared * c2_squared * (abs(exp_term_d))**2 * (abs(exp_term_xi))**2
    
    # Calculate G(omega)
    denominator_var = omega_complex**4 - b_x_var * omega_complex**2 + 0.5 * b_x_var**2
    G_complex_var = (a_x * omega_complex**3) / denominator_var
    result_var = -np.imag(G_complex_var) / np.pi
    
    # Update data
    line.set_data(omega_values, result_var)
    
    # Update ylim dynamically (optional)
    y_min_current = np.min(result_var)
    y_max_current = np.max(result_var)
    margin = 0.1 * (y_max_current - y_min_current) if y_max_current != y_min_current else 0.1
    ax.set_ylim(y_min_current - margin, y_max_current + margin)
    
    # Update title
    ax.set_title(rf"$\rho(\omega)$ with $t = {t_var:.2f}$")
    
    return (line,)

# Create the animation
anim = FuncAnimation(fig, animate, frames=t_values_anim,
                     init_func=init, blit=False, interval=100)

# Save the GIF
print("Saving GIFs with different frame rates")
anim.save(os.path.join(figures_dir, 'ldos_evolution15.gif'), writer='pillow', fps=15, dpi=200)
anim.save(os.path.join(figures_dir, 'ldos_evolution10.gif'), writer='pillow', fps=10, dpi=200)
anim.save(os.path.join(figures_dir, 'ldos_evolution30.gif'), writer='pillow', fps=30, dpi=200)