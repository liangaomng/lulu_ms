from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Define the system of equations
def system(t, z, wn, p):
    u, v = z
    dudt = v
    dvdt = (p - u**2) * v - wn * u
    return [dudt, dvdt]

# Parameters
wn = 1.0  # Example value for wn
p = 2.0  # Example value for p
z0 = [1.0, 0.0]  # Initial conditions: [u(0), v(0)]

# Time span for the integration
t_span = [0, 20]

# Numerically solve the system of equations
solution = solve_ivp(system, t_span, z0, args=(wn, p), dense_output=True)

# Evaluate the solution at 500 points within the time span
t_eval = np.linspace(t_span[0], t_span[1], 500)
u, v = solution.sol(t_eval)

# Plotting the solution
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t_eval, u, label='u(t)')
plt.xlabel('Time')
plt.ylabel('u')
plt.title('Numerical Simulation of the Given Equation')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_eval, v, label='v(t)')
plt.xlabel('Time')
plt.ylabel('v')
plt.legend()

plt.tight_layout()
plt.show()
