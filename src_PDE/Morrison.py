import numpy as np
import matplotlib.pyplot as plt

class FluidForceVisualizer:
    def __init__(self, density, diameter, drag_coeff, added_mass_coeff, velocity_potential_func, time_range, num_points=100):
        self.rho = density
        self.diameter = diameter
        self.radius = diameter / 2
        self.Cd = drag_coeff
        self.Cm = added_mass_coeff
        self.A = np.pi * self.radius**2
        self.V0 = self.A
        self.velocity_potential_func = velocity_potential_func
        self.time = np.linspace(*time_range, num_points)

    def compute_forces(self):
        u_x, u_y = self.velocity_potential_func(self.time)
        du_x_dt = np.gradient(u_x, self.time)
        du_y_dt = np.gradient(u_y, self.time)

        f_d_x = 0.5 * self.Cd * self.rho * self.A * np.abs(u_x) * u_x
        f_i_x = (self.rho * self.V0 + self.Cm * self.rho * self.V0) * du_x_dt

        f_d_y = 0.5 * self.Cd * self.rho * self.A * np.abs(u_y) * u_y
        f_i_y = (self.rho * self.V0 + self.Cm * self.rho * self.V0) * du_y_dt

        return f_d_x, f_i_x, f_d_y, f_i_y

    def plot_forces(self):
        f_d_x, f_i_x, f_d_y, f_i_y = self.compute_forces()

        plt.figure(figsize=(14, 12))
        plt.subplot(2, 2, 1)
        plt.plot(self.time, f_d_x, label='Damping Force (X)')
        plt.title('Damping Force over Time (X)')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(self.time, f_i_x, label='Inertia Force (X)')
        plt.title('Inertia Force over Time (X)')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(self.time, f_d_y, label='Damping Force (Y)')
        plt.title('Damping Force over Time (Y)')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(self.time, f_i_y, label='Inertia Force (Y)')
        plt.title('Inertia Force over Time (Y)')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

# 示例：使用一个复杂的速度势函数
def complex_velocity_potential(t, frequency=1000.0, amplitude=1.0):
    # 示例：x方向是一个正弦波，y方向是一个余弦波
    u_x = amplitude * np.sin(frequency * t)
    u_y = amplitude * np.cos(frequency * t)
    return u_x, u_y

visualizer = FluidForceVisualizer(
    density=1000,
    diameter=1.0,
    drag_coeff=1.2,
    added_mass_coeff=2.0,
    velocity_potential_func=complex_velocity_potential,
    time_range=(0, 2*np.pi / 1000.0),
    num_points=100
)

visualizer.plot_forces()
