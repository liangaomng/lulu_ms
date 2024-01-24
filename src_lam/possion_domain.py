import numpy as np
import matplotlib.pyplot as plt

class PoissonEquation:
    def __init__(self, domain_size, lambda_param, mu_param):
        self.domain_size = domain_size
        self.lambda_param = lambda_param
        self.mu_param = mu_param


    def sample_to_solu(self, num_samples_boundary, num_samples_interior):
        # Sample boundary points
        x_boundary_samples = []
        y_boundary_samples = []
        while len(x_boundary_samples) < num_samples_boundary:
            # Randomly choose an edge and sample a point from it
            edge = np.random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                x_boundary_samples.append(np.random.uniform(-self.domain_size, self.domain_size))
                y_boundary_samples.append(self.domain_size)
            elif edge == 'bottom':
                x_boundary_samples.append(np.random.uniform(-self.domain_size, self.domain_size))
                y_boundary_samples.append(-self.domain_size)
            elif edge == 'left':
                x_boundary_samples.append(-self.domain_size)
                y_boundary_samples.append(np.random.uniform(-self.domain_size, self.domain_size))
            elif edge == 'right':
                x_boundary_samples.append(self.domain_size)
                y_boundary_samples.append(np.random.uniform(-self.domain_size, self.domain_size))

        # Sample interior points
        x_interior_samples = np.random.uniform(-self.domain_size, self.domain_size, num_samples_interior)
        y_interior_samples = np.random.uniform(-self.domain_size, self.domain_size, num_samples_interior)

        # Filter out any points that are on the boundary
        interior_samples = np.array([(x, y) for x, y in zip(x_interior_samples, y_interior_samples)
                                     if x not in (-self.domain_size, self.domain_size) and
                                     y not in (-self.domain_size, self.domain_size)])

        # If we have filtered out too many points, sample additional points
        while len(interior_samples) < num_samples_interior:
            x_additional = np.random.uniform(-self.domain_size, self.domain_size)
            y_additional = np.random.uniform(-self.domain_size, self.domain_size)
            if (x_additional not in (-self.domain_size, self.domain_size) and
                    y_additional not in (-self.domain_size, self.domain_size)):
                interior_samples = np.append(interior_samples, [(x_additional, y_additional)], axis=0)

        # Convert lists to numpy arrays
        boundary_samples = np.array(list(zip(x_boundary_samples, y_boundary_samples)))

        cat=np.concatenate((boundary_samples, interior_samples), axis=0)
        value=self.exact_solution(cat[:,0], cat[:,1])
        sample_solution=np.column_stack((cat, value))
        return sample_solution




    # Exact solution based on the given problem
    def exact_solution(self,x1, x2):
        sinh_lambda = np.sinh(self.lambda_param)
        sin_mu = np.sin(self.mu_param)
        U = (sin_mu / sinh_lambda) * (np.sinh(self.lambda_param * x1) + np.sin(self.mu_param * x1)) + \
            (sin_mu / sinh_lambda) * (np.sinh(self.lambda_param *x2) + np.sin(self.mu_param * x2))

        return U

    def plot_contour(self):
        x = np.linspace(-self.domain_size, self.domain_size, 200)
        y = np.linspace(-self.domain_size, self.domain_size, 200)
        X, Y = np.meshgrid(x, y)
        U = self.exact_solution(X, Y)

        plt.figure(figsize=(8, 8))
        contour = plt.contourf(X, Y, U, levels=15, cmap='bwr')
        plt.colorbar(contour)
        plt.title('Plot of the Exact Solution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

if __name__=="__main__":
    # Example usage:
    domain_size = 1.0
    lambda_param = 2  # or 2, based on the value of d
    mu_param = 30  # or 30, based on the value of d

    # Create an instance of the class
    poisson = PoissonEquation(domain_size, lambda_param, mu_param)

    # Plot the contour of the exact solution
    poisson.plot_contour()
    a = poisson.sample_to_solu(4000, 5000)
    #plot
    plt.figure(figsize=(8, 8))
    plt.scatter(a[:,0], a[:,1], c=a[:,2], cmap='bwr')
    plt.colorbar()
    plt.title('Plot of the Exact Solution')
    plt.xlabel('x')
    plt.show()
