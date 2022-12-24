import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from numpy.random import normal
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0
#0.11412213735155026

""" Solution """

np.random.seed(SEED)

class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        self.f_VAR = .5
        self.v_VAR = np.sqrt(2)

        self.f_kernel = ConstantKernel(constant_value=self.f_VAR, constant_value_bounds='fixed')*Matern(length_scale=0.5, nu=2.5, length_scale_bounds='fixed')
        self.v_kernel = ConstantKernel(constant_value=1.5, constant_value_bounds='fixed') + ConstantKernel(constant_value=self.v_VAR, constant_value_bounds='fixed')*Matern(length_scale=0.5, nu=2.5, length_scale_bounds='fixed')

        self.f_GP = GaussianProcessRegressor(kernel=self.f_kernel, alpha=0.15**2, random_state=SEED, optimizer=None) 
        self.v_GP = GaussianProcessRegressor(kernel=self.v_kernel, alpha=0.0001**2, random_state=SEED, optimizer=None) # What about the mean?

        self.v_min_safe_value = 1.2
        self.x_values = np.array([]).reshape(-1, domain.shape[0])
        self.y_values = np.array([]).reshape(-1, domain.shape[0])
        self.v_values = np.array([]).reshape(-1, domain.shape[0])


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """
        if self.x_values.size == 0:
            return domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(1, domain.shape[0])
        res = self.optimize_acquisition_function()
        
        res = np.reshape(res, (-1, 1))
        return res

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return x_values[ind]

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f (1,)

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        assert x.shape == (1,)
        beta_coef = .5
        #epsilon_coef = 1e-6
        #UCB:

        #x = self.s.transform(x)

        mu_f, std_f = self.f_GP.predict(np.reshape(x, (-1, 1)), return_std=True)
        mu_v, std_v = self.v_GP.predict(np.reshape(x, (-1, 1)), return_std=True)

        #assert std_f == np.sqrt(self.f_VAR)
        #assert std_v == np.sqrt(self.v_VAR)

        # std_f = np.sqrt(self.f_VAR)
        # std_v = np.sqrt(self.v_VAR)
        
        eic_est = np.squeeze(mu_f + beta_coef * std_f) * np.squeeze(norm.cdf((mu_v - self.v_min_safe_value) / std_v))
        #######

        #EI:
        # mu_f, std_f = self.f_GP.predict(np.reshape(x, (-1, 1)), return_std=True)
        # mu_v, std_v = self.v_GP.predict(np.reshape(x, (-1, 1)), return_std=True)
        
        # mu_sample = self.f_GP.predict(self.x_values)
        # z_f = (mu_f - np.max(mu_sample) - epsilon_coef)/std_f      
        # ei_est = std_f * (z_f*norm.cdf(z_f) + norm.pdf(z_f))
        
        # eic_est = np.squeeze(ei_est) * np.squeeze(norm.cdf((mu_v - self.v_min_safe_value) / std_v))        
        #######
        
        return eic_est

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters (1 x domain.shape[0])
        f: np.ndarray
            Model accuracy (1,)
        v: np.ndarray
            Model training speed (1,)
        """
        #Problemi:
        #1- Ma dobbiamo usare tutti i punti ogni volta?
        x = np.reshape(x, (-1, domain.shape[0]))
        assert x.shape == (1,1)
        assert f.shape == (1,)
        assert v.shape == (1,)
        self.x_values = np.vstack((self.x_values, x))
        self.y_values = np.vstack((self.y_values, f))
        self.v_values = np.vstack((self.v_values, v))

        #self.x_values = self.s.fit_transform(self.x_values)

        self.f_GP.fit(self.x_values, self.y_values)
        self.v_GP.fit(self.x_values, self.v_values)


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        #Problemi:
        #1- Come ricavare la soluzione ottimale con i dati che abbiamo?
        temp = self.y_values
        temp[self.v_values < self.v_min_safe_value] = np.NINF
        x_sol = np.argmax(temp)
        return self.x_values[x_sol]


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()