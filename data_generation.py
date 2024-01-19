# data_generation.py

import numpy as np
from scipy.integrate import solve_ivp

def generate_explosive_kuramoto_data(N, t_span, t_eval, K, seed=None):
    """
    Generate data for the Explosive Kuramoto Model.

    Parameters:
    N : int
        Number of oscillators.
    t_span : list
        Time span for the simulation [start, end].
    t_eval : ndarray
        Time points at which to store the simulated values.
    K : float
        Coupling strength.
    seed : int, optional
        Seed for random number generation.

    Returns:
    sol : OdeResult
        Solution object containing the simulation results.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random initial phases
    y0 = np.random.uniform(-np.pi, np.pi, N)

    # Random natural frequencies
    omega = np.random.uniform(-1, 1, N)

    # Create an all-to-all adjacency matrix
    adjacency_matrix = np.ones((N, N)) - np.eye(N)

    # Define the ODE
    def kuramoto_ode(t, y):
        dydt = omega.copy()
        for i in range(N):
            for j in range(N):
                if adjacency_matrix[i, j] > 0:
                    dydt[i] += K / N * abs(omega[i]) * np.sin(y[j] - y[i])
        return dydt

    # Solve the differential equations
    sol = solve_ivp(kuramoto_ode, t_span, y0, t_eval=t_eval, method='RK45')
    return sol, omega

def sim_explosive_kuramoto_data(N, t_span, t_eval, K, omega, y0, adj_mat):
    """
    Generate data for the Explosive Kuramoto Model.

    Parameters:
    N : int
        Number of oscillators.
    t_span : list
        Time span for the simulation [start, end].
    t_eval : ndarray
        Time points at which to store the simulated values.
    K : float
        Coupling strength.
    seed : int, optional
        Seed for random number generation.

    Returns:
    sol : OdeResult
        Solution object containing the simulation results.
    """

    # Create an all-to-all adjacency matrix
    if adj_mat is not None:
        adjacency_matrix = adj_mat
    else:
        adjacency_matrix = np.ones((N, N)) - np.eye(N)

    # Define the ODE
    def kuramoto_ode(t, y):
        dydt = omega.copy()
        for i in range(N):
            for j in range(N):
                if adjacency_matrix[i, j] > 0:
                    dydt[i] += K / N * abs(omega[i]) * np.sin(y[j] - y[i])
        return dydt

    # Solve the differential equations
    sol = solve_ivp(kuramoto_ode, t_span, y0, t_eval=t_eval, method='RK45')
    return sol
