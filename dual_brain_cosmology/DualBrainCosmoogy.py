from scipy.integrate import solve_ivp
import numpy as np

class DualBraneCosmology:
    def __init__(self, g0, m_phi, lambda_brane, R_screening):
        """
        Initialize 5D Theory-Driven Parameters
        :param g0: Inter-brane coupling strength
        :param m_phi: Radion field mass (determines range of force)
        :param lambda_brane: Brane tension (5D projection scale)
        :param R_screening: Screening scale for structural growth
        """

        self.g0 = g0
        self.m_phi = m_phi
        self.lambda_brane = lambda_brane
        self.R_screening = R_screening


    def background_evolution(self, a):
        """
        Dynamical Radion Energy Density Fraction f_phi(a)
        Ensures GR recovery at early times (a -> 0)
        """
        # Radion density emerges as branes separate
        # f_phi(a) acts as the physical 'switching' function
        return self.g0 * (a ** 3) / (1 + (a / 0.5) ** 3)


    def coupled_system(self, t, y, k):
        """
        Coupled 5D Projection ODEs
        y = [delta_phi, delta_phi_dot, xi, xi_dot, E_weyl]
        """
        delta_phi, d_phi_dot, xi, xi_dot, E_weyl = y
        a = np.exp(t)  # Use log-scale for stability (t = ln(a))
        H = 1.0  # Simplified Hubble for kernel derivation

        # 1. Radion Perturbation Equation (Klein-Gordon)
        # Sourced by metric potentials (simplified here as k-coupling)
        dd_phi_dot = -2 * H * d_phi_dot - (k ** 2 + (a * self.m_phi) ** 2) * delta_phi

        # 2. Brane Bending Mode (xi)
        # This mode breaks k-a separability
        d_xi_dot = -H * xi_dot - k ** 2 * xi + self.g0 * delta_phi

        # 3. Weyl Projection Evolution (E_weyl)
        # Enforcing the Bianchi projection constraint
        d_E_weyl = -3 * H * E_weyl - k ** 2 * xi + self.lambda_brane * delta_phi

        return [d_phi_dot, dd_phi_dot, xi_dot, d_xi_dot, d_E_weyl]

    def derive_kernels(self, k_range, a_values):
        """
        Reconstruct Diagnostic Kernels mu and eta from the evolved system
        """

        mu_results = np.zeros((len(k_range), len(a_values)))
        eta_results = np.zeros((len(k_range), len(a_values)))

        for i, k in enumerate(k_range):
            # Initial conditions: small primordial perturbations
            y0 = [1e-6, 0, 1e-8, 0, 0]
            t_span = (np.log(a_values[0]), np.log(a_values[-1]))

            sol = solve_ivp(lambda t, y: self.coupled_system(t, y, k),
                            t_span, y0, t_eval=np.log(a_values))

        # Reconstruction Logic
        for j, a in enumerate(a_values):
            f_phi = self.background_evolution(a)
        delta_phi = sol.y[0][j]
        E_weyl = sol.y[4][j]

        # Derive mu: Modified Poisson (Energy sector)
        mu_results[i, j] = 1 + f_phi * (1 / (1 + (k * self.R_screening) ** -2))

        # Derive eta: Gravitational Slip (Shear sector)
        # Fixed by Weyl projection and radion anisotropic stress
        eta_results[i, j] = 1 - (E_weyl / (delta_phi + 1e-12))

        return mu_results, eta_results