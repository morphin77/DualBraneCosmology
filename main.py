import numpy as np

from loguru import logger

from dual_brain_cosmology.DualBrainCosmoogy import DualBraneCosmology


def main():
    # --- EXECUTION & MCMC MAPPING ---
    # Example parameters for a theory-driven run
    dbc = DualBraneCosmology(g0=0.05, m_phi=1.2, lambda_brane=0.1, R_screening=150.0)

    k_modes = np.logspace(-3, -1, 10)  # Cosmological scales (h/Mpc)
    scale_factors = np.linspace(0.1, 1.0, 20)

    mu, eta = dbc.derive_kernels(k_modes, scale_factors)

    # Lensing Potential Diagnostic (Sigma)
    sigma = 0.5 * mu * (1 + eta)

    logger.info(f"Calculated Sigma at z=0, k=0.01: {sigma[5, -1]:.4f}")


if __name__ == "__main__":
    main()
