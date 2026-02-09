#!/usr/bin/env python3
"""
Chandler Wobble Simulation using Laplace's Tidal Equation (LTE) approach

This module implements a mathematical geoenergy treatment of the Chandler wobble
mechanism based on the approach described in Mathematical GeoEnergy (Wiley, 2018).

The Chandler wobble is a small variation in the Earth's axis of rotation,
with a period of approximately 433 days. The wobble emerges from aliasing between
lunar draconic forcing (~27.21 days) and annual/semi-annual inertia impulses:
    T_CW = (1/2) * |1/(1/T_d - 13/T_y)| ≈ 433 days
where the factor of 1/2 arises from quadratic (π-symmetric) coupling at the poles.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Tuple, List, Optional
import argparse


class ChandlerWobble:
    """
    Chandler Wobble model using Laplace's Tidal Equation approach.
    
    The model solves the differential equation for the wobble amplitude
    driven by periodic forcing terms.
    """
    
    def __init__(self, params: Optional[dict] = None):
        """
        Initialize the Chandler Wobble model.
        
        Parameters:
        -----------
        params : dict, optional
            Model parameters including:
            - 'period': Chandler wobble period in days (default: 433)
            - 'damping': Damping coefficient (default: 0.01)
            - 'forcing_amp': Forcing amplitude (default: 0.1)
            - 'forcing_freq': Forcing frequency array (default: tidal frequencies)
        """
        if params is None:
            params = {}
            
        # Default parameters
        self.period = params.get('period', 433.0)  # days
        self.omega = 2 * np.pi / self.period  # angular frequency
        self.damping = params.get('damping', 0.01)
        self.forcing_amp = params.get('forcing_amp', 0.1)
        
        # Tidal forcing frequencies (rad/day)
        # Based on lunar and solar tidal components
        self.forcing_freq = params.get('forcing_freq', [
            2 * np.pi / 365.25,      # Annual
            2 * np.pi / 27.32,       # Tropical month
            2 * np.pi / 13.66,       # Semi-monthly
            2 * np.pi / 182.62,      # Semi-annual
        ])
        
    def lte_equation(self, state: np.ndarray, t: float, params: dict) -> np.ndarray:
        """
        Laplace's Tidal Equation for Chandler wobble.
        
        Differential equation: d²x/dt² + 2γ dx/dt + ω₀²x = F(t)
        where F(t) is the forcing function from tidal components.
        
        Parameters:
        -----------
        state : array_like
            State vector [x, dx/dt]
        t : float
            Time (days)
        params : dict
            Model parameters
            
        Returns:
        --------
        dstate : array_like
            Time derivatives [dx/dt, d²x/dt²]
        """
        x, v = state
        
        # Calculate forcing function from multiple tidal components
        forcing = 0.0
        for i, freq in enumerate(self.forcing_freq):
            # Each component contributes with its own amplitude and phase
            amp = params.get(f'amp_{i}', self.forcing_amp / len(self.forcing_freq))
            phase = params.get(f'phase_{i}', 0.0)
            forcing += amp * np.cos(freq * t + phase)
        
        # LTE: d²x/dt² = -2γ dx/dt - ω₀²x + F(t)
        dvdt = -2 * self.damping * v - self.omega**2 * x + forcing
        
        return [v, dvdt]
    
    def solve(self, t_span: Tuple[float, float], 
              t_eval: Optional[np.ndarray] = None,
              initial_state: Optional[np.ndarray] = None,
              params: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the Chandler wobble equation.
        
        Parameters:
        -----------
        t_span : tuple
            Time span (t_start, t_end) in days
        t_eval : array_like, optional
            Times at which to evaluate the solution
        initial_state : array_like, optional
            Initial state [x₀, v₀], default is [0.0, 0.0]
        params : dict, optional
            Model parameters to use (default uses instance parameters)
            
        Returns:
        --------
        t : array_like
            Time points
        solution : array_like
            Solution array with shape (len(t), 2) containing [x, dx/dt]
        """
        if initial_state is None:
            initial_state = [0.0, 0.0]
            
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)
            
        if params is None:
            params = {}
        
        # Solve the ODE
        solution = odeint(self.lte_equation, initial_state, t_eval, args=(params,))
        
        return t_eval, solution
    
    def fit_to_data(self, t_data: np.ndarray, x_data: np.ndarray,
                    initial_params: Optional[dict] = None) -> dict:
        """
        Fit model parameters to observed data.
        
        Parameters:
        -----------
        t_data : array_like
            Time points of observations (days)
        x_data : array_like
            Observed wobble amplitude
        initial_params : dict, optional
            Initial guess for parameters
            
        Returns:
        --------
        fitted_params : dict
            Optimized model parameters
        """
        if initial_params is None:
            initial_params = {}
        
        # Parameter vector for optimization
        # [damping, forcing_amp, amp_0, amp_1, ..., phase_0, phase_1, ...]
        n_freq = len(self.forcing_freq)
        x0 = [
            initial_params.get('damping', self.damping),
            initial_params.get('forcing_amp', self.forcing_amp),
        ]
        
        # Add amplitudes and phases for each frequency component
        for i in range(n_freq):
            x0.append(initial_params.get(f'amp_{i}', self.forcing_amp / n_freq))
        for i in range(n_freq):
            x0.append(initial_params.get(f'phase_{i}', 0.0))
        
        def objective(x):
            """Objective function: sum of squared residuals."""
            params = {
                'damping': x[0],
                'forcing_amp': x[1],
            }
            for i in range(n_freq):
                params[f'amp_{i}'] = x[2 + i]
                params[f'phase_{i}'] = x[2 + n_freq + i]
            
            # Temporarily update model parameters
            old_damping = self.damping
            old_forcing_amp = self.forcing_amp
            self.damping = params['damping']
            self.forcing_amp = params['forcing_amp']
            
            # Solve with current parameters
            _, solution = self.solve((t_data[0], t_data[-1]), t_eval=t_data,
                                    initial_state=[x_data[0], 0.0], params=params)
            
            # Restore original parameters
            self.damping = old_damping
            self.forcing_amp = old_forcing_amp
            
            # Calculate residuals
            residuals = solution[:, 0] - x_data
            return np.sum(residuals**2)
        
        # Optimize
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-8})
        
        # Extract fitted parameters
        fitted_params = {
            'damping': result.x[0],
            'forcing_amp': result.x[1],
        }
        for i in range(n_freq):
            fitted_params[f'amp_{i}'] = result.x[2 + i]
            fitted_params[f'phase_{i}'] = result.x[2 + n_freq + i]
        
        return fitted_params
    
    def plot_solution(self, t: np.ndarray, solution: np.ndarray,
                     t_data: Optional[np.ndarray] = None,
                     x_data: Optional[np.ndarray] = None,
                     save_path: Optional[str] = None):
        """
        Plot the Chandler wobble solution.
        
        Parameters:
        -----------
        t : array_like
            Time points
        solution : array_like
            Solution array [x, dx/dt]
        t_data : array_like, optional
            Observed time points
        x_data : array_like, optional
            Observed data points
        save_path : str, optional
            Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot position
        ax1.plot(t, solution[:, 0], 'b-', label='Model', linewidth=2)
        if t_data is not None and x_data is not None:
            ax1.plot(t_data, x_data, 'r.', label='Data', markersize=4, alpha=0.5)
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Wobble Amplitude')
        ax1.set_title('Chandler Wobble - Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot velocity
        ax2.plot(t, solution[:, 1], 'g-', linewidth=2)
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Chandler Wobble - Velocity')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Figure saved to {save_path}")
        else:
            plt.show()


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Chandler wobble data from file.
    
    Expected format: Two columns (time, amplitude) separated by whitespace.
    
    Parameters:
    -----------
    filename : str
        Path to data file
        
    Returns:
    --------
    t : array_like
        Time points (converted to days from years)
    x : array_like
        Wobble amplitude
    """
    data = np.loadtxt(filename)
    t = data[:, 0] * 365.25  # Convert years to days
    x = data[:, 1]
    return t, x


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Chandler Wobble Simulation using Laplace\'s Tidal Equation'
    )
    parser.add_argument('--mode', choices=['simulate', 'fit'], default='simulate',
                       help='Operation mode: simulate or fit to data')
    parser.add_argument('--data', type=str, help='Path to data file for fitting')
    parser.add_argument('--period', type=float, default=433.0,
                       help='Chandler wobble period in days (default: 433)')
    parser.add_argument('--duration', type=float, default=3650.0,
                       help='Simulation duration in days (default: 3650, ~10 years)')
    parser.add_argument('--output', type=str, help='Output file for plot')
    
    args = parser.parse_args()
    
    # Create model
    model = ChandlerWobble({'period': args.period})
    
    if args.mode == 'simulate':
        print(f"Simulating Chandler wobble for {args.duration} days...")
        print(f"Period: {args.period} days")
        
        # Run simulation
        t, solution = model.solve((0, args.duration))
        
        print(f"Simulation complete. Generated {len(t)} data points.")
        print(f"Max amplitude: {np.max(np.abs(solution[:, 0])):.6f}")
        print(f"Min amplitude: {np.min(np.abs(solution[:, 0])):.6f}")
        
        # Plot results
        model.plot_solution(t, solution, save_path=args.output)
        
    elif args.mode == 'fit':
        if args.data is None:
            print("Error: --data argument required for fit mode")
            return
        
        print(f"Loading data from {args.data}...")
        t_data, x_data = load_data(args.data)
        
        print(f"Loaded {len(t_data)} data points")
        print(f"Time range: {t_data[0]:.1f} to {t_data[-1]:.1f} days")
        
        print("Fitting model to data...")
        fitted_params = model.fit_to_data(t_data, x_data)
        
        print("\nFitted parameters:")
        for key, value in fitted_params.items():
            print(f"  {key}: {value:.6f}")
        
        # Generate solution with fitted parameters
        t_sim, solution = model.solve((t_data[0], t_data[-1]), t_eval=t_data,
                                      initial_state=[x_data[0], 0.0],
                                      params=fitted_params)
        
        # Calculate fit quality
        residuals = solution[:, 0] - x_data
        rms_error = np.sqrt(np.mean(residuals**2))
        correlation = np.corrcoef(solution[:, 0], x_data)[0, 1]
        
        print(f"\nFit quality:")
        print(f"  RMS error: {rms_error:.6f}")
        print(f"  Correlation: {correlation:.6f}")
        
        # Plot results
        model.plot_solution(t_sim, solution, t_data, x_data, save_path=args.output)


if __name__ == '__main__':
    main()
