"""
pt_vs_metropolis_comparison.py

Compare parallel tempering to standard Metropolis sampling for the
2D Ising model at a single temperature near the critical point.

This script:
* loads energy time series from a parallel tempering run,
* runs an equivalent Metropolis-only simulation at the same temperature,
* computes autocorrelation functions and integrated autocorrelation times,
* estimates effective sample sizes and standard errors,
* produces comparison plots of traces, histograms, and autocorrelation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.ising2d import Ising2DModel
from src.metropolis import metropolis_single_flip_2d


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the comparison script.

    Returns:
        argparse.Namespace: Parsed arguments for the analysis.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare parallel tempering to standard Metropolis for the "
            "2D Ising model at a single temperature."
        )
    )
    parser.add_argument(
        "--lattice_size",
        type=int,
        default=24,
        help="Linear lattice size L (default: 24).",
    )
    parser.add_argument(
        "--temperature_index",
        type=int,
        default=None,
        help=(
            "Index of the replica in the parallel tempering ladder to "
            "analyse. Default: central replica."
        ),
    )
    parser.add_argument(
        "--pt_data_path",
        type=str,
        default="results/raw/ising2d_parallel_tempering_L_24.npz",
        help="Path to the .npz file produced by extensions.parallel_tempering.",
    )
    parser.add_argument(
        "--number_of_sweeps",
        type=int,
        default=20_000,
        help="Total sweeps for the Metropolis-only run (default: 20000).",
    )
    parser.add_argument(
        "--burn_in_sweeps",
        type=int,
        default=5_000,
        help="Burn-in sweeps for the Metropolis-only run (default: 5000).",
    )
    parser.add_argument(
        "--sampling_interval",
        type=int,
        default=50,
        help="Sampling interval in sweeps (default: 50).",
    )
    parser.add_argument(
        "--coupling_constant",
        type=float,
        default=1.0,
        help="Coupling constant J (default: 1.0).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=321,
        help="Random seed for the Metropolis-only run (default: 321).",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="results",
        help="Base output directory for figures (default: results).",
    )
    parser.add_argument(
        "--max_lag_fraction",
        type=float,
        default=0.25,
        help=(
            "Maximum lag as a fraction of the number of samples when "
            "computing autocorrelation (default: 0.25)."
        ),
    )
    return parser.parse_args()


def compute_energy(
    spins: np.ndarray,
    coupling_constant: float,
) -> float:
    """Compute total energy for a 2D Ising configuration.

    Args:
        spins (np.ndarray): Spin configuration (+/-1) of shape (L, L).
        coupling_constant (float): Nearest neighbour coupling constant J.

    Returns:
        float: Total energy of the configuration.
    """
    shifted_right = np.roll(spins, shift=-1, axis=1)
    shifted_down = np.roll(spins, shift=-1, axis=0)
    interaction_sum = np.sum(spins * (shifted_right + shifted_down))
    energy = -coupling_constant * float(interaction_sum)
    return energy


def run_metropolis_chain(
    lattice_size: int,
    temperature: float,
    number_of_sweeps: int,
    burn_in_sweeps: int,
    sampling_interval: int,
    coupling_constant: float,
    random_seed: int,
) -> np.ndarray:
    """Run a standard Metropolis simulation and record energy per site.

    Args:
        lattice_size (int): Linear lattice size L.
        temperature (float): Simulation temperature.
        number_of_sweeps (int): Total number of sweeps.
        burn_in_sweeps (int): Number of burn-in sweeps.
        sampling_interval (int): Interval between stored samples.
        coupling_constant (float): Coupling constant J.
        random_seed (int): Random seed.

    Returns:
        np.ndarray: Energy per site samples of shape (S,), where S is the
        number of stored samples.
    """
    if burn_in_sweeps >= number_of_sweeps:
        raise ValueError("Burn-in sweeps must be smaller than total sweeps.")

    np.random.seed(random_seed)
    model = Ising2DModel(
        lattice_size=lattice_size,
        coupling_constant=coupling_constant,
    )
    beta = 1.0 / float(temperature)
    number_of_sites = float(lattice_size * lattice_size)

    number_of_samples = (number_of_sweeps - burn_in_sweeps) // sampling_interval
    energy_samples = np.empty(number_of_samples, dtype=float)

    sample_index = 0
    for sweep_index in range(number_of_sweeps):
        metropolis_single_flip_2d(model, inverse_temperature=beta)
        if (
            sweep_index >= burn_in_sweeps
            and (sweep_index - burn_in_sweeps) % sampling_interval == 0
        ):
            energy = compute_energy(model.spins, coupling_constant=coupling_constant)
            energy_samples[sample_index] = energy / number_of_sites
            sample_index += 1

    return energy_samples


def autocorrelation(
    samples: np.ndarray,
    max_lag: int | None = None,
) -> np.ndarray:
    """Compute the normalised autocorrelation function of a time series.

    Args:
        samples (np.ndarray): One dimensional array of samples.
        max_lag (int | None): Maximum lag to compute. If None, uses the full
            length of the series minus one.

    Returns:
        np.ndarray: Autocorrelation values rho_k for lags k = 0, 1, ..., max_lag.
    """
    x = np.asarray(samples, dtype=float)
    x = x - np.mean(x)
    n = x.size

    if max_lag is None or max_lag >= n:
        max_lag = n - 1

    # Use FFT based convolution for efficiency.
    padded_length = 1
    while padded_length < 2 * n:
        padded_length *= 2

    fft_x = np.fft.rfft(x, n=padded_length)
    power_spectrum = fft_x * np.conjugate(fft_x)
    corr = np.fft.irfft(power_spectrum, n=padded_length)[: n]
    corr = corr / corr[0]  # normalise so that rho_0 = 1

    return corr[: max_lag + 1]


def integrated_autocorrelation_time(
    rho: np.ndarray,
) -> float:
    """Estimate the integrated autocorrelation time from rho_k.

    Uses the standard windowed sum:
        tau_int = 0.5 + sum_{k=1}^{k_max} rho_k
    where the sum stops once rho_k first becomes negative.

    Args:
        rho (np.ndarray): Autocorrelation values rho_k, starting at lag 0.

    Returns:
        float: Estimated integrated autocorrelation time.
    """
    if rho.size < 2:
        return 0.5

    tau_int = 0.5
    for k in range(1, rho.size):
        if rho[k] <= 0.0:
            break
        tau_int += rho[k]
    return float(tau_int)


def mc_statistics(
    samples: np.ndarray,
    max_lag_fraction: float = 0.25,
) -> Tuple[float, float, float, float, np.ndarray]:
    """Compute mean, variance, tau_int, effective sample size, and rho_k.

    Args:
        samples (np.ndarray): One dimensional array of samples.
        max_lag_fraction (float): Maximum lag as a fraction of the number of
            samples when computing the autocorrelation function.

    Returns:
        Tuple[float, float, float, float, np.ndarray]:
            mean, variance, tau_int, effective_sample_size, rho.
    """
    x = np.asarray(samples, dtype=float)
    n = x.size
    mean = float(np.mean(x))
    variance = float(np.var(x, ddof=1))

    max_lag = max(1, int(max_lag_fraction * n))
    rho = autocorrelation(x, max_lag=max_lag)
    tau_int = integrated_autocorrelation_time(rho)

    effective_sample_size = n / (2.0 * tau_int)
    return mean, variance, tau_int, effective_sample_size, rho


def make_plots(
    temperature: float,
    pt_energies: np.ndarray,
    metro_energies: np.ndarray,
    rho_pt: np.ndarray,
    rho_metro: np.ndarray,
    lattice_size: int,
    output_directory: str,
) -> None:
    """Create trace, histogram, and autocorrelation comparison plots.

    Args:
        temperature (float): Simulation temperature.
        pt_energies (np.ndarray): Energy per site samples from PT (S,).
        metro_energies (np.ndarray): Energy per site samples from Metropolis (S,).
        rho_pt (np.ndarray): Autocorrelation function for PT.
        rho_metro (np.ndarray): Autocorrelation function for Metropolis.
        lattice_size (int): Lattice size L.
        output_directory (str): Base directory for output figures.
    """
    figures_dir = Path(output_directory) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    n = len(pt_energies)
    sample_indices = np.arange(n, dtype=int)

    # Trace comparison.
    plt.figure()
    plt.plot(sample_indices, pt_energies, label="Parallel tempering", linestyle="-")
    plt.plot(
        sample_indices,
        metro_energies,
        label="Metropolis only",
        linestyle="--",
    )
    plt.xlabel("Sample index")
    plt.ylabel("Energy per site")
    plt.title(
        "Energy trace comparison at T = "
        f"{temperature:.3f}, L = {lattice_size}"
    )
    plt.legend()
    trace_path = (
        figures_dir
        / f"ising2d_pt_vs_metropolis_energy_trace_L_{lattice_size}_T_{temperature:.3f}.png"
    )
    plt.savefig(trace_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved trace comparison to {trace_path}")

    # Histogram comparison.
    plt.figure()
    plt.hist(
        pt_energies,
        bins=30,
        density=True,
        alpha=0.6,
        label="Parallel tempering",
    )
    plt.hist(
        metro_energies,
        bins=30,
        density=True,
        alpha=0.6,
        label="Metropolis only",
    )
    plt.xlabel("Energy per site")
    plt.ylabel("Probability density")
    plt.title(
        "Energy histogram comparison at T = "
        f"{temperature:.3f}, L = {lattice_size}"
    )
    plt.legend()
    hist_path = (
        figures_dir
        / f"ising2d_pt_vs_metropolis_energy_hist_L_{lattice_size}_T_{temperature:.3f}.png"
    )
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram comparison to {hist_path}")

    # Autocorrelation comparison.
    lags_pt = np.arange(rho_pt.size, dtype=int)
    lags_metro = np.arange(rho_metro.size, dtype=int)

    plt.figure()
    plt.plot(lags_pt, rho_pt, label="Parallel tempering", linestyle="-", marker="o")
    plt.plot(
        lags_metro,
        rho_metro,
        label="Metropolis only",
        linestyle="--",
        marker="x",
    )
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation rho")
    plt.title(
        "Energy autocorrelation comparison at T = "
        f"{temperature:.3f}, L = {lattice_size}"
    )
    plt.legend()
    ac_path = (
        figures_dir
        / f"ising2d_pt_vs_metropolis_energy_acf_L_{lattice_size}_T_{temperature:.3f}.png"
    )
    plt.savefig(ac_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved autocorrelation comparison to {ac_path}")


def main() -> None:
    """Entry point for the PT vs Metropolis comparison."""
    args = parse_arguments()

    data = np.load(args.pt_data_path)
    temperatures = data["temperatures"]
    energy_samples = data["energy_samples"]  # shape (R, S)

    number_of_replicas, number_of_samples = energy_samples.shape
    if number_of_samples == 0:
        raise RuntimeError("No PT samples available in the provided file.")

    if args.temperature_index is None:
        replica_index = number_of_replicas // 2
    else:
        replica_index = int(args.temperature_index)
        if replica_index < 0 or replica_index >= number_of_replicas:
            raise ValueError("temperature_index is out of range.")

    temperature = float(temperatures[replica_index])
    pt_energies = energy_samples[replica_index].copy()

    metro_energies = run_metropolis_chain(
        lattice_size=args.lattice_size,
        temperature=temperature,
        number_of_sweeps=args.number_of_sweeps,
        burn_in_sweeps=args.burn_in_sweeps,
        sampling_interval=args.sampling_interval,
        coupling_constant=args.coupling_constant,
        random_seed=args.random_seed,
    )

    # Align lengths if they differ by a sample.
    min_length = min(len(pt_energies), len(metro_energies))
    pt_energies = pt_energies[:min_length]
    metro_energies = metro_energies[:min_length]

    (
        mean_pt,
        var_pt,
        tau_pt,
        n_eff_pt,
        rho_pt,
    ) = mc_statistics(pt_energies, max_lag_fraction=args.max_lag_fraction)
    (
        mean_metro,
        var_metro,
        tau_metro,
        n_eff_metro,
        rho_metro,
    ) = mc_statistics(metro_energies, max_lag_fraction=args.max_lag_fraction)

    se_pt = np.sqrt(var_pt / n_eff_pt)
    se_metro = np.sqrt(var_metro / n_eff_metro)

    print("")
    print(f"Temperature T = {temperature:.3f}, lattice size L = {args.lattice_size}")
    print("Parallel tempering:")
    print(
        f"  mean(E) = {mean_pt:.5f}, var(E) = {var_pt:.5e}, "
        f"tau_int = {tau_pt:.2f}, N_eff = {n_eff_pt:.1f}, SE = {se_pt:.5e}"
    )
    print("Metropolis only:")
    print(
        f"  mean(E) = {mean_metro:.5f}, var(E) = {var_metro:.5e}, "
        f"tau_int = {tau_metro:.2f}, N_eff = {n_eff_metro:.1f}, SE = {se_metro:.5e}"
    )
    print(
        "Speed-up in effective sample size "
        f"(PT / Metropolis) ~= {n_eff_pt / n_eff_metro:.2f}"
    )
    print("")

    make_plots(
        temperature=temperature,
        pt_energies=pt_energies,
        metro_energies=metro_energies,
        rho_pt=rho_pt,
        rho_metro=rho_metro,
        lattice_size=args.lattice_size,
        output_directory=args.output_directory,
    )


if __name__ == "__main__":
    main()
