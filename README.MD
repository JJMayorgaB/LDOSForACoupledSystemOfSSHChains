# Local Density of States (LDOS) Calculator for Coupled SSH Chains

This repository contains a Python implementation for calculating and visualizing the Local Density of States (LDOS) of two coupled topological Su-Schrieffer-Heeger (SSH) chains using Green's function methods.

## Overview

The SSH model is a fundamental theoretical framework for understanding topological properties in one-dimensional systems, originally developed to model soliton formation in polyacetylene. This implementation focuses on studying the interaction between edge states of two coupled SSH chains through tunneling interactions.

## Directory Structure

```
Fismatavan/
├── figures/                    # Output directory for generated plots and animations
│   ├── ldos_plot.jpg          # Main LDOS plot (JPEG format)
│   ├── ldos_plot.svg          # Main LDOS plot (SVG format)
│   ├── ldos_t_*.jpg           # Individual LDOS plots for different coupling values
│   ├── ldos_t_*.svg           # Individual LDOS plots (SVG format)
│   ├── ldos_evolution*.gif    # Animated evolution of LDOS with varying number of FPS
│   └── ...
├── ldos.py                    # Main Python script for LDOS calculations
├── log.txt                    # Output log with calculated parameters
└── README.md                  # This file
```

## Theory Background

### SSH Model and Edge States

The system consists of two coupled SSH chains with edge states described by:

- **Right edge state**: `|ψ_R⟩ = c_R(e^(-x/d_+) - e^(-x/d_-))|φ⟩`
- **Left edge state**: `|ψ_L⟩ = c_L(e^(x/ξ_+) - e^(x/ξ_-))|φ⟩`

Where the decay parameters are given by:
- `d_±^(-1) = |A|/(2m_2) * (1 ± √(1 - 4m_1m_2/A²))`
- `ξ_±^(-1) = |A'|/(2m_2') * (1 ± √(1 - 4m_1'm_2'/A'²))`

### Green's Function Approach

The coupled system is analyzed using the Dyson equation with a tunneling Hamiltonian:

```
V̂ = t(|ψ_R⟩⟨ψ_L| + |ψ_L⟩⟨ψ_R|)
```

The resulting Green's function for the right edge state is:

```
G_RR(x,ω) = [a(x)ω³] / [ω⁴ - b(x)ω² + ½b²(x)]
```

Where:
- `a(x) = 2c₁²[e^(-x/d_+) - e^(-x/d_-)]²`
- `b(x) = 16t²c₁²c₂²[e^(-x/d_+) - e^(-x/d_-)]²[e^(x/ξ_+) - e^(x/ξ_-)]²`

### Normalization Constants

The normalization constants are calculated as:
- `c₁ = √(d_+ + d_-) / |d_+ - d_-|`
- `c₂ = √(ξ_+ + ξ_-) / |ξ_+ - ξ_-|`

## Code Functionality

### Main Features

1. **Parameter Calculation**: Computes all relevant physical parameters including decay lengths, normalization constants, and coupling terms.

2. **LDOS Computation**: Calculates the Local Density of States using the formula:
   ```
   ρ(ω) = -1/π * Im{G(ω + iη)}
   ```
   where η is a small imaginary part for analytical continuation.

3. **Visualization**: Generates multiple types of plots:
   - Static LDOS plots for different coupling strengths
   - Animated evolution showing the transition from uncoupled to strongly coupled regimes

4. **Parameter Studies**: Systematically varies the coupling parameter `t` to study the evolution of the LDOS.

### Key Parameters

- **Physical Parameters**:
  - `w = 1.0`: Hopping amplitude
  - `a_val = 1.0`: Lattice parameter
  - `v = 0.8`: Intra-cell hopping
  - `x_eval = 0.5`: Evaluation position
  - `t`: Tunneling coupling strength (varied)

- **Numerical Parameters**:
  - `n = 5.0`: Frequency range [-n, n]
  - `m = 2000`: Number of discretization points
  - `η = 5Δω`: Imaginary part for analytical continuation

### Output

The code generates:

1. **Console Output**: Detailed parameter values and intermediate calculations
2. **Static Plots**: Individual LDOS plots for t = [0.0, 0.05, 0.5, 1.0, 2.0]
3. **Animations**: GIF files showing smooth evolution of LDOS with varying coupling strength
4. **Log File**: Complete parameter dump for reference

## Usage

```bash
python ldos.py
```

### Requirements

- Python 3.x
- NumPy
- Matplotlib
- IPython (for display utilities)

### Installation

```bash
pip install numpy matplotlib ipython
```

## Results Interpretation

The LDOS plots reveal several key physics:

- **t = 0**: Uncoupled system showing isolated edge state peaks
- **Small t**: Weak coupling regime with slight peak splitting
- **Intermediate t**: Hybridization of edge states becomes apparent
- **Large t**: Strong coupling regime with significant spectral weight redistribution

The animated evolution demonstrates the smooth transition between these regimes, providing insight into the nature of edge state coupling in topological systems.

## Applications

This code is useful for:
- Understanding topological edge state physics
- Designing quantum devices with controlled edge state interactions
- Educational demonstrations of Green's function methods
- Research into coupled topological systems

## References

### Foundational SSH Model and Topological Physics

1. **Su, W. P., Schrieffer, J. R., & Heeger, A. J.** (1979). Solitons in Polyacetylene. *Physical Review Letters*, 42(25), 1698-1701. [DOI: 10.1103/PhysRevLett.42.1698](https://link.aps.org/doi/10.1103/PhysRevLett.42.1698)
   - *The original SSH model paper that introduced the theoretical framework for studying topological solitons in one-dimensional systems, forming the foundation for this work.*

2. **Kitaev, A. Yu.** (2001). Unpaired Majorana fermions in quantum wires. *Physics-Uspekhi*, 44(10S), 131. [DOI: 10.1070/1063-7869/44/10S/S29](https://dx.doi.org/10.1070/1063-7869/44/10S/S29)
   - *Seminal work on topological superconductors and Majorana edge states, providing theoretical context for understanding topological edge modes in coupled systems.*

### Experimental Realizations and Applications

3. **von Klitzing, K., Dorda, G., & Pepper, M.** (1980). New Method for High-Accuracy Determination of the Fine-Structure Constant Based on Quantized Hall Resistance. *Physical Review Letters*, 45(6), 494-497. [DOI: 10.1103/PhysRevLett.45.494](https://link.aps.org/doi/10.1103/PhysRevLett.45.494)
   - *Discovery of the quantum Hall effect, demonstrating topological quantum phases in condensed matter systems and establishing experimental precedent for topological edge states.*

4. **Lu, L., Joannopoulos, J. D., & Soljačić, M.** (2014). Topological photonics. *Nature Photonics*, 8(11), 821-829. [DOI: 10.1038/nphoton.2014.248](https://doi.org/10.1038/nphoton.2014.248)
   - *Comprehensive review of topological concepts in photonic systems, relevant for understanding optical implementations of SSH-like models and coupled topological structures.*

### Local Density of States Theory and Methods

5. **Mignuzzi, S., Vezzoli, S., Horsley, S. A. R., Barnes, W. L., Maier, S. A., & Sapienza, R.** (2019). Nanoscale Design of the Local Density of Optical States. *Nano Letters*, 19(3), 1613-1617. [DOI: 10.1021/acs.nanolett.8b04515](https://doi.org/10.1021/acs.nanolett.8b04515)
   - *Modern treatment of LDOS calculations and design principles, providing methodological background for the Green's function approach used in this implementation.*

6. **Iffländer, T., Rolf-Pissarczyk, S., Winking, L., Ulbrich, R. G., Al-Zubi, A., Blügel, S., & Wenderoth, M.** (2015). Local Density of States at Metal-Semiconductor Interfaces: An Atomic Scale Study. *Physical Review Letters*, 114(14), 146804. [DOI: 10.1103/PhysRevLett.114.146804](https://link.aps.org/doi/10.1103/PhysRevLett.114.146804)
   - *Experimental and theoretical study of LDOS at interfaces, demonstrating the importance of local electronic structure analysis in understanding coupled quantum systems.*

### Theoretical Context

The theoretical framework implemented in this code draws primarily from the SSH model [1] for understanding topological edge states, extended to coupled systems using Green's function methods. The LDOS calculation approach follows established quantum mechanical principles [5,6], while the physical motivation comes from both condensed matter [3] and photonic [4] realizations of topological systems. The study of coupled edge modes is particularly relevant for understanding Majorana fermions [2] and other exotic topological phases.

## License

Academic Project - Universidad Nacional de Colombia
Advanced Mathematical Physics - 2025-I

