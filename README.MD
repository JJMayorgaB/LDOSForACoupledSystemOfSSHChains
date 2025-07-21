# 🔬 Local Density of States (LDOS) Calculator for Coupled SSH Chains

This repository contains a Python implementation for calculating and visualizing the Local Density of States (LDOS) of two coupled topological Su-Schrieffer-Heeger (SSH) chains using Green's function methods.

## 📋 Overview

The SSH model is a fundamental theoretical framework for understanding topological properties in one-dimensional systems, originally developed to model soliton formation in polyacetylene. This implementation focuses on studying the interaction between edge states of two coupled SSH chains through tunneling interactions.

## 📁 Repository Structure

```
Fismatavan/
├── figures/                   # Output directory for generated plots and animations
│   ├── ldos_plot.jpg          # Main LDOS plot (JPEG format)
│   ├── ldos_plot.svg          # Main LDOS plot (SVG format)
│   ├── ldos_t_0_00.jpg        # LDOS plot for t=0.0 (uncoupled case)
│   ├── ldos_t_0_00.svg        # LDOS plot for t=0.0 (SVG format)
│   ├── ldos_t_0_05.jpg        # LDOS plot for t=0.05 (weak coupling)
│   ├── ldos_t_0_05.svg        # LDOS plot for t=0.05 (SVG format)
│   ├── ldos_t_2_00.jpg        # LDOS plot for t=2.0 (strong coupling)
│   ├── ldos_t_2_00.svg        # LDOS plot for t=2.0 (SVG format)
│   ├── ldos_t_5_00.jpg        # LDOS plot for t=5.0 (very strong coupling)
│   ├── ldos_t_5_00.svg        # LDOS plot for t=5.0 (SVG format)
│   ├── ldos_evolution.gif     # Animated evolution of LDOS (t=0 to t=5)
│   └── ...
├── LDOS_for...chains.pdf      # PDF paper where the simulations where presented
├── ldos.py                    # Main Python script for LDOS calculations
└── README.MD                  # This file
```

## 🧮 Theory Background

### ⚛️ SSH Model and Edge States

The system consists of two coupled SSH chains with edge states described by:

- **Right edge state**: $|\psi_R\rangle = c_R(e^{-x/d_+} - e^{-x/d_-})|\phi\rangle$
- **Left edge state**: $|\psi_L\rangle = c_L(e^{x/\xi_+} - e^{x/\xi_-})|\phi\rangle$

Where the decay parameters are given by:
- $d_{\pm}^{-1} = \frac{|A|}{2m_2} \left(1 \pm \sqrt{1 - \frac{4m_1m_2}{A^2}}\right)$
- $\xi_{\pm}^{-1} = \frac{|A'|}{2m_2'} \left(1 \pm \sqrt{1 - \frac{4m_1'm_2'}{A'^2}}\right)$

### 🌊 Green's Function Approach

The coupled system is analyzed using the Dyson equation with a tunneling Hamiltonian:

$$\hat{V} = t(|\psi_R\rangle\langle\psi_L| + |\psi_L\rangle\langle\psi_R|)$$

The resulting Green's function takes the form:

$$G(\omega) = \frac{a(x)\omega}{\omega^2 - b(x)}$$

Where:
- $a(x) = 2c_1^2[e^{-x/d_+} - e^{-x/d_-}]^2$
- $b(x) = 4t^2c_1^2c_2^2[e^{-x/d_+} - e^{-x/d_-}]^2[e^{x/\xi_+} - e^{x/\xi_-}]^2$

### 🔢 Normalization Constants

The normalization constants are calculated as:
- $c_1 = \frac{\sqrt{d_+ + d_-}}{|d_+ - d_-|}$
- $c_2 = \frac{\sqrt{\xi_+ + \xi_-}}{|\xi_+ - \xi_-|}$

## ⚙️ Code Functionality

### ✨ Main Features

1. **Parameter Calculation**: Computes all relevant physical parameters including decay lengths, normalization constants, and coupling terms.

2. **LDOS Computation**: Calculates the Local Density of States using the formula:
   $$\rho(\omega) = -\frac{1}{\pi} \text{Im}\{G(\omega + i\eta)\}$$
   where $\eta$ is a small imaginary part for analytical continuation.

3. **Visualization**: Generates multiple types of plots:
   - Main LDOS plot with custom tick labels at ±b, ±0.5, and 0
   - Special treatment for t=0 case: $\rho(x,\omega) = 2f(x,x)\delta(\omega)$
   - Regular plots for t>0: $\rho(x,\omega) = f(x,x)[\delta(\omega-b)+\delta(\omega+b)]$
   - Animated evolution showing smooth transition from t=0 to t=5

4. **Parameter Studies**: Systematically varies the coupling parameter $t$ to study the evolution of the LDOS.

### 🎛️ Key Parameters

- **Physical Parameters**:
  - $w = 1.0$: Hopping amplitude
  - $a_{val} = 1.0$: Lattice parameter
  - $v = 0.8$: Intra-cell hopping
  - $x_{eval} = 0.5$: Evaluation position
  - $t$: Tunneling coupling strength (varied: 0.0, 0.05, 2.0, 5.0)

- **Numerical Parameters**:
  - $n = 4.0$: Frequency range $[-n, n]$
  - $m = 8000$: Number of discretization points
  - $\eta = 2\Delta\omega$: Imaginary part for analytical continuation

### 📤 Output

The code generates:

1. **Console Output**: Detailed parameter values and intermediate calculations
2. **Main LDOS Plot**: Shows the characteristic form with labeled tick marks at critical frequencies
3. **Special t=0 Plot**: Highlights the delta function behavior at ω=0 for uncoupled chains
4. **Coupling Evolution Plots**: Individual plots for different coupling strengths showing the transition from isolated peaks to hybridized states
5. **Animation**: Smooth GIF evolution from t=0 to t=5 showing the complete coupling dynamics
6. **Multiple Formats**: All plots saved in both SVG (vector) and JPG (raster) formats

## 🚀 Usage

```bash
python ldos.py
```

### 📦 Requirements

- Python 3.x
- NumPy
- Matplotlib
- IPython (for display utilities)

### 💾 Installation

```bash
pip install numpy matplotlib ipython
```

## 📊 Results Interpretation

The LDOS plots reveal several key physics regimes:

### 🔍 Special Cases

- **$t = 0$**: Uncoupled system showing a single delta function at ω=0, representing the isolated edge state
  - Title: $\rho(x,\omega) = 2f(x,x)\delta(\omega)$
  - No |b| annotation (since b=0)

### 🌊 Coupling Evolution

- **$t = 0.05$**: Weak coupling regime with slight peak splitting
  - Peaks appear at ±b with finite width due to analytical continuation
  - Shows beginning of level repulsion

- **$t = 2.0$**: Intermediate coupling with clear hybridization
  - Well-separated peaks at ±b frequencies
  - Significant spectral weight redistribution

- **$t = 5.0$**: Strong coupling regime
  - Large peak separation indicating strong hybridization
  - Edge states significantly modified by inter-chain interactions

All plots (except t=0) show:
- Custom x-axis ticks at positions: -b, -0.5, 0, 0.5, b
- Corresponding labels: $-b$, $-0.5$, $0$, $0.5$, $b$
- |b| value annotation in the lower-right corner

## 🎬 Animation Features

The GIF animation demonstrates:
- Smooth evolution from t=0 to t=3
- Real-time |b| value updates
- Dynamic peak splitting and evolution
- Transition from delta function to hybridized doublet

## 🔧 Applications

This code is useful for:
- Understanding topological edge state physics
- Visualizing quantum tunneling effects between topological modes
- Designing quantum devices with controlled edge state interactions
- Educational demonstrations of Green's function methods and analytical continuation
- Research into coupled topological systems and level repulsion phenomena

## 🎓 Academic Context

This computational implementation was developed as part of the course Advanced Mathematical Physics as a final research project at the **National University of Colombia - Department of Physics**. The work investigates the local density of states (LDOS) for coupled topological Su-Schrieffer-Heeger (SSH) chains using Green's function methods, providing both analytical and numerical insights into edge state physics.

### 📊 Research Contributions

The study presents a comprehensive analysis of:

- **Single SSH Chain LDOS**: Analytical derivation confirming the existence of zero-energy edge states localized at boundaries, manifesting as Dirac delta peaks at ω = 0
- **Coupled System Dynamics**: Investigation of inter-chain tunneling effects that destroy zero-energy modes and create symmetric hybridized states
- **Symmetry Preservation**: Demonstration that chiral symmetry is maintained throughout the coupling evolution, ensuring symmetric peak distribution around the origin
- **Phase Transitions**: Systematic exploration of the transition from well-localized topological edge states to delocalized bulk resonances as coupling strength increases

### 🔬 Key Findings

The research reveals several important physical phenomena:

1. **Level Repulsion**: Weak coupling (t = 0.05) splits the zero-energy peak into two symmetric peaks due to protected level repulsion from chiral symmetry
2. **Progressive Hybridization**: Intermediate coupling strengths show peaks shifting away from ω = 0 with decreasing amplitudes, indicating partial delocalization
3. **Bulk Merger**: Strong coupling (t = 5) causes edge states to merge with the bulk continuum, creating broad, low-amplitude resonances
4. **Topological Robustness**: The evolution demonstrates both the robustness of edge states against weak perturbations and their eventual dissolution under strong interactions

This work contributes to the theoretical understanding of topological materials and provides valuable insights for designing quantum heterostructures and devices that exploit edge state interactions.

### 💡 Theoretical Context

The theoretical framework implemented in this code draws primarily from the SSH model [1] for understanding topological edge states, extended to coupled systems using Green's function methods. The LDOS calculation approach follows established quantum mechanical principles [5,6], while the physical motivation comes from both condensed matter [3] and photonic [4] realizations of topological systems. The study of coupled edge modes is particularly relevant for understanding Majorana fermions [2] and other exotic topological phases.

## 📚 References

### 🏗️ Foundational SSH Model and Topological Physics

1. **Su, W. P., Schrieffer, J. R., & Heeger, A. J.** (1979). Solitons in Polyacetylene. *Physical Review Letters*, 42(25), 1698-1701. [DOI: 10.1103/PhysRevLett.42.1698](https://link.aps.org/doi/10.1103/PhysRevLett.42.1698)
   - *The original SSH model paper that introduced the theoretical framework for studying topological solitons in one-dimensional systems, forming the foundation for this work.*

2. **Kitaev, A. Yu.** (2001). Unpaired Majorana fermions in quantum wires. *Physics-Uspekhi*, 44(10S), 131. [DOI: 10.1070/1063-7869/44/10S/S29](https://dx.doi.org/10.1070/1063-7869/44/10S/S29)
   - *Seminal work on topological superconductors and Majorana edge states, providing theoretical context for understanding topological edge modes in coupled systems.*

### 🧪 Experimental Realizations and Applications

3. **von Klitzing, K., Dorda, G., & Pepper, M.** (1980). New Method for High-Accuracy Determination of the Fine-Structure Constant Based on Quantized Hall Resistance. *Physical Review Letters*, 45(6), 494-497. [DOI: 10.1103/PhysRevLett.45.494](https://link.aps.org/doi/10.1103/PhysRevLett.45.494)
   - *Discovery of the quantum Hall effect, demonstrating topological quantum phases in condensed matter systems and establishing experimental precedent for topological edge states.*

4. **Lu, L., Joannopoulos, J. D., & Soljačić, M.** (2014). Topological photonics. *Nature Photonics*, 8(11), 821-829. [DOI: 10.1038/nphoton.2014.248](https://doi.org/10.1038/nphoton.2014.248)
   - *Comprehensive review of topological concepts in photonic systems, relevant for understanding optical implementations of SSH-like models and coupled topological structures.*

### 📐 Local Density of States Theory and Methods

5. **Mignuzzi, S., Vezzoli, S., Horsley, S. A. R., Barnes, W. L., Maier, S. A., & Sapienza, R.** (2019). Nanoscale Design of the Local Density of Optical States. *Nano Letters*, 19(3), 1613-1617. [DOI: 10.1021/acs.nanolett.8b04515](https://doi.org/10.1021/acs.nanolett.8b04515)
   - *Modern treatment of LDOS calculations and design principles, providing methodological background for the Green's function approach used in this implementation.*

6. **Iffländer, T., Rolf-Pissarczyk, S., Winking, L., Ulbrich, R. G., Al-Zubi, A., Blügel, S., & Wenderoth, M.** (2015). Local Density of States at Metal-Semiconductor Interfaces: An Atomic Scale Study. *Physical Review Letters*, 114(14), 146804. [DOI: 10.1103/PhysRevLett.114.146804](https://link.aps.org/doi/10.1103/PhysRevLett.114.146804)
   - *Experimental and theoretical study of LDOS at interfaces, demonstrating the importance of local electronic structure analysis in understanding coupled quantum systems.*

## 📄 License

Academic Project - Universidad Nacional de Colombia  
Advanced Mathematical Physics - 2025-I
