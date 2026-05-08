# EEE 598 Project: Equation Discovery for Nonlinear Dynamical Systems

## Overview

This project investigates equation discovery methods for nonlinear dynamical systems governed by ordinary differential equations (ODEs) and partial differential equations (PDEs). The work compares multiple data-driven and physics-informed approaches for discovering governing equations directly from observed data

The project focuses on nonlinear systems where deriving analytical governing equations is difficult due to:

- chaotic behavior
- nonlinear interactions
- high-dimensional state spaces
- sensitivity to initial conditions
- sparse or noisy observations

The project includes experiments involving:

- Burgers’ equation
- Kuramoto–Sivashinsky equation
- nonlinear ODE systems

The repository contains:

- equation discovery implementations
- PINN-based PDE discovery
- sparse regression workflows
- symbolic equation generation
- PDE simulation datasets
- visualization tools
- diagnostic plots
- training and evaluation pipelines

---

# Motivation

Equation discovery aims to identify explicit mathematical expressions that govern the evolution of physical systems directly from data

A nonlinear dynamical system is commonly represented as:

$$
\dot{x}(t)=f(x(t))
$$

where:

- $x(t)$ is the system state
- $f(\cdot)$ represents the unknown governing dynamics

Traditional derivation of governing equations often requires:

- first-principles modeling
- domain expertise
- handcrafted assumptions
- complex analytical derivations

However, many real-world systems exhibit highly nonlinear behavior where analytical derivation becomes extremely difficult

Examples include:

- fluid dynamics
- climate systems
- power systems
- plasma dynamics
- biological systems
- turbulence
- reaction-diffusion systems
- chaotic systems

Equation discovery provides a scalable framework for extracting interpretable governing laws directly from observed data

---

# Project Objectives

The primary objectives of this project are:

1. Study nonlinear PDE and ODE systems
2. Compare multiple equation discovery methodologies
3. Analyze sparse regression approaches
4. Evaluate physics-informed neural network approaches
5. Investigate symbolic equation generation techniques
6. Compare interpretability versus predictive accuracy
7. Study sparse identification under noisy conditions
8. Evaluate generalization across unseen trajectories
9. Analyze optimization and refinement strategies
10. Recover governing equations from simulated datasets

---

# Methodologies Studied

The project studies three primary equation discovery methodologies

---

# 1. Physics-Informed Neural Networks (PINNs)

The PINN-based equation discovery framework combines:

- deep neural networks
- automatic differentiation
- sparse regression
- PDE residual minimization

The framework learns:

1. the solution field $u(x,t)$
2. the governing PDE coefficients

simultaneously.

The neural network approximates the spatio-temporal solution:

$$
u_\theta(x,t)
$$

Automatic differentiation is then used to compute derivatives such as:

- $u_t$
- $u_x$
- $u_{xx}$
- $u_{xxx}$

These derivatives are combined into a candidate PDE library

Sparse regression (STRidge) is then used to identify the active governing terms

The PINN framework combines:

- data loss
- physics loss
- sparse coefficient optimization

through alternating optimization

---

# 2. Symbolic Regression (SR)

The symbolic regression framework attempts to directly recover explicit mathematical expressions from observed data

The approach treats equation discovery as a symbolic search problem where candidate mathematical expressions are evolved and optimized using genetic programming techniques

The symbolic regression workflow consists of:

1. numerical dataset generation
2. derivative feature construction
3. symbolic expression generation
4. expression simplification
5. equation evaluation

For PDE systems, the symbolic regression datasets contain features such as:

- $u$
- $u_x$
- $u_{xx}$
- $u_{xxxx}$
- nonlinear interaction terms such as $u u_x$

The symbolic regression model searches for interpretable expressions that map these candidate features to the target derivative:

$$
u_t
$$

The project studies symbolic regression on:

- Burgers’ equation
- Kuramoto–Sivashinsky equation

using generated numerical datasets and genetic programming-based sparse symbolic search

---

# 3. PDE-FIND (Sparse Identification for PDEs)

PDE-FIND is a sparse regression framework for discovering governing PDEs from data

The methodology assumes the governing PDE can be represented as a sparse linear combination of candidate terms:

$$
u_t = \Theta(u,u_x,u_{xx},\ldots)\xi
$$

where:

- $\Theta$ is the candidate PDE library
- $\xi$ is a sparse coefficient vector

The workflow consists of:

1. generating numerical PDE solution data
2. computing derivatives using finite differences
3. constructing a candidate PDE library
4. applying sparse regression using STRidge
5. identifying active PDE terms

The project applies PDE-FIND to:

- Burgers’ equation
- Kuramoto–Sivashinsky equation

to recover sparse governing equations directly from simulated data

---

# Unified Equation Discovery Framework

Across all methodologies, the project follows a unified iterative framework:

1. Generate candidate equations
2. Convert equations into structured representations
3. Evaluate candidates using scoring functions
4. Optimize high-performing equations
5. Refine sparse structures
6. Iterate until convergence

Candidate equations are evaluated using:

- residual error
- prediction error
- sparsity
- equation complexity
- coefficient stability

A priority-based selection mechanism retains high-performing candidates while weaker candidates are discarded

---

# Mathematical Formulation

A nonlinear dynamical system is represented as:

$$
\dot{x}(t)=f(x(t),u(t))
$$

where:

- $x(t) \in \mathbb{R}^n$ is the system state
- $u(t)$ represents control inputs
- $f(\cdot)$ represents unknown governing dynamics

Equation discovery seeks an explicit expression:

$$
\dot{x}=F(x(t);\xi)
$$

where:

- $F$ is the governing equation
- $\xi$ are unknown coefficients

Sparse regression reformulates the system as:

$$
\dot{X}=\Theta(X)\Xi
$$

where:

- $\Theta(X)$ is a library of candidate basis functions
- $\Xi$ is a sparse coefficient matrix

For PDE systems:

$$
u_t = F(u,u_x,u_{xx},\ldots)
$$

where:

- $u(x,t)$ is the system state over space and time
- spatial derivatives capture physical interactions

---

# PDE Systems Studied

## Burgers’ Equation

$$
u_t = -u u_x + 0.1u_{xx}
$$

Properties:

- nonlinear advection
- diffusion
- shock formation

---

## Kuramoto–Sivashinsky Equation

$$
u_t = -u u_x - u_{xx} - u_{xxxx}
$$

Properties:

- chaotic dynamics
- instability-dissipation balance
- pattern formation

---

# Project Structure

```text
project/
│
├── README.md
│
├── PINN/
│   ├── Burgers/
│   ├── Kuramoto_Sivashinsky/
│   └── datasets/
│
├── SR/
│   ├── Burgers_SR/
│   ├── KS_SR/
│   ├── datasets/
│   └── outputs/
│
└── PDEFIND/
    ├── Burgers_PDEFIND/
    ├── KS_PDEFIND/
    ├── outputs/
    └── notebooks/
```

This structure separates the repository into:

- PINN-based PDE discovery
- symbolic regression experiments
- PDE-FIND sparse regression experiments

with independent datasets, outputs, and notebooks for each methodology

---

# Modified EQDiscovery PINN Implementation

This project includes a modified and modernized version of the original EQDiscovery repository

The modified implementation was adapted for:

- Google Colab
- Python 3.12
- TensorFlow compatibility mode
- modern NumPy
- modern Matplotlib

The modified implementation includes:

- TensorFlow 2.x compatibility fixes
- pyDOE replacement using SMT
- STRidge assignment fixes
- sparse library reduction
- plotting compatibility fixes
- organized result directories

The modified Burgers implementation successfully recovers dominant PDE terms:

$$
u_t \approx -u u_x + u_{xx}
$$

Detailed implementation notes are provided in the EQDiscovery-specific README

---

# Visualization Outputs

The project generates:

- solution surface plots
- contour plots
- sparse coefficient plots
- PDE term histories
- loss convergence plots
- predicted trajectory plots
- coefficient evolution plots

---

# Surface Plot Interpretation

The generated surface plots visualize:

$$
u(x,t)
$$

across space and time.

Axes:

- x-axis: spatial coordinate
- y-axis: time
- z-axis: system state value

The plots demonstrate:

- nonlinear wave propagation
- diffusion smoothing
- shock formation
- chaotic structures
- reaction-diffusion behavior

Comparison between:

- ground truth surfaces
- learned surfaces

shows how accurately the framework reconstructs the underlying dynamics

---

# Sparse Coefficient Visualization

The sparse coefficient plots compare:

- true PDE coefficients
- discovered PDE coefficients

The plots indicate:

- whether the correct governing terms were identified
- sparsity quality
- coefficient accuracy
- regression stability

---

# Training Diagnostics

The project records:

- Adam loss history
- PDE residual loss
- sparse regression loss
- validation loss
- coefficient convergence

These diagnostics help analyze:

- optimization stability
- sparse selection behavior
- PDE identification quality

---

# Hardware Requirements

Experiments were tested on:

- NVIDIA A100 GPUs
- NVIDIA V100 GPUs
- GTX 1080Ti GPUs
- Intel i9 workstations

---

# Software Requirements

The project uses:

- Python
- TensorFlow
- NumPy
- SciPy
- Matplotlib
- tqdm
- SMT
- scikit-learn
- gplearn
- sympy

---

# Installation

Install dependencies:

```bash
pip install tensorflow
pip install numpy
pip install scipy
pip install pandas
pip install matplotlib
pip install tqdm
pip install smt
pip install scikit-learn
pip install gplearn
pip install sympy
pip install openpyxl
```

---

# Running Experiments

1. Install dependencies
2. Upload datasets
3. Enable GPU acceleration
4. Run the notebooks or Python scripts
5. Inspect generated visualizations and logs

---

# Expected Outputs

The framework produces:

- discovered equations
- coefficient vectors
- prediction trajectories
- PDE residual diagnostics
- visualization figures
- sparse regression outputs
- evaluation metrics

---

# Current Research Focus

The project currently focuses on:

- sparse PDE identification
- nonlinear PDE discovery
- physics-informed learning
- chaotic systems
- sparse regression
- symbolic equation generation
- interpretable machine learning for scientific systems

---

# References

## PINN-Based PDE Discovery

Chen, Zhao, Yang Liu, and Hao Sun.

*Physics-informed learning of governing equations from scarce data.*

arXiv preprint arXiv:2005.03448 (2020).

https://github.com/ZhaoChenCivilSciML/EQDiscovery-1

---

## PDE-FIND

Rudy, Samuel H., Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz.

*Data-driven discovery of partial differential equations.*

Science Advances 3, no. 4 (2017): e1602614.

https://github.com/zihanzhou2002/Data-Driven-PDE/tree/main

---

## Symbolic Regression

Angel Lagrange.

*Genetic Equation Discovery Repository.*

Repository:

https://github.com/AngelLagr/genetic-eq-discovery/tree/main

---

# Notes

This repository contains both:

1. original methodology reproductions
2. modified implementations adapted for modern environments

The project is intended for:

- nonlinear system identification
- sparse PDE discovery
- scientific machine learning
- interpretable AI for physical systems
- physics-informed modeling research

Detailed implementation notes for the modified EQDiscovery PINN implementation are provided separately
