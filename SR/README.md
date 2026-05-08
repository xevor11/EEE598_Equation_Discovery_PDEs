# PDE Symbolic Regression Dataset Generation and Discovery

This notebook generates numerical datasets for nonlinear partial differential equations (PDEs) and applies symbolic regression to recover governing equations from data. The notebook focuses on two benchmark PDEs:

1. **Burgers' Equation**
2. **Kuramoto–Sivashinsky (KS) Equation**

The overall workflow is:

1. Define the PDE and numerical domain
2. Generate synthetic solution data using numerical solvers
3. Compute derivative-based features
4. Save the dataset for symbolic regression
5. Train a symbolic regression model
6. Convert and simplify the discovered symbolic expression
7. Compare true and predicted time derivatives
8. Visualize the generated PDE datasets

---

## 1. Project Overview

The objective of this notebook is to create PDE datasets suitable for symbolic regression. Symbolic regression attempts to discover an interpretable mathematical expression that maps input features such as `u`, `u_x`, `u_xx`, `u_xxxx`, and nonlinear terms like `u*u_x` to the target derivative `u_t`

Instead of training only a black-box predictive model, the goal is to recover equations in explicit symbolic form, such as:

```text
u_t = -u*u_x + 0.2*u_xx
```

or:

```text
u_t = -u*u_x - u_xx - u_xxxx
```

This makes the discovered model physically interpretable and directly comparable with the known governing PDE

---

## 2. PDEs Considered

### 2.1 Burgers' Equation

The Burgers' equation used in this notebook is:

```text
u_t = -u*u_x + nu*u_xx
```

where:

- `u(x,t)` is the solution field
- `u_t` is the time derivative
- `u_x` is the first spatial derivative
- `u_xx` is the second spatial derivative
- `nu` is the viscosity coefficient

In this notebook:

```text
nu = 0.2
```

Therefore, the expected PDE is:

```text
u_t = -u*u_x + 0.2*u_xx
```

This equation combines nonlinear advection through the term `-u*u_x` and diffusion through the term `0.2*u_xx`

---

### 2.2 Kuramoto–Sivashinsky Equation

The Kuramoto–Sivashinsky equation used in this notebook is:

```text
u_t = -u*u_x - u_xx - u_xxxx
```

where:

- `u(x,t)` is the solution field
- `u_t` is the time derivative
- `u_x` is the first spatial derivative
- `u_xx` is the second spatial derivative
- `u_xxxx` is the fourth spatial derivative
- `u*u_x` represents nonlinear advection

The KS equation is more complex than Burgers' equation because it includes a fourth-order derivative term. It is commonly used as a benchmark equation because it can produce unstable, oscillatory, and chaotic dynamics

---

## Numerical Dataset Generation

### Burgers' Dataset Generation

The Burgers' PDE is solved on the domain:

```text
x in [-8, 8]
t in [0, 10]
```

with:

```text
nx = 256
nt = 5000
nu = 0.2
```

The spatial and temporal grids are created using:

```python
x = np.linspace(-8, 8, nx)
t = np.linspace(0, 10, nt)
```

The initial condition is:

```python
u_data[0, :] = np.exp(-0.25 * x**2)
```

This creates a smooth Gaussian-shaped initial profile

The numerical method used for Burgers' equation is an explicit finite-difference method. The spatial derivatives are approximated using central differences:

```python
u_x[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
```

The time update is performed using forward Euler:

```python
u_data[n + 1, :] = u + dt * (-u * u_x + nu * u_xx)
```

Dirichlet boundary conditions are applied at the domain boundaries:

```python
u_data[n + 1, 0] = 0
u_data[n + 1, -1] = 0
```

---

### Kuramoto–Sivashinsky Dataset Generation

The KS equation is solved on the domain:

```text
x in [0, 32*pi]
t in [0, 50]
```

with:

```text
nx = 256
nt = 5000
```

The spatial grid is periodic:

```python
x = np.linspace(0, 32 * np.pi, nx, endpoint=False)
```

The initial condition is:

```python
u_data[0, :] = np.cos(x / 16) * (1 + np.sin(x / 16))
```

The KS equation is solved using a semi-implicit spectral Euler method. Spectral derivatives are computed using Fourier modes:

```python
k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
```

The linear part of the PDE is:

```text
-u_xx - u_xxxx
```

In Fourier space, this becomes:

```python
linear_operator = k**2 - k**4
```

The nonlinear term is computed using:

```python
nonlinear_hat = -0.5j * k * np.fft.fft(u**2)
```

This corresponds to:

```text
-u*u_x = -0.5 * d(u^2)/dx
```

The semi-implicit update is:

```python
u_hat = (u_hat + dt * nonlinear_hat) / (1 - dt * linear_operator)
```

---

## 4. Subsampling

After generating full numerical solution data, the notebook subsamples the solution before symbolic regression.

For both Burgers' and KS equations, the default subsampling parameters are:

```python
time_stride = 20
space_stride = 2
```

The subsampled data is created using:

```python
u_sub = u_data[::time_stride, ::space_stride]
x_sub = x[::space_stride]
t_sub = t[::time_stride]
```

The corresponding grid spacing is recomputed:

```python
dx_sub = x_sub[1] - x_sub[0]
dt_sub = t_sub[1] - t_sub[0]
```

---

## 5. Derivative Feature Construction

After subsampling, the notebook computes derivative features using `np.gradient`.

### Burgers' Features

For Burgers' equation, the symbolic regression dataset contains:

```text
u
u_x
u_xx
u_u_x
y
```

where:

```text
u_u_x = u*u_x
y = u_t
```

The derivatives are computed as:

```python
u_t = np.gradient(u_sub, dt_sub, axis=0)
u_x = np.gradient(u_sub, dx_sub, axis=1)
u_xx = np.gradient(u_x, dx_sub, axis=1)
u_u_x = u_sub * u_x
```

The final Burgers' dataset is saved as:

```text
datasets/burgers_dataset.csv
```

The expected relationship is:

```text
y = -u_u_x + 0.2*u_xx
```

---

### KS Features

For the KS equation, the symbolic regression dataset contains:

```text
u
u_x
u_xx
u_xxxx
u_u_x
y
```

where:

```text
u_u_x = u*u_x
y = u_t
```

The derivatives are computed as:

```python
u_t = np.gradient(u_sub, dt_sub, axis=0)
u_x = np.gradient(u_sub, dx_sub, axis=1)
u_xx = np.gradient(u_x, dx_sub, axis=1)
u_xxx = np.gradient(u_xx, dx_sub, axis=1)
u_xxxx = np.gradient(u_xxx, dx_sub, axis=1)
u_u_x = u_sub * u_x
```

The final KS dataset is saved as:

```text
datasets/ks_dataset.csv
```

The expected relationship is:

```text
y = -u_u_x - u_xx - u_xxxx
```

For KS, weak-dynamics rows are removed before training:

```python
df = df[
    (np.abs(df["y"]) > 1e-3) &
    (np.abs(df["u_xx"]) > 1e-4) &
    (np.abs(df["u_xxxx"]) > 1e-4)
].copy()
```

This filtering removes nearly flat regions where derivative values are too small and may not provide useful information for symbolic regression

---

## Symbolic Regression

The notebook uses genetic programming-based symbolic regression through `SymbolicRegressor`

The symbolic regression model searches for mathematical expressions that map the input features to the target derivative `u_t`

The supervised learning structure is:

```text
X = derivative and nonlinear candidate features
y = u_t
```

For Burgers' equation:

```python
X = [u, u_x, u_xx, u_u_x]
y = u_t
```

For KS equation:

```python
X = [u, u_x, u_xx, u_xxxx, u_u_x]
y = u_t
```

## 9. Formula Conversion and Simplification

The symbolic regressor returns expressions in terms of generic variables such as:

```text
X0, X1, X2, X3, X4
```

These are mapped back to physically meaningful PDE features

### KS PDE Dataset: `u_t` vs `u*u_x`

The plot titled:

```text
KS PDE Dataset: u_t vs u*u_x
```

shows the relationship between the nonlinear advection feature:

```text
u*u_x
```

and the target time derivative:

```text
u_t
```

The horizontal axis represents `u*u_x`, while the vertical axis represents `u_t`

For the KS dataset, the scatter plot is much more irregular and spread out compared to Burgers' equation. This is expected because the KS equation is not governed only by the nonlinear term `u*u_x`. The full equation also contains the second derivative term `u_xx` and the fourth derivative term `u_xxxx`:

```text
u_t = -u*u_x - u_xx - u_xxxx
```

Therefore, plotting only `u_t` against `u*u_x` does not show a simple one-dimensional relationship. The spread in the data reflects the influence of the additional derivative terms. The dense cluster near the origin indicates many regions where the local time derivative and nonlinear transport term are relatively small. The larger scattered branches and outer points correspond to stronger local dynamics in the KS solution

---

### KS PDE: True vs Predicted `u_t`

The plot titled:

```text
KS PDE: True vs Predicted u_t
```
compares the true target derivative with the symbolic-regression-predicted derivative
The horizontal axis represents true `u_t`, while the vertical axis represents predicted `u_t`
A good symbolic regression model should produce points close to the diagonal line:

```text
Predicted u_t = True u_t
```
In the generated plot, most points follow a strong diagonal trend. This indicates that the discovered symbolic expression captures the main structure of the KS time derivative. The dense diagonal region near the center shows that the model performs well on the majority of the dataset

Some deviations and curved off-diagonal structures are also visible. These deviations may come from:

1. Numerical differentiation error
2. Subsampling effects
3. Difficulty estimating the fourth derivative `u_xxxx`

---

### Burgers' PDE Dataset: `u_t` vs `u*u_x`

The plot titled:

```text
Burgers' PDE Dataset: u_t vs u*u_x
```

shows the relationship between `u*u_x` and `u_t`.

For Burgers' equation, the expected PDE is:

```text
u_t = -u*u_x + 0.2*u_xx
```

The scatter plot forms a smooth, structured pattern rather than a random cloud. This is because Burgers' equation has simpler dynamics than KS and contains only two main candidate terms:

```text
-u*u_x
0.2*u_xx
```

The elongated curved shape indicates that the nonlinear advection term `u*u_x` has a strong relationship with the time derivative `u_t`. However, the relationship is not perfectly linear because the diffusion term `0.2*u_xx` also contributes to `u_t`.

The plot shows values of `u*u_x` roughly in the range:

```text
-0.3 to 0.3
```

and values of `u_t` roughly in the range:

```text
-0.35 to 0.3
```

The smooth loops and bands reflect the evolution of the Gaussian initial condition over time as the solution spreads and changes under the combined effects of nonlinear transport and diffusion

---

### Burgers' Generated Dataset Plot

The generated Burgers' plot is visually similar to the original Burgers' dataset plot because the learned symbolic expression is evaluated on the same feature space and compared with the generated derivative behavior.

If the discovered equation is close to:

```text
u_t = -u_u_x + 0.2*u_xx
```

then the generated derivative structure should closely match the original dataset structure. Similarity between the original and generated plots indicates that symbolic regression successfully learned the dominant PDE terms

---

## Burgers' vs KS Results

Burgers' equation produces a cleaner and more structured dataset because the underlying PDE is simpler. The target derivative `u_t` is mainly controlled by nonlinear advection and diffusion:

```text
u_t = -u*u_x + 0.2*u_xx
```

The KS equation is more difficult because it contains nonlinear advection, second-order instability, and fourth-order dissipation:

```text
u_t = -u*u_x - u_xx - u_xxxx
```

As a result:

- Burgers' data forms smoother and more interpretable scatter structures
- KS data forms more chaotic and widely scattered structures
- KS requires a larger symbolic-regression search space
- KS predictions may show more deviation from the diagonal due to the difficulty of estimating high-order derivatives

---

## Requirements

The notebook uses the following Python libraries:

```text
numpy
pandas
matplotlib
scikit-learn
gplearn
sympy
os
```

---

## Directory Structure

The project uses the following directory structure:

```text
project/
│
├── datasets/
│   ├── burgers_dataset.csv
│   ├── burgers_dataset_generated.csv
│   ├── ks_dataset.csv
│   └── ks_dataset_generated.csv
│
├── output/
│   ├── burgers_dataset.png
│   ├── burgers_dataset_generated.png
│   ├── ks_dataset.png
│   └── ks_dataset_generated.png
│
└── notebook.ipynb
```

## References

### Genetic Equation Discovery Repository

This project references and builds upon ideas from:

**Angel Lagrange, Genetic Equation Discovery**

Repository:

- [https://github.com/AngelLagr/genetic-eq-discovery/tree/main](https://github.com/AngelLagr/genetic-eq-discovery/tree/main)
