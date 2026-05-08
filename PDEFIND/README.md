# PDE-FIND Equation Discovery for Burgers' and Kuramoto–Sivashinsky PDEs

The main goal is to test whether PDE-FIND can recover the correct PDE terms and coefficients from simulated data for:

1. **Burgers' equation**
2. **Kuramoto–Sivashinsky (KS) equation**

The workflow is based on generating numerical PDE solution data, constructing a candidate library of possible PDE terms, solving a sparse regression problem, and comparing the discovered equation with the known ground-truth PDE

---

## Overview

Instead of assuming the governing equation is fully known, the notebooks generate solution data `u(x,t)` and then attempt to recover the PDE using a sparse regression framework.

The general PDE discovery problem is written as:

```text
u_t = Θ(u, u_x, u_xx, ..., u^p, u*u_x, ...) ξ
```

where:

- `u_t` is the target time derivative
- `Θ` is the candidate feature/library matrix
- `ξ` is the sparse coefficient vector
- Nonzero entries in `ξ` identify the active PDE terms
- STRidge is used to select a sparse set of terms

The desired output is an interpretable PDE such as:

```text
u_t = -u*u_x + ν*u_xx
```

or:

```text
u_t = -u*u_x - u_xx - u_xxxx
```

---

### Burgers' PDE-FIND Notebook

```text
EQ_Discovery_Burgers_PDEFIND.ipynb
```

This notebook focuses on the viscous Burgers' equation:

```text
u_t + u*u_x = ν*u_xx
```

or equivalently:

```text
u_t = -u*u_x + ν*u_xx
```

The notebook:

- Defines Burgers' equation and its finite-difference discretization.
- Implements a Crank–Nicholson-style solver for generating solution data.
- Builds a PDE-FIND regression library.
- Uses STRidge to discover the active PDE terms.
- Tests different viscosity values `ν`.
- Evaluates parameter error.
- Adds different noise levels to test robustness.
- Saves tables and plots for analysis.

---

### Kuramoto–Sivashinsky PDE-FIND Notebook

```text
EQ_Discovery_KS_PDEFIND.ipynb
```

This notebook focuses on the Kuramoto–Sivashinsky equation:

```text
u_t = -u*u_x - u_xx - u_xxxx
```

The notebook:

- Defines a numerical solver for the KS equation.
- Generates solution data for multiple initial conditions.
- Builds a PDE-FIND regression library with derivatives up to fourth order.
- Uses STRidge to identify the active KS terms.
- Compares discovered coefficients with ground-truth coefficients.
- Tests robustness under different noise levels.
- Saves result tables and error plots.

---

## Background: PDE-FIND

PDE-FIND is a sparse regression method for discovering governing PDEs from data. The method assumes that the true PDE can be represented as a sparse combination of candidate terms.

For example, for Burgers' equation, the candidate library may include terms such as:

```text
1
u
u^2
u^3
u_x
u*u_x
u^2*u_x
u_xx
u*u_xx
u^2*u_xx
...
```

The goal is to select only the physically relevant terms:

```text
u*u_x
u_xx
```

so that the discovered equation becomes:

```text
u_t = c1*u*u_x + c2*u_xx
```

For the correct Burgers' equation, the expected coefficients are:

```text
c1 = -1
c2 = ν
```

For the KS equation, the expected active terms are:

```text
u*u_x
u_xx
u_xxxx
```

with coefficients:

```text
-1, -1, -1
```

---

## Sparse Regression with STRidge

The notebooks use **STRidge**, or Sparse Threshold Ridge Regression, to solve the sparse regression problem.

The basic idea is:

1. Build the candidate library matrix `R`.
2. Compute the target derivative vector `Ut`.
3. Solve a ridge regression problem.
4. Threshold small coefficients.
5. Refit using the remaining active terms.
6. Repeat until a sparse PDE is found.

In the notebooks, the general function call is:

```python
w = TrainSTRidge(R, Ut, 10**-5, tolerance)
```

where:

- `R` is the candidate library matrix
- `Ut` is the vectorized time derivative
- `10**-5` is the ridge regularization parameter
- `tolerance` controls sparse coefficient thresholding
- `w` is the discovered coefficient vector

The discovered PDE is printed using:

```python
print_pde(w, rhs_des)
```

---

## Burgers' Equation Notebook

## Governing Equation

The Burgers' equation studied in the notebook is:

```text
u_t + u*u_x = ν*u_xx
```

or:

```text
u_t = -u*u_x + ν*u_xx
```

where:

- `u(x,t)` is the solution field
- `u_t` is the time derivative
- `u_x` is the first spatial derivative
- `u_xx` is the second spatial derivative
- `ν` is the viscosity coefficient

The expected PDE terms are:

```text
-u*u_x
ν*u_xx
```

---

## Numerical Setup

The Burgers' notebook uses parameters such as:

```python
N = 256
M = 200
T = 10
L = 10
D = 3
P = 3
```

where:

- `N` is the number of spatial intervals
- `M` is the number of time steps
- `T` is the final simulation time
- `L` controls the spatial domain length
- `D = 3` means the PDE library includes derivatives up to third order
- `P = 3` means polynomial powers up to `u^3` are included

The grid spacing is:

```python
dx = L / N
dt = T / M
```

The spatial grid is:

```python
x = np.linspace(-L/2, L/2, N+1)
```

The time grid is:

```python
t = np.linspace(0, T, M+1)
```

---

## Initial Conditions

The notebook defines multiple initial condition functions, including:

```python
def sinInitial(x):
    return np.sin(np.pi * x)

def GaussianInitial(x):
    x = np.array(x)
    return np.exp(-x**2)
```

The main Burgers' experiments use the Gaussian initial condition:

```text
u(x,0) = exp(-x^2)
```

This creates a smooth localized profile that evolves under the combined effects of nonlinear advection and diffusion

---

## Crank–Nicholson Solver

The notebook includes a Crank–Nicholson-style solver:

```python
CK_Burger(N, nu, T, M, L, initial)
```

This solver generates the solution matrix:

```text
u(x,t)
```

with shape approximately:

```text
(N+1, M+1)
```

The solver uses:

- periodic-style boundary handling,
- central finite differences for spatial terms,
- implicit/Crank–Nicholson-inspired discretization,
- nonlinear advection and viscous diffusion

The generated solution is then used as the input data for PDE-FIND

---

## PDE-FIND Library Construction

After generating `u`, the notebook constructs the PDE-FIND system using:

```python
Ut, R, rhs_des = build_linear_system(
    u,
    dt,
    dx,
    D=3,
    P=3,
    time_diff='FD',
    space_diff='FD'
)
```

where:

- `Ut` contains the approximated time derivative `u_t`
- `R` contains candidate PDE terms
- `rhs_des` contains readable descriptions of the candidate terms

The derivative method used here is finite differences:

```python
time_diff='FD'
space_diff='FD'
```

---

## STRidge Discovery for Burgers' Equation

The STRidge solver is called using:

```python
w = TrainSTRidge(R, Ut, 10**-5, 0.1)
```

The discovered PDE is printed with:

```python
print_pde(w, rhs_des)
```

For the clean case with `ν = 0.20`, the notebook recovers a PDE close to:

```text
u_t = -1.000038*u*u_x + 0.199749*u_xx
```

This is very close to the expected equation:

```text
u_t = -u*u_x + 0.2*u_xx
```

---

## Burgers' Parameter Error

The notebook evaluates the parameter error by comparing the discovered coefficients with the true coefficients.

For Burgers' equation, the relevant target coefficients are:

```text
coefficient of u*u_x = -1
coefficient of u_xx = ν
```

The notebook computes mean and standard deviation of the parameter error:

```python
err = abs(np.array([
    (-1 - w[5]) * 100,
    (nu - w[8]) * 100 / 0.1
]))
```

For the clean `ν = 0.20` case, the notebook reports a low parameter error, indicating that PDE-FIND successfully recovers the governing terms

---

## Burgers' Viscosity Sweep

The notebook tests multiple viscosity values:

```python
nu_values = [0.02, 0.2, 2.0]
```

For each value of `ν`, the notebook:

1. Generates Burgers' solution data
2. Builds the PDE-FIND candidate library
3. Applies STRidge
4. Stores the discovered PDE
5. Computes mean parameter error
6. Saves the results in a table
7. 
This experiment shows how PDE-FIND performance depends on the viscosity parameter. Low viscosity can lead to sharper gradients or shock-like behavior, making derivative estimation and PDE discovery more difficult

---

## Burgers' Noise Experiment

The notebook also tests robustness under different noise levels:

```python
lams = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1])
```

Noise is added as:

```python
un = u + lam*np.std(u)*np.random.randn(u.shape[0], u.shape[1])
```

For each noise level `λ`, the notebook:
1. Adds noise to the clean solution data
2. Rebuilds the PDE-FIND system
3. Applies STRidge
4. Stores the discovered PDE
5. Computes parameter error

This experiment helps evaluate how sensitive PDE-FIND is to noisy observations

---

## 5.10 Burgers' Plots

The Burgers' notebook generates multiple plots, including:

```text
Burger3d_02
Burger2d_02
Burger3d_004.png
Burger2d_002004.png
Error_nu.png
Error_Burger_noise.png
```

The 3D plots show the evolution of the solution surface `u(x,t)` over space and time

The 2D snapshot plots show solution profiles at selected time steps, such as:

- initial condition,
- early time,
- middle time,
- final time

These plots help visualize how viscosity affects the Burgers' solution. Higher viscosity produces stronger smoothing, while lower viscosity allows sharper gradients to develop

---

## Kuramoto–Sivashinsky Notebook

## Governing Equation

The KS equation studied in the notebook is:

```text
u_t = -u*u_x - u_xx - u_xxxx
```

where:

- `u(x,t)` is the solution field
- `u_t` is the time derivative
- `u_x` is the first spatial derivative
- `u_xx` is the second spatial derivative
- `u_xxxx` is the fourth spatial derivative

The expected active PDE terms are:

```text
u*u_x
u_xx
u_xxxx
```

with coefficients:

```text
-1
-1
-1
```

---

## 6.3 KS Solver

The notebook defines the solver:

```python
KS_Solver(N, T, M, L, initial)
```

The solver computes:

```text
u_x
u_xx
u_xxxx
```

using centered finite-difference formulas with periodic behavior through `np.roll`

The update follows the KS equation:

```text
u_t = -u*u_x - u_xx - u_xxxx
```

The numerical update uses the computed spatial derivatives to advance the solution in time

---

## KS Initial Conditions

The notebook tests several initial conditions:

```python
def KSInitial1(x):
    return np.exp(-x**2)

def KSInitial2(x):
    return np.cos(2 * np.pi * x / L) + 0.25 * np.sin(4 * np.pi * x / L)

def KSInitial3(x):
    return np.sin(2 * np.pi * x / L) + 0.5 * np.sin(6 * np.pi * x / L)
```

The corresponding case names are:

```text
Gaussian
Mixed Fourier 1
Mixed Fourier 2
```

Testing multiple initial conditions helps evaluate whether PDE-FIND can recover the same governing equation from different solution trajectories

---

## KS PDE-FIND Library Construction

The PDE-FIND system is built using:

```python
Ut, R, rhs_des = build_linear_system(
    u_case_fit,
    dt_fit,
    dx,
    D=4,
    P=2,
    time_diff='FD',
    space_diff='FD'
)
```

The KS candidate library includes:

- polynomial powers of `u`,
- first spatial derivative terms,
- second spatial derivative terms,
- third spatial derivative terms,
- fourth spatial derivative terms,
- nonlinear combinations such as `u*u_x`.

Because the true KS equation contains `u_xxxx`, the derivative order must be at least:

```text
D = 4
```

---

## STRidge Discovery for KS Equation

The STRidge solver is applied to recover the sparse coefficient vector:

```python
w = TrainSTRidge(R, Ut, 10**-5, 0.5)
```

The expected discovered PDE should be close to:

```text
u_t = -u*u_x - u_xx - u_xxxx
```

The KS equation is harder to discover than Burgers' equation because:

- it includes a fourth-order derivative,
- derivative estimation is more sensitive,
- the dynamics are more complex,
- the solution can contain oscillatory behavior,
- multiple terms interact strongly

---

## KS Initial Condition Experiment

The notebook loops over multiple initial conditions:

```python
ks_cases = [
    ("Gaussian", KSInitial1),
    ("Mixed Fourier 1", KSInitial2),
    ("Mixed Fourier 2", KSInitial3)
]
```

For each initial condition, the notebook:
1. Simulates the KS equation
2. Builds the PDE-FIND library
3. Applies STRidge
4. Extracts discovered coefficients
5. Computes parameter error
6. Stores the discovered PDE

The results are saved in:

```text
KS.xlsx
```

The error plot is saved as:

```text
Error_KS.png
```

This experiment evaluates whether the recovered PDE is consistent across different initial profiles

---

## KS Noise Experiment

The KS notebook also tests robustness under additive noise:

```python
lams = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
```

Noise is added as:

```python
un = u + lam * np.std(u) * np.random.randn(u.shape[0], u.shape[1])
```

For each noise level, the notebook:
1. Adds noise to the solution matrix
2. Subsamples the time dimension using `skip_t`
3. Builds the PDE-FIND system
4. Applies STRidge
5. Stores the discovered PDE
6. Computes mean and standard deviation of parameter error

The results are saved in:

```text
KS.xlsx
```

The noise error plot is saved as:

```text
Error_KS_noise.png
```

Since the KS equation contains high-order derivatives, it is expected to be more sensitive to noise than Burgers' equation

---

## 6.9 KS Plots

The KS notebook generates plots such as:

```text
KS3d.png
KS2d.png
Error_KS.png
Error_KS_noise.png
```

The 3D plot shows the full numerical solution surface:

```text
u(x,t)
```

The 2D snapshot plot compares solution profiles at selected times:

- initial condition,
- early time,
- middle time,
- final time

The error plots show how the mean parameter error changes across:

- different initial conditions,
- different noise levels

---

## Requirements

The notebooks use the following Python libraries:

```text
numpy
scipy
pandas
matplotlib
IPython
openpyxl
```

The PDE-FIND helper functions are included directly inside the notebooks.

Install the required packages using:

```bash
pip install numpy scipy pandas matplotlib ipython openpyxl
```

If running in Jupyter:

```bash
pip install notebook
```

or:

```bash
pip install jupyterlab
```

---

## Methodology Summary

The overall methodology is:

1. **Simulate PDE data**

   Numerical solvers generate `u(x,t)` for Burgers' and KS equations.

2. **Compute derivatives**

   PDE-FIND computes `u_t` and spatial derivatives using finite differences.

3. **Build candidate library**

   A large library of possible terms is created using polynomial powers and spatial derivatives.

4. **Apply sparse regression**

   STRidge selects a small number of active PDE terms.

5. **Recover symbolic PDE**

   Nonzero coefficients are converted into a readable equation.

6. **Compare with ground truth**

   Discovered terms and coefficients are compared with known PDE coefficients.

7. **Analyze robustness**

   Experiments are repeated across different parameters, initial conditions, and noise levels.

---

## 11. Interpretation of Results

### Burgers' Equation

For the clean Burgers' case, PDE-FIND successfully identifies the dominant physical terms:

```text
u*u_x
u_xx
```

The expected result is:

```text
u_t = -u*u_x + ν*u_xx
```

When `ν = 0.20`, the discovered coefficients are close to:

```text
-1
0.2
```

This shows that the method works well when the solution is smooth and the derivative estimates are reliable. However, when viscosity is very low, the solution can develop sharp gradients which make finite-difference derivative estimates less accurate, which can reduce the reliability of PDE-FIND

---

### Kuramoto–Sivashinsky Equation

For the KS equation, PDE-FIND attempts to identify:

```text
u*u_x
u_xx
u_xxxx
```

with coefficients:

```text
-1
-1
-1
```

This problem is more challenging because the fourth derivative `u_xxxx` is sensitive to numerical error and noise. Even small perturbations in the data can become amplified when high-order derivatives are estimated, hence a larger number of time steps is required

---

## Numerical Accuracy

The accuracy of PDE-FIND depends heavily on derivative quality

1. **Grid resolution**
   Higher spatial and temporal resolution improves derivative estimates

2. **Time step size**
   Smaller time steps can improve stability and accuracy

3. **Noise level**
   Noise can severely affect derivative estimation, especially for high-order derivatives

4. **Derivative method**
   The notebooks primarily use finite differences. Polynomial differentiation or smoothing may be better for noisy data

5. **Library size**
   If the candidate library is too large, sparse regression may select incorrect terms

6. **Threshold tolerance**
   STRidge tolerance controls sparsity. If the threshold is too high, true terms may be removed. If it is too low, extra false terms may remain

7. **Equation complexity**
   Burgers' equation is easier to discover than KS because it has fewer terms and lower derivative order

---

## Project Structure

```text
pdefind-equation-discovery/
│
├── notebooks/
│   ├── EQ_Discovery_Burgers_PDEFIND.ipynb
│   └── EQ_Discovery_KS_PDEFIND.ipynb
│
├── outputs/
│   ├── burgers/
│   │   ├── Burger.xlsx
│   │   ├── Burger3d_02.png
│   │   ├── Burger2d_02.png
│   │   ├── Error_nu.png
│   │   └── Error_Burger_noise.png
│   │
│   └── ks/
│       ├── KS.xlsx
│       ├── KS3d.png
│       ├── KS2d.png
│       ├── Error_KS.png
│       └── Error_KS_noise.png
│
├── README.md
└── requirements.txt
```

---

## References

### PDE-FIND

```text
Rudy, Samuel H., Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz.
"Data-driven discovery of partial differential equations."
Science Advances 3, no. 4 (2017): e1602614.
```
