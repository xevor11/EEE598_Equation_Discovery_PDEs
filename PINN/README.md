# EQDiscovery — Colab-Compatible PINN-SR PDE Discovery

## Overview

This project is a modified and modernized implementation of the **EQDiscovery** source code associated with the paper:

> Zhao Chen, Yang Liu, and Hao Sun, “Physics-informed learning of governing equations from scarce data,” arXiv preprint arXiv:2005.03448, 2020

The original repository demonstrates how **physics-informed deep learning** can be used to discover governing partial differential equations (PDEs) from scarce and noisy spatiotemporal data. The method combines:

- Physics-Informed Neural Networks (PINNs)
- Automatic differentiation
- Sparse regression
- STRidge coefficient selection
- Symbolic PDE discovery

The goal is to approximate the unknown solution field \(u(x,t)\), compute derivatives such as \(u_t\), \(u_x\), and \(u_{xx}\), and identify the sparse PDE structure that best explains the observed data

This modified version focuses on running the Burgers equation discovery experiment in a modern Google Colab / Python 3.12 environment

---

## Original Repository and Paper

This work references and modifies the original implementation from:

```text
EQDiscovery Repository:
https://github.com/isds-neu/EQDiscovery

Paper:
Chen, Zhao, Yang Liu, and Hao Sun.
"Physics-informed learning of governing equations from scarce data."
arXiv preprint arXiv:2005.03448 (2020).
```

---

## Target PDE: Burgers Equation

The main test case used in this modified implementation is Burgers’ equation:

```text
u_t = -u*u_x + 0.1*u_xx
```

The true sparse PDE structure contains two dominant terms:

| Term | Meaning | True Coefficient |
|---|---|---|
| `u*u_x` | nonlinear advection | `-1.0` |
| `u_xx` | diffusion | `0.1` |

The purpose of the model is to recover these terms and coefficients directly from the training data

---

## Original Source Code Environment

The original EQDiscovery code was written for an older software stack:

```text
Python 3.7
TensorFlow-GPU 1.15
CUDA 10
cuDNN 7.x
pyDOE
NumPy
SciPy
Matplotlib
tqdm
Spyder / Anaconda
```

The original code relies heavily on TensorFlow 1.x APIs such as:

```python
tf.Session()
tf.placeholder()
tf.contrib.opt.ScipyOptimizerInterface()
tf.truncated_normal()
tf.set_random_seed()
```

These APIs either changed or were removed in modern TensorFlow environments. Therefore, several modifications were required to run the code in Google Colab.

---

## Modified Environment

The modified version was tested in Google Colab using:

```text
Python 3.12
TensorFlow 2.x with tensorflow.compat.v1
NVIDIA A100 GPU
NumPy
SciPy
Matplotlib
SMT
tqdm
```

---

## Installation

Run the following commands in Google Colab before executing the notebook:

```python
!pip install tensorflow
!pip install scipy
!pip install matplotlib
!pip install tqdm
!pip install smt
```

The `smt` package is used as a replacement for `pyDOE` / `pyDOE2`.

---

## Required Dataset

The notebook expects the Burgers dataset file:

```text
burgers.mat
```

Place this file in the current working directory of the Colab notebook

You can verify that it is available by running:

```python
!ls
```

Expected output should include:

```text
burgers.mat
sample_data
```

---

## Modifications

### TensorFlow Compatibility Mode

The original code used:

```python
import tensorflow as tf
```

In the modified code, this was changed to:

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

This allows TensorFlow 1.x graph-style code to run inside a modern TensorFlow 2.x environment

---

### Random Seed Fix

The original TensorFlow 1.x code used:

```python
tf.set_random_seed(1234)
```

If using `tensorflow.compat.v1`, this remains correct:

```python
np.random.seed(1234)
tf.set_random_seed(1234)
```

### Replacing `pyDOE` / `pyDOE2`

The original code used Latin Hypercube Sampling through:

```python
from pyDOE import lhs
```

or:

```python
from pyDOE2 import lhs
```

However, `pyDOE2` imports the deprecated `imp` module, which was removed in Python 3.12. Therefore, the modified code replaces it with the `SMT` package:

```python
from smt.sampling_methods import LHS
```

A wrapper function was added so the old code can still call `lhs(dim, samples)`:

```python
def lhs(dim, samples):
    xlimits = np.array([[0.0, 1.0]] * dim)
    sampling = LHS(xlimits=xlimits)
    return sampling(samples)
```

This preserves the original call style:

```python
X_f_train = lb + (ub-lb)*lhs(2, N_f)
```

---

### Plotting

The original code used:

```python
ax = fig.gca(projection='3d')
```

Modern Matplotlib no longer supports this usage. It was replaced with:

```python
ax = fig.add_subplot(111, projection='3d')
```

This change was applied to both the model prediction surface and the ground-truth surface plots

---

### Removing `tf.contrib`

The original code used:

```python
tf.contrib.opt.ScipyOptimizerInterface
```

This API was removed from TensorFlow and is not available even under `tensorflow.compat.v1`

Because of this, the original L-BFGS-B optimizer was disabled in the Colab-compatible version. The modified workflow uses:

```text
Adam optimizer
+
STRidge sparse regression
```

instead of:

```text
L-BFGS-B pretraining
+
Adam
+
L-BFGS-B
+
STRidge
```

This is the largest methodological change from the original implementation

---

### Adama Optimizer Modification

The Adam optimizer is used to train the neural network solution approximation

The modified version uses:

```python
self.optimizer_Adam = tf.train.AdamOptimizer(
    learning_rate=5e-4,
    beta1=0.99,
    beta2=0.9,
    epsilon=1e-8
)
```

A lower learning rate was used to reduce oscillatory spikes in the predicted solution surface

---

### STRidge Assignment

The original style assignment:

```python
self.lambda1 = tf.assign(self.lambda1, tf.convert_to_tensor(lambda2, dtype=tf.float32))
```

does not immediately update the TensorFlow variable in graph mode

In TensorFlow 1.x graph execution, an assignment operation must be explicitly executed using the session:

```python
assign_op = tf.assign(
    self.lambda1,
    tf.convert_to_tensor(lambda2, dtype=tf.float32)
)

self.sess.run(assign_op)
```

This fix is essential. Without `self.sess.run(assign_op)`, the discovered PDE coefficients remain unchanged

---

### STRidge Initialization

The original STRidge implementation initialized coefficients from the current TensorFlow variable:

```python
w = self.sess.run(self.lambda1)/Mreg
```

Since `lambda1` starts at zero, STRidge could remain stuck at zero

The modified version initializes STRidge using ridge regression:

```python
if lam != 0:
    w = np.linalg.lstsq(
        X.T.dot(X) + lam * np.eye(d),
        X.T.dot(y),
        rcond=None
    )[0]
else:
    w = np.linalg.lstsq(X, y, rcond=None)[0]
```

A similar initialization was also used in `TrainSTRidge()` for `w_best`.

---

### NumPy Compatibility Fix

The original STRidge code contained:

```python
if biginds != []:
    w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]
```

This can fail with modern NumPy because `biginds` is a NumPy array

It was replaced with:

```python
if len(biginds) > 0:
    w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]
```

---

### Reduced PDE Library for Burgers Equation

The original Burgers experiment used a large 16-term candidate library:

```python
[
    '1',
    'u', 'u**2', 'u**3',
    'u_x', 'u*u_x', 'u**2*u_x', 'u**3*u_x',
    'u_xx', 'u*u_xx', 'u**2*u_xx', 'u**3*u_xx',
    'u_xxx', 'u*u_xxx', 'u**2*u_xxx', 'u**3*u_xxx'
]
```

This caused correlated terms and unstable sparse selection in the modified Colab setting

The library was reduced to Burgers-relevant terms:

```python
[
    '1',
    'u',
    'u**2',
    'u_x',
    'u*u_x',
    'u_xx'
]
```

The corresponding `Phi` matrix is:

```python
Phi = tf.concat([
    tf.constant(1, shape=[N_f, 1], dtype=tf.float32),
    u,
    u**2,
    u_x,
    u*u_x,
    u_xx
], 1)
```

This reduced library made sparse PDE recovery significantly more stable

---

### Lambda Dimension Change

Because the library was reduced from 16 terms to 6 terms, all coefficient arrays were updated

Original:

```python
self.lambda1 = tf.Variable(tf.zeros([16, 1], dtype=tf.float32), dtype=tf.float32, name='lambda')
lambda_history_Adam = np.zeros((16, 1))
lambda_normalized_history_STRidge = np.zeros((16, 1))
lambda1_true = np.zeros((16, 1))
lambda1_true[5] = -1
lambda1_true[8] = 0.1
```

Modified:

```python
self.lambda1 = tf.Variable(tf.zeros([6, 1], dtype=tf.float32), dtype=tf.float32, name='lambda')
lambda_history_Adam = np.zeros((6, 1))
lambda_normalized_history_STRidge = np.zeros((6, 1))

lambda1_true = np.zeros((6, 1))
lambda1_true[4] = -1.0
lambda1_true[5] = 0.1
```

The new index mapping is:

| Index | Term |
|---|---|
| 0 | `1` |
| 1 | `u` |
| 2 | `u**2` |
| 3 | `u_x` |
| 4 | `u*u_x` |
| 5 | `u_xx` |

---

### Project Structure

```text
results_YYYYMMDD_HHMMSS/
│
├── adam/
│   ├── 8.png
│   ├── 9.png
│   ├── 10.png
│   ├── 11.png
│   ├── 12.png
│   ├── 13.png
│   └── 14.png
│
├── stridge/
│   ├── 22.png
│   ├── 23.png
│   ├── 24.png
│   ├── 25.png
│   ├── 26.png
│   └── 27.png
│
└── general/
    ├── stdout.txt
    ├── Pred.mat
    ├── 28.png
    ├── 29.png
    └── 30.png
```

---

## Hyperparameters

The best-performing modified setup used approximately:

```python
learning_rate = 5e-4
lam = 1e-5
d_tol = 0.01 or 0.02
maxit = 100
STR_iters = 10
noise = 0.0
N_f = 160000
```

The STRidge settings are controlled in:

```python
def callTrainSTRidge(self):
    lam = 1e-5
    d_tol = 0.02
    maxit = 100
    STR_iters = 10
```

---

## How to Run

### Step 1: Install packages

```python
!pip install tensorflow
!pip install scipy
!pip install matplotlib
!pip install tqdm
!pip install smt
```

### Step 2: Upload dataset

Upload:

```text
burgers.mat
```

to the Colab working directory

### Step 3: Enable GPU

In Google Colab:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

## Visualization Description

### Ground Truth Surface

The ground-truth surface plot shows the exact Burgers solution from the dataset

Axes:

```text
x-axis: spatial coordinate x
y-axis: time t
z-axis: solution value u(x,t)
```

This plot shows the nonlinear wave profile and diffusion-smoothed shock-like behavior

---

The model result surface shows the PINN-predicted solution over the same space-time domain

A good prediction should visually match the ground truth in:

- wave location
- peak height
- shock/front shape
- decay behavior over time

Earlier runs produced oscillatory spikes near the steep front. Reducing the Adam learning rate from `1e-3` to `5e-4` improved stability

---

### 3. Lambda Values Plot

The lambda plot compares:

```text
the true coefficients
the identified coefficients
```

For the reduced 6-term library, the true vector is:

```python
lambda1_true = [0, 0, 0, 0, -1.0, 0.1]
```

The ideal plot should show two dominant nonzero entries:

```text
index 4 -> u*u_x
index 5 -> u_xx
```

---

## Example Result

One representative discovered equation was:

```text
u_t = 0.00099125881
      + 0.057980493*u
      - 0.18947923*u**2
      - 0.08837528*u_x
      - 0.65957695*u*u_x
      + 0.042938795*u_xx
```

The dominant recovered PDE terms are:

```text
u_t ≈ -0.65957695*u*u_x + 0.042938795*u_xx
```

The true Burgers equation is:

```text
u_t = -1.0*u*u_x + 0.1*u_xx
```

The modified framework successfully recovered the correct PDE structure, although the coefficient values remain approximate

---

## Notes on Interpretation

The current Colab-compatible implementation is not an exact reproduction of the original paper because the original L-BFGS-B optimization path using `tf.contrib` was removed

Therefore, differences in recovered coefficients are expected

The most important result is that the sparse regression identifies the correct dominant terms:

```text
u*u_x
u_xx
```

Coefficient accuracy can be improved by further tuning:

```text
Adam learning rate
STRidge lambda
STRidge tolerance increment
number of Adam iterations
noise level
collocation points
candidate library size
```

---

## Common Errors and Fixes

### Error: `No module named pyDOE`

Use `smt` instead:

```python
!pip install smt
```

and replace `lhs` with the wrapper shown above

---

### Error: `No module named imp`

This occurs because `pyDOE2` is incompatible with Python 3.12

Use:

```python
from smt.sampling_methods import LHS
```

---

### Error: `module 'tensorflow' has no attribute 'set_random_seed'`

Use TensorFlow compatibility mode:

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(1234)
```

---

### Error: `module 'tensorflow.compat.v1' has no attribute 'contrib'`

`tf.contrib` was removed

Comment out or remove:

```python
tf.contrib.opt.ScipyOptimizerInterface
```

and use the Adam + STRidge workflow.

---

### Error: `Input vector should be 1-D`

For cosine similarity, flatten vectors:

```python
cosine_similarity = 1 - distance.cosine(
    lambda1_true.flatten(),
    lambda1_value.flatten()
)
```

---

## References

```text
Chen, Zhao, Yang Liu, and Hao Sun.
"Physics-informed learning of governing equations from scarce data."
arXiv preprint arXiv:2005.03448 (2020).

Original EQDiscovery Repository:
https://github.com/isds-neu/EQDiscovery
```

