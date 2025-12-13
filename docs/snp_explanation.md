# Reference: Spiking Neural P Systems (SN P Systems)
## Deterministic Implementation Guide

**Abstract**
> This document serves as a technical reference for Spiking Neural P Systems (SN P Systems), a class of distributed parallel computing models within Membrane Computing. It details the formal definition, neuron rules, and the matrix representation method used for computational simulation. 
> 
> **Note:** For the purpose of this reference implementation and verification, non-determinism is disregarded in favor of a fixed rule priority order.

---

## 1. Overview
Spiking Neural P Systems are inspired by the neurophysiological behavior of neurons sending electrical impulses (spikes) along axons to other neurons via synapses. Structurally, an SN P system is a **directed graph** where nodes represent neurons and edges represent synapses. The system operates under a global clock, synchronizing all neurons to work in parallel.



---

## 2. Formal Definition
An SN P system of degree $m \ge 1$ is formally defined as a tuple:

$$\Pi = (O, \sigma_1, ..., \sigma_m, syn, in, out)$$

**Where:**
* **$O = \{a\}$**: The singleton alphabet representing a spike.
* **$\sigma_1, ..., \sigma_m$**: Neurons of the form $\sigma_i = (n_i, \mathcal{R}_i)$, where:
    * $n_i \ge 0$ is the initial number of spikes in neuron $i$.
    * $\mathcal{R}_i$ is a finite set of rules.
* **$syn \subseteq \{1, ..., m\} \times \{1, ..., m\}$**: Represents synapses between neurons.
* **$in, out \in \{1, ..., m\}$**: Indicate the input and output neurons.

---

## 3. Neuron Rules and Semantics
Neurons process spikes using two specific types of rules. A rule is considered **applicable** if the current number of spikes in the neuron satisfies the rule's regular expression condition.

### 3.1 Firing Rules
A firing rule takes the form:
$$E/a^c \rightarrow a^p; d$$

* **$E$**: A regular expression over $\{a\}$. The rule applies only if the neuron's spike count $k \in L(E)$.
* **$c$**: The number of spikes consumed.
* **$p$**: The number of spikes produced.
* **$d$**: The delay (in time steps) before spikes are emitted.

When a firing rule with delay $d$ is applied, the neuron becomes **closed** (inactive). It cannot receive or send spikes for $d$ time steps. After the delay elapses, it becomes **open** and emits $p$ spikes to all connected neighbors.

### 3.2 Forgetting Rules
A forgetting rule takes the form:
$$a^s \rightarrow \lambda$$

If the neuron contains exactly $s$ spikes (and the regular expression $E$ is satisfied), $s$ spikes are removed, and no spikes are produced.

### 3.3 Deterministic Execution Assumption
Standard SN P systems are non-deterministic. **For this implementation guide, we disregard non-determinism.** We assume a total ordering of rules (e.g., by Rule ID). If multiple rules are applicable for a neuron, the rule with the **lowest ID** is selected automatically.

---

## 4. Matrix Representation
To enable efficient simulation on CPUs and GPUs, the system state is represented using linear algebra vectors and matrices.

### 4.1 System State Vectors
At any time step $k$:

| Vector | Symbol | Description |
| :--- | :--- | :--- |
| **Configuration** | $C^{(k)}$ | A $1 \times m$ vector where $c_i^{(k)}$ is the spike count in neuron $i$. |
| **Status** | $St^{(k)}$ | A $1 \times m$ vector: $1$ if neuron $i$ is open, $0$ if closed. |
| **Indicator** | $Iv^{(k)}$ | A $1 \times n$ vector ($n$ = total rules) where $iv_j^{(k)} = 1$ if rule $j$ fires. |
| **Spike Train** | $STv^{(k)}$ | A $1 \times m$ vector representing spikes entering from the environment. |

### 4.2 Spiking Transition Matrix ($M_{\Pi}$)
The Spiking Transition Matrix $M_{\Pi}$ is an $n \times m$ matrix (Rules $\times$ Neurons) that encodes the net spike change:

$$M_{\Pi} = [a_{ij}]_{n \times m} \quad \text{where } a_{ij} = 
\begin{cases} 
  -c & \text{if rule } i \text{ is in neuron } j \text{ and consumes } c \\
  p & \text{if rule } i \text{ produces } p \text{ spikes sent to neuron } j \\
  0 & \text{otherwise}
\end{cases}$$

### 4.3 The Transition Equation
The evolution of the system from time $k$ to $k+1$ is defined by:

$$C^{(k+1)} = C^{(k)} + St^{(k+1)} \odot (Iv^{(k)} \cdot M_{\Pi} + STv^{(k)})$$

* $\odot$ denotes the **Hadamard (element-wise) product**.
* $\cdot$ denotes standard **matrix multiplication**.
* $St^{(k+1)}$ acts as a mask, ensuring closed neurons do not receive spikes.

---

## 5. Algorithmic Implementation

```text
Algorithm: Compute Next Configuration (Deterministic)
Input: C(k), St(k), M_Pi, STv(