# Rotation-native clustering

### What the e-means dissection means for the Topological-Rotation, HELICON and QUADRANT/QARE frameworks — and the one bridge none of them wrote down

> **Inputs.** (1) The companion analysis `README.md` in this folder: e-means (Bessa et al., arXiv:2611.09501) rewritten as preconditioned gradient descent, its five failure modes, and the two-axis grid *which sufficient statistics × how you forget*. (2) Three framework documents by the same author: *Topological Rotation* (Weyl–CORDIC–semimetal, June 2026), *HELICON* (curvature-ladder option engine), *QUADRANT/QARE* (reversible CORDIC ladder for amplitude estimation). (3) Two new experiment files, `rotation_experiments.py` → `rotation_results.json`, reproducible with `python3 -I rotation_experiments.py`.
>
> **Method.** Every claim about the frameworks is tied to a section of the framework text. Every numerical claim is either from the e-means paper (cited by table) or from a script in this folder. Where a framework claim is not supported by the clustering evidence, that is said plainly and a falsifiable replacement is offered.

---

## 0. Verdict

The three frameworks share one thesis: *rotation, not multiplication, is the atomic operation; determinism outranks accuracy when the clock is not negotiable; a pipeline should attest its own correctness with a conservation law.* The e-means dissection lands on that thesis from the opposite side, and the two meet at a single object:

**The Sherman–Morrison rank-1 inverse update and the Givens rotation are the same mathematical operation in two number systems.** SM maintains $\Sigma^{-1}$ by subtraction and division — a multiplier-and-divider object that needs floating-point *exponent range* (it overflows float16 at step 0, Exp A). The square-root form maintains the Cholesky factor $R$ with $R^\top R = \Sigma$ by a sweep of Givens rotations — one CORDIC vectoring step and one rotation step per entry, needing only *mantissa*, staying positive-semidefinite by construction, and surviving float16 at 1.5–3.7% error where SM cannot start (Exp A). On fixed-point Q-format fabric with no exponent, SM is not hostable and the Givens factor is the native object.

That single fact reorganizes the clustering grid from the companion analysis:

```
                   forget: never       by epoch (batch)       constant rate         adaptive rate
count + sum        MacQueen/Sculley    Lloyd                  e-means               (gap)
                   ─── multiply-native row: GEMM assignment, S/N ratio, floating point ───
count+sum+scatter  —                   Mahalanobis k-means    SM precision w/ λ     SM w/ adaptive λ
                   ─── rotation-native row: Givens factor update, triangular solve, fixed point ───
```

e-means lives in the *multiply-native* row: its hot loop is $XC^\top$ (the one workload where Rocket-Volder concedes 13% to matmul silicon, Topological Rotation §III.2), its state is a ratio $S/N$ with a $10^{-10}$ initialization that a Q24.40 register resolves to 6.8 bits (Exp B), its output depends on batch order and BF16 ties (companion §3.5), and it carries no invariant until its last step. The shape-aware, drift-aware, exact alternative — the second row — is the one these frameworks' hardware is built for, and it was never named as a workload in any of the three documents.

The rest of this document works that out framework by framework, lists what each gets right, what each needs to correct, and ends with a concrete rotation-native clustering design with costs on the Volder-1 VPE.

---

## 1. The two results that transfer

From the companion analysis, restated so this document stands alone.

**1.1 The optimizer identity.** With $m_j$ the batch mean of cluster $j$ and $N_j' = \gamma N_j + (1-\gamma)n_j$:

$$
c_j' = c_j - \big(N_j'\big)^{-1} I_d\,\nabla_{c_j} L_B .
$$

e-means is gradient descent with an isotropic, stale-count preconditioner. Lloyd is the $\gamma=0$ full-batch case (exact block-Newton). The Fisher information of $\mathcal N(\mu_j, \Sigma_j)$ w.r.t. $\mu_j$ is $n_j\Sigma_j^{-1}$, so Mahalanobis / natural-gradient hard-EM replaces $I_d$ by $\Sigma_j^{-1}$. **e-means is natural gradient under the assumption $\kappa(F_j) = 1$ for every cluster.** Hold that sentence for §4.2.

**1.2 The failure modes, with the numbers that matter here.** Low-$d$/small-$n$: the laws (fit at $d\ge96$, no $d$ term) clip to $\alpha_0=1$; 37% worse than Lloyd++ at $d=4,8$; given the same k-means++ seeds, Lloyd holds the optimum 6/6, e-means 3/6. Anisotropy: ARI 0.12 (e-means) vs 0.47 (Fisher metric) at $\kappa=100$. Drift: cosine-annealed EMA lags an SM tracker by 2.2×. Determinism: six batch orders → MSE $\{15.9,\dots,179\}$; Lloyd from the same init → bitwise identical. Revival: multiplicative BF16-$\varepsilon$ split gives *identical* children at a zero coordinate; count-based criterion never fires on merged clusters.

---

## 2. The bridge: Sherman–Morrison is the floating-point shadow of a Givens rotation

### 2.1 Two representations of one update

Forgetting-weighted scatter of a cluster, one sample $e = x - \mu$ at a time:

$$
\Sigma_t = \lambda\Sigma_{t-1} + e e^\top .
$$

*Inverse form (Sherman–Morrison).* With $P = \Sigma^{-1}$, $u = e/\sqrt\lambda$:

$$
P_t = \frac1\lambda\Big(P_{t-1} - \frac{P_{t-1}u\,u^\top P_{t-1}}{1 + u^\top P_{t-1}u}\Big).
$$

Cost $O(d^2)$: one matvec, one outer product, one scalar division. Entries of $P$ span $[1/\lambda_{\max}(\Sigma), 1/\lambda_{\min}(\Sigma)]$; the outer product spans the *square* of that. Dynamic range $\sim \kappa(\Sigma)^2\|e\|^4$. Symmetry and positive-definiteness are not enforced by the formula; rounding breaks both.

*Square-root form (QR-RLS / inverse-QR-RLS; Gentleman–Kung 1981, McWhirter 1983).* With $R$ upper-triangular, $R^\top R = \Sigma$:

$$
\begin{pmatrix}\sqrt\lambda\,R \\ e^\top\end{pmatrix}
\xrightarrow{\;G_1 G_2\cdots G_d\;}
\begin{pmatrix}R_t\\ 0\end{pmatrix},
$$

where each $G_i$ is a Givens rotation zeroing one entry of the appended row. Cost: $d$ rotation angles (CORDIC **vectoring**: given $(a,b)$, find $\theta$ with $b\to0$; this is the m=+1 vectoring mode of HELICON §1 and Rocket-Volder's `cordic.circ`), each applied to a row segment of length $d-i$ (CORDIC **rotation** mode). Total $d(d+1)/2$ micro-rotations. Entries of $R$ span $[\sqrt{\lambda_{\min}}, \sqrt{\lambda_{\max}}]$: dynamic range $\sim\sqrt{\kappa}\,\|e\|$. $R^\top R$ is PSD for *any* $R$, so the representation cannot leave the cone. Each $G_i$ is orthogonal — $\det G_i = 1$ — so the update is a product of **unimodular** maps, which is exactly the property QARE's reversible ladder requires (QARE §"Foundation: Unimodularity"). The SM update is not unimodular and not reversible.

The Mahalanobis distance needed for assignment is then a triangular solve: $\|R^{-\top}(x-c)\|^2$, i.e. $d$ back-substitution steps of multiply-accumulate — linear-mode CORDIC (m=0) in HELICON's taxonomy — with no inverse ever formed.

### 2.2 Measured (Exp A, `rotation_experiments.py`)

$d=16$, $\lambda=0.99$, 4,000 samples from a Gaussian with condition number $\kappa$, both trackers run in the stated precision, error measured against a float64 direct inverse every 200 steps.

| precision | $\kappa$ | Givens factor: rel. error (final / max) | Sherman–Morrison: rel. error (final / max) | SM min eigenvalue at end | SM checkpoints not PD (of 20) | SM diverged at step |
|---|---|---|---|---|---|---|
| float16 (10-bit mantissa, 5-bit exp) | 1 | 0.016 / 0.027 | — | — | — | **0** |
| float16 | 50 | 0.018 / 0.042 | — | — | — | **0** |
| float16 | 500 | 0.037 / 0.039 | — | — | — | **0** |
| bfloat16 (8-bit mantissa, 8-bit exp) | 1 | 0.45 / 0.53 | 0.31 / 0.35 | +0.0075 | 0 | — |
| bfloat16 | 50 | 0.35 / 0.46 | 1.90 / 1.91 | **−0.019** | **10** | — |
| bfloat16 | 500 | 0.58 / 0.69 | 6.2 / 6.4 | **−0.055** | **12** | **2575** |
| float32 (control) | 500 | $1.2\times10^{-6}$ / $2.0\times10^{-6}$ | $1.3\times10^{-6}$ / $3.7\times10^{-4}$ | $+1.9\times10^{-5}$ | 0 | — |

Reading it:

- **float16** has mantissa but no range. The Givens factor is fine. SM overflows in the first outer product ($P_0 = 100I$, $u\sim\sqrt{\kappa}$, $Pu\,u^\top P \sim 10^6 > 65504$). SM needs exponent; Givens needs mantissa.
- **bfloat16** has range but 8 mantissa bits. Neither is accurate at $\lambda = 0.99$ (the forgetting step $\sqrt{0.99}$ rounds to $0.99609$; a check against the effective $\lambda$ shows this *does not* explain the Givens error, so it is accumulated rounding in both — recorded in `A_bf16_lambda_bias`). But the two fail differently: Givens stays bounded (≤0.69) and PSD; SM reaches 6× error, goes indefinite at half the checkpoints, and diverges.
- **float32**: both fine, SM has a transient 300× spike in max error that Givens does not.

**Consequence for fixed point.** A Q-format datapath (HELICON §9, Q24.40) has no exponent at all. SM's requirement is not "more bits", it is *range*, which Q-format cannot supply without block-floating scaffolding. The Givens factor's requirement is mantissa, which Q24.47 supplies with 7 guard bits to spare (HELICON §9.2 — the guard-bit rule $\lceil\log_2 N\rceil + 1$ applies here verbatim, with $N$ the number of micro-rotations per Givens step). **On the frameworks' own hardware, the exact online precision tracker is the Givens-QR form, and it is built from the primitives the frameworks already have: vectoring, rotation, and linear-mode multiply.**

### 2.3 Cost on the Volder-1 VPE

Per sample, per cluster, $d$ Givens angles (vectoring, ~$N$ stages each) and $d(d+1)/2$ rotation applications (rotation mode, ~$N$ stages each, fully pipelined one per clock per CIU). Topological Rotation §III.1: 128 CIUs, 16 tiles × 8.

| $d$ | $k$ | micro-rotations / sample (all clusters) | CIU-cycles / sample at 1 rot/clock/CIU | throughput on 128 CIUs @ 1 GHz |
|---|---|---|---|---|
| 16 | 64 | $64\cdot136 = 8.7\times10^3$ | 68 | $\approx 15$ M samples/s |
| 96 | 1,024 | $1024\cdot4656 = 4.8\times10^6$ | $3.7\times10^4$ | $\approx 27$ k samples/s |
| 96 | 1,024, **update assigned cluster only** | $4656$ | 36 | $\approx 28$ M samples/s |
| 1,024 | 16 | $16\cdot 5.2\times10^5 = 8.4\times10^6$ | $6.6\times10^4$ | $\approx 15$ k samples/s |

The third row is the real design point: hard assignment means only the winning cluster's factor changes per sample, so the tracker is $O(d^2)$ per sample regardless of $k$, and 28 M samples/s at $d=96$ is a D1-class rate (QARE tier table: sub-millisecond, streaming). The assignment step itself — $k$ triangular solves of length $d$ — is $kd(d+1)/2$ linear-mode MACs per sample, which at $k=1024$, $d=96$ is $4.8\times10^6$ MACs: this is the GEMM-shaped part, and it is where a matmul co-processor (d-Matrix Corsair in the heterogeneous split, Topological Rotation §I.3) belongs. The rotation-native engine owns the *state*; the multiply-native engine owns the *assignment*. That split is new and it is clean.

Memory: $k(d + d(d+1)/2 + 1)$ words. At $k=1024$, $d=96$: 4.9 M words — fits Rocket-Volder's L2 "manifold cache" comfortably; at $k = 10^6$ (e-means' 1B-scale regime) it is 4.9 G words and does not fit anything, which is why e-means does not carry a second moment and should not at that scale. The rotation-native row is for $k \lesssim 10^4$, $d \lesssim 10^3$: the regime of every non-embedding clustering job, every streaming anomaly detector, every per-asset risk-factor cluster in a HELICON-class book.

---

## 3. HELICON: six laws, read against e-means

HELICON's value proposition is stated in its §13: bit-exact reproducibility, zero jitter, attestable correctness — *the same wrong number, everywhere, every time, provably*. e-means is the perfect foil: a modern algorithm that is fast, good, and none of those three things.

### 3.1 Law I (residue harvesting) → e-means has two free invariants it never uses

HELICON's point: the Greeks are algebraic recombinations of five primitives already resident in pipeline registers. The clustering analogue: e-means' state $(N_j, S_j)$ satisfies two exact conservation laws at every step, in exact arithmetic, with no extra computation:

$$
\sum_j N_j = \mathrm{EMA}_\gamma(B),\qquad \sum_j S_j = \mathrm{EMA}_\gamma\Big(\sum_{x\in B} x\Big).
$$

Mass and first moment of the whole state equal the EMA of batch size and batch sum, because the per-cluster EMAs are linear and the assignment is a partition. Exp C measures this: between revivals the first-moment residual is $8\times10^{-13}$ (float eps); mass is conserved to eps. **At each revival both laws are violated by a known amount** — mass by the discarded dead-cluster count $N_j^{\text{old}}$ (max 0.65 in Exp C), first moment by a jump of $\|S_j^{\text{old}}\| + \tfrac{\delta}{2}N_{j'}\|c_{j'}\|$ (24.8 in Exp C, one revival). So a residual monitor built exactly like HELICON §5 — two adders on the running sums, compared to two scalar EMAs — gives, at zero marginal cost: (i) a per-step proof that the accounting is intact, catching a dropped batch, an SEU in a count, a mis-ordered reduction; (ii) a *timestamped log of every revival*, which the paper's algorithm currently performs silently. e-means never taps these registers; HELICON's design discipline says it should.

### 3.2 Law II (sensitivity budget) → the $10^{-10}$ initialization is a floating-point idiom hostile to fixed point

HELICON §3.1: $\text{bits}(m_j) \ge \log_2(|\partial G/\partial m_j| / \tau) + \text{guard}$. For e-means the output is $c_j = S_j/N_j$ and the Jacobian is $\partial c_j/\partial S_j = 1/N_j$. The paper initializes $N_j = 10^{-10}$ so that $c_j = c_j^{(0)}$ with negligible weight. Exp B, in Q24.40:

| quantity | value |
|---|---|
| ULP $2^{-40}$ | $9.09\times10^{-13}$ |
| $N_0 = 10^{-10}$ in ULPs | 110 |
| resolution of $c_j$ recoverable from $S_j = N_0 c_j$ | $9.1\times10^{-3}$ → **6.8 bits** |
| bits needed on $S_j$ for $\tau = 10^{-3}, 10^{-6}, 10^{-9}$ on $c_j$ | 43, 53, 63 |

At initialization the centroid is representable to 6.8 bits in a 64-bit register; to hold it to $10^{-6}$ the sum register would need 53 fractional bits. This is the HELICON §3 asymmetry in its purest form: the register that "holds the answer" is not the register that needs the bits. The fix is trivial and the frameworks' own: initialize $N_0 = 1$, $S_0 = c^{(0)}$ (one sample's worth of weight, representable exactly), which changes the first-batch step from $\alpha$ to $\alpha n_j/(\gamma + (1-\gamma)n_j)$ — a difference that vanishes after one batch at the paper's own $\alpha_0$. More generally: **a ratio-of-EMAs representation has a time-varying Jacobian $1/N_j$ and therefore a time-varying bit requirement; the Givens factor's Jacobian is bounded by $\sqrt{\kappa}$ and the forgetting factor, fixed at synthesis.** HELICON's P3 ("uniform bit-width ends") is right, and e-means is an example of an algorithm whose bit requirement is not even stationary.

### 3.3 Law IV (residual monitor) → clustering has conservation laws too; e-means borrows one, once

HELICON's PDE residual $\Theta + (r-q)S\Delta + \tfrac12\sigma^2S^2\Gamma - rC \equiv 0$ is a per-result attestation. Lloyd has an equally exact per-iteration attestation: at a fixed point, every centroid equals the mean of its cell ($\nabla_{c_j} L = 0$ exactly), and between iterations $L_{t+1} \le L_t$ with equality only at a fixed point. e-means has *no* per-step attestation — its loss is not monotone, its centroids are never the mean of anything until the final last-epoch-mean step, which *is* a Lloyd M-step. So e-means borrows Lloyd's invariant exactly once, at the end, and reports the guarantee of that one step ("does not increase train MSE") as if it were the algorithm's. HELICON would call that a pipeline that runs fast but does not attest. The rotation-native tracker of §2 attests continuously: $R^\top R$ is PSD by construction, and $\sum_j N_j$ / $\sum_j S_j$ are conserved by construction.

### 3.4 Law V (Bachelier bypass) → a metric mux at the assignment rung

HELICON §6: an engine whose only path to $d_1$ runs through a logarithm has embedded an economic assumption into its routing. Clustering has the same structure. An engine whose only assignment rule is $\arg\min\|x - c\|^2$ has embedded *isotropy* into its routing — and the April-2020 analogue is any dataset with $\kappa(\Sigma_j)\gg1$ or unstandardized features, where Euclidean assignment produces deterministic garbage (ARI 0.10–0.12 at $\kappa=100$, companion Exp 2). The fix is also a multiplexer on one rung:

```
                ┌──────────────────────────────┐
   x, {c_j} ───►│  METRIC MUX                  │
                │  E: ‖x−c_j‖²           (GEMM) │
                │  M: ‖R_j^{-T}(x−c_j)‖² + log det (tri-solve) │
                │  L: −⟨x,c_j⟩_L         (Minkowski, m=−1)     │
                └──────────────┬───────────────┘
                               ▼
                      argmin_j  →  update (N_j,S_j) or (N_j,c_j,R_j)
```

E is e-means/Lloyd; M is the Fisher-metric row; L is the hyperboloid (§4.4). HELICON's design law restated for clustering: *any engine whose only path to an assignment runs through the Euclidean norm has embedded a geometric assumption into its routing, and assumptions in silicon do not update on a change of dataset.*

### 3.5 §13 (determinism outranks accuracy) → arithmetic determinism is not algorithmic determinism

This is the most important correction the e-means evidence forces, and it reaches all three frameworks. HELICON §13 and Topological Rotation P7 both argue that bit-exact fixed-point arithmetic delivers reproducibility — "the same inputs give the same bits on every die, forever" — and P7 extends this to reproducible training of 10T-parameter models by 2028.

e-means shows the gap. Run it on bit-exact arithmetic and the result still depends on (i) the batch permutation, (ii) the revival RNG draws, (iii) the tie-breaking order in argmin. Companion Exp 5: identical data, identical initial centroids, six permutations → six different converged states with a 165% spread in loss. **No arithmetic can fix this; it is a property of the algorithm's dependence on the order of its inputs.** Lloyd is order-independent (its M-step is a sum, and sums are permutation-invariant up to reduction order, which fixed point makes exact). The Givens tracker is order-*dependent* by design (it is a stream estimator), but *deterministically* so: same stream, same bits.

So the correct statement is: **reproducibility = deterministic arithmetic ∧ (order-invariant algorithm ∨ logged schedule transcript).** HELICON's pipeline satisfies the first disjunct (Black–Scholes is a function). e-means satisfies neither unless the permutation seed, revival RNG state, precision and reduction order are logged — which brings in QARE's direction record (§5.2). P7 should be restated to say so; as written it is falsified by any EMA-with-random-batches algorithm on bit-exact hardware.

### 3.6 Law VI (closed inversion) and the Hartigan correction

HELICON §7 observes that residue harvesting inverts the usual cost ordering: an order-4 Householder step is cheaper than an order-2 Newton step because the higher derivatives are already in registers. The clustering analogue is the Hartigan correction. Lederman et al. (ICML 2026) prove Lloyd's assignment has a self-influence bias of $1/|C_j|$ that makes essentially every partition a fixed point when $d \gtrsim n^3$; Hartigan's weights $n_j/(n_j-1)$ and $n_j/(n_j+1)$ remove it. On a multiply-native engine that is two extra divisions per candidate and is usually skipped. On a CORDIC engine the weights are one linear-mode vectoring step each (division is native, HELICON §1 table, $m=0$), and the counts $N_j$ are already in registers. **Hartigan becomes cheaper than Lloyd on rotation-native hardware**, for the same reason Householder-4 becomes cheaper than Newton-2: the correction terms are residue. e-means does not apply the correction and inherits the bias (companion §3.6).

---

## 4. Topological Rotation: the thesis is right about which clustering is native, and three of its constants need work

### 4.1 "Rotation, not multiplication" predicts the second row

The framework's central claim (§II.1) is that rotation algebra is the shared primitive across Weyl physics, CORDIC silicon and curved-manifold AI, and (§III.2, the "honest assessment") that Volder-1 concedes ~13% on flat-space matmul and wins 3–20× on curvature, iteration and integral geometry. Apply that honestly to clustering:

| clustering step | native operation | engine |
|---|---|---|
| e-means / Lloyd assignment $\arg\min_j\|x-c_j\|^2$ | GEMM $XC^\top$ | multiply-native (loses 13%) |
| e-means update $S_j/N_j$ | ratio, EMA | multiply-native, float range needed (Exp B) |
| Fisher-metric assignment $\|R_j^{-\top}(x-c_j)\|^2$ | triangular solve | linear-mode CORDIC, or DSP |
| **precision update** $R_j \leftarrow \mathrm{Givens}(\sqrt\lambda R_j, e)$ | $d$ vectoring + $d(d+1)/2$ rotations | **rotation-native** (Exp A) |
| Hartigan weights $N_j/(N_j\pm1)$ | division | linear-mode CORDIC |
| hyperboloid centroid $S/\sqrt{-\langle S,S\rangle_L}$ | Minkowski norm, $x^2-y^2$ invariant | hyperbolic CORDIC, $m=-1$ (Exp D) |
| revival split $c \pm \delta\,\sigma_j\odot r$ | needs $\sigma_j$ = diagonal of $R_j^\top R_j$ | free once $R_j$ exists |

The framework never lists clustering, quantization or codebook learning as a workload. It should: **the shape-aware half of clustering is rotation-native and the shape-blind half is multiply-native**, which is precisely the heterogeneous split the framework proposes for prefill/decode (§I.3). e-means is the decode-era algorithm for the matmul engine; the Givens tracker is the workload for the rotation engine; the metric mux of §3.4 is the dispatcher.

### 4.2 "$\kappa(F) = \varphi$" — not supported; here is the falsifiable version

§II.3 asserts a "master identity" $\kappa = 1/\varphi^2 = \exp(-2\log\varphi) = \exp(-2S_c)$ with the Fisher condition number at the generalization phase equal to the golden ratio. In clustering the Fisher information of cluster $j$ with respect to its mean is $n_j\Sigma_j^{-1}$ and its condition number is $\kappa(\Sigma_j)$ — a property of the data, not a universal constant. Companion Exp 2 sweeps it through 1, 10, 100 and the algorithms respond exactly as $\kappa$ dictates. There is no phase at which $\kappa(F)$ equals 1.618 for a reason; e-means *assumes* $\kappa(F)=1$ (§1.1) and that assumption is what breaks it at $\kappa = 100$.

Two further internal inconsistencies in the same section: (a) §I.2 states the CORDIC contraction constant as $k = 0.5$ with $|z_n| \le 2^{-n}$ (correct — one bit per iteration), while §II.3 states the per-iteration convergence rate as $1/\varphi^2 \approx 0.382$. Both cannot hold; the first is a theorem, the second has no derivation. (b) "Fermi arc count scales as $F(n)$" and "chirality separation $\Delta E \sim \varphi E_0$ (experimental observation in TaAs)" carry no citation and I could not find one; they should either cite a measurement or be removed.

What *would* be a falsifiable rotation-native statement about $\kappa(F)$: the Givens factor's entries span $\sqrt{\kappa}$ rather than $\kappa$ (§2.1), so **the bit-width required to track a cluster's Fisher metric to tolerance $\tau$ is $\tfrac12\log_2\kappa(\Sigma_j) + \log_2(1/\tau) + \lceil\log_2 d\rceil + 1$**, versus $\log_2\kappa + \ldots$ for the inverse form. That is a measurable half-exponent that the framework's own hardware buys, and it generalizes HELICON Law II from option Greeks to any second-moment tracker. Replace the golden-ratio line with it.

### 4.3 col(F)/ker(F) partition, made concrete

§II.2 describes every system as splitting into an estimable sector col(F) and a hidden sector ker(F), with the Fisher matrix's principal singular values as the generalization direction. In clustering this is literal. For cluster $j$, $\Sigma_j = R_j^\top R_j$; directions with large $R$ diagonal (large variance) are where the cluster is *wide* and its mean is *well-determined relative to spread*; directions with tiny $R$ diagonal are where the Fisher information $\Sigma_j^{-1}$ is *large* and assignment is *sharp*. Euclidean assignment ignores this partition entirely — which is the mechanism of companion Exp 3 (one feature at scale 100 → ARI 0.40 vs 0.91). The rotation-native tracker maintains the partition explicitly as the triangular factor, and the framework's "the partition protects itself" (§II.2) has a precise meaning here: once $R_j$ is maintained by unimodular rotations it cannot leave the PSD cone, whereas the SM inverse leaves it at 10 of 20 checkpoints in bfloat16 (Exp A). The topological-protection language is apt; the clustering experiment is the first place in these documents where it is measured.

### 4.4 Mixture-of-Curvature ↔ e-means on the hyperboloid

P3 and §V.2 propose per-token curvature routing between Euclidean and hyperbolic experts with CORDIC mode selection. Clustering on hyperbolic embeddings (HELM-class) is the obvious test workload and the frameworks do not mention it. Two facts make it rotation-native:

1. In the Lorentz model the centroid of points $x_i$ on the hyperboloid is the normalized sum $\mu = S/\sqrt{-\langle S,S\rangle_L}$ with $S = \sum_i x_i$ (Law et al. 2019). **This is e-means' sum-and-normalize with the Euclidean count $N$ replaced by the Minkowski norm of $S$.** The EMA of raw Lorentz vectors followed by one hyperbolic vectoring step ($x^2 - y^2$ invariant, $m=-1$) is therefore an exact e-means on the hyperboloid. Exp D: the EMA centroid sits on the manifold to machine precision ($\langle c,c\rangle_L = -1.0$) and is within the EMA's own variance of the batch Lorentz centroid.
2. The Fisher metric on the hyperboloid is the Lorentz metric itself, so the "right metric" for assignment is $-\langle x, c_j\rangle_L$, computed by the same hyperbolic unit. The anisotropy problem of §1.2 does not disappear in hyperbolic space; it becomes a per-cluster Lorentz-covariance, tracked by Givens rotations in the tangent space at $c_j$ — Lorentz boosts (hyperbolic rotations, $m=-1$) to move to the tangent frame, circular Givens ($m=+1$) to update the factor. Both modes, one datapath. That is a Mixture-of-Curvature workload that exists today and has a baseline (Euclidean e-means in the tangent space) to beat.

### 4.5 P7 (bit-exact training → reproducibility) must be restated

See §3.5. As written: "CORDIC's bit-exact arithmetic … enables reproducible training." Evidence: e-means on exact arithmetic is not reproducible across batch orders. Restated: *bit-exact arithmetic removes the arithmetic source of non-reproducibility; algorithmic sources (data order, RNG, tie-breaking) require either order-invariant algorithms or a logged transcript, and fixed-point hardware makes the transcript sufficient because replay is then exact.* That is still a strong and defensible prediction, and it is the one QARE already implements (§5.2).

---

## 5. QUADRANT / QARE: the transcript, the type system, and the tiers

### 5.1 Unimodularity is why the rotation-native row is reversible and the multiply-native row is not

QARE's foundation (§"Unimodularity"): a map on a finite-word register is realizable in place iff $\det = \pm1$; classical CORDIC pseudo-rotations have $\det = 1 + 4^{-i}$ and must be replaced by shear triples. The same test sorts the clustering grid. A Givens rotation has $\det = 1$ and factors into three shears exactly as QARE's $R(\theta) = S_x(\tan\tfrac\theta2)\,S_y(\sin\theta)\,S_x(\tan\tfrac\theta2)$ — so **the forgetting-weighted precision factor can be maintained by a reversible ladder**, with the Givens angles forming a direction record exactly as in QARE §"Direction Record". The Sherman–Morrison update $P \mapsto P - Puu^\top P/(1+u^\top Pu)$ has $\det(P_t) = \det(P_{t-1})/(1 + u^\top P u)$ (matrix determinant lemma) — not unimodular, not a permutation, not reversible without garbage. And the e-means update $S \mapsto \gamma S + (1-\gamma)s$ is a contraction ($\det = \gamma^d$), which is *irreversible by construction* — forgetting is information destruction. This is not a criticism of e-means; it is the reason it cannot be run on a QARE-class register and the Givens tracker can. The frameworks' hardware distinction (reversible ladder vs classical datapath) coincides with the algorithmic distinction (orthogonal update vs contraction).

### 5.2 The direction record is the transcript e-means is missing

QARE's direction bits $\{d_i\}$ reconstruct the input angle exactly, serve as a checksum, and make the computation replayable by an auditor (§"Direction Record", §"Replayability"). The IQAE output is typed `Estimated(value, halfwidth, confidence, queries, transcript, residual)` and Q-F12 forbids collapsing it to a point. Apply the type system to clustering outputs:

| algorithm | output type | transcript needed for replay | invariant carried |
|---|---|---|---|
| Lloyd from fixed init | **Attested** (function of data + init) | none | monotone loss; $\nabla L = 0$ at fixed point |
| Givens tracker on a stream | **Attested** (function of stream) | the stream order, which is already the data | PSD by construction; mass + first moment conserved |
| e-means | **Estimated** | permutation seed, revival RNG state, precision, reduction order, finalization mode | none until the final Lloyd step |

An e-means codebook shipped into an IVF index, a VQ bottleneck or a quantizer is currently handed over as a point. QARE's rule says it must carry its transcript or it cannot be audited; §3.5 says without the transcript it cannot even be reproduced. **This is a concrete, cheap change to the Faiss `contrib/e_means` API: return the permutation seed, revival log, dtype and finalization mode alongside the centroids, and refuse to serialize without them.** It costs nothing and converts an Estimated into a replayable one.

### 5.3 D0–D4 register placement for clustering

QARE places quantum estimation at D2 (one expectation, intraday, legal deadline) and never at D1 (real-time) or D4 (per-household fan-out), by matching each algorithm's cost structure to a tier. The same placement for clustering, using QARE's tier table:

| tier | latency | clustering object | algorithm | why |
|---|---|---|---|---|
| D0 | snapshot, 2.5 ns read | the **authoritative codebook** at seq $n$ | Lloyd (or Hartigan) from k-means++, run offline, attested | order-independent; the thing everyone else reads |
| D1 | <1 ms, streaming | per-cluster $(c_j, R_j, N_j)$ **tracker**; Mahalanobis anomaly score per message | Givens-QR with forgetting $\lambda$ | $O(d^2)$ per sample, exact, PSD, deterministic; 28 M samples/s at $d=96$ (§2.3) |
| D2 | 1–10 min cycle | **re-fit** of the codebook on the accumulated window when the D1 tracker reports drift (residual energy $> \mathrm{tr}\,\Sigma_j$ persistently) | **e-means** at scale, or Lloyd below $10^6$ | this is e-means' actual regime: one batch job, many epochs, GPU, no latency constraint |
| D3/D4 | fan-out | per-entity scenario clustering | Lloyd, parallel | parallelism beats everything (QARE D4 argument) |

e-means at D1 is the misplacement QARE's Q-F13 warns about: an epoch-structured batch optimizer with an annealed-to-floor learning rate, deployed as a stream tracker (companion Exp 4: 2.2× the tracking error of the Givens tracker). e-means at D2 is correct and nothing here argues against it.

### 5.4 Q-F16 applies to e-means' headline

QARE Q-F16: "speedup claimed on arithmetic alone … constant factor 2× better, but no exponent change." e-means' 10–45× is a constant-factor speedup at fixed epoch count on the same $O(nkd)$ per-epoch assignment cost; it changes no exponent, and the paper is careful to say "in one-tenth the epochs". Downstream packaging (Faiss README "~10× speedup", third-party issues calling it "state of the art for the large case") is where the Q-F16 collapse happens. The frameworks' discipline of stating $C_c$, $C_q$ and the crossover explicitly (QARE Q-F15) would, applied to e-means, read: *crossover at $n \approx 10^6$, $d \gtrsim 96$, $\kappa(\Sigma_j) \approx 1$; below any of those, Lloyd++.* That sentence is in the companion decision matrix and should be in the e-means README.

---

## 6. A rotation-native clustering engine (design)

Name it whatever you like; here it is called **GYRE** (Givens-Yielded Rotation Estimator). It is the second row of the grid, built from the three frameworks' primitives, with every correction from §§3–5 applied.

### 6.1 State per cluster

```
N_j   count (EMA, forgetting λ)                  1 word    Q(integer-heavy)
c_j   mean                                       d words   Q24.47 internal
R_j   upper-triangular Cholesky factor of        d(d+1)/2  Q24.47 internal
      forgetting-weighted scatter, R_jᵀR_j = Σ_j
ℓ_j   log det Σ_j = 2 Σ_i log R_j[i,i]           1 word    harvested from R_j diag via hyperbolic ln (m=−1)
```

No inverse is ever stored. $\sigma_j$ (per-coordinate RMS) is the column norm of $R_j$ — harvested, not computed.

### 6.2 Per-sample pipeline

```
INGRESS      x  →  standardize by global σ (one linear-mode multiply per coord)  [metric mux default: M]

ASSIGN       for each j:  y = R_j^{−T}(x − c_j)      d back-substitution steps, m=0
                          D_j = ‖y‖² + ℓ_j           + Hartigan weight N_j/(N_j+1), one m=0 divide
             j* = argmin_j D_j                        tie-break by cluster index (deterministic)

UPDATE j*    e = x − c_{j*}
             N  ← λN + 1 ;  c ← c + e/N              one m=0 divide
             R  ← Givens(√λ R, e)                    d vectoring (m=+1) + d(d+1)/2 rotations (m=+1)
             ℓ  ← ℓ + 2[log R'_ii − log R_ii] on the touched diagonal   (m=−1, harvested)

MONITOR      Σ_j N_j − EMA(1)            → mass residual      (must be 0 to ULP)
             Σ_j N_j c_j − EMA(x)        → first-moment residual (must be 0 to ULP)
             R_{j*} diagonal > 0          → PSD attestation (free: it is a sign check)
             |e|² / tr(R_{j*}ᵀR_{j*}) EMA > 1 + τ_drift persistently → DRIFT flag for j*

REVIVE       criterion: loss-aware. Candidate = cluster with smallest N_j·tr(Σ_j) (least mass × spread);
             donor = cluster with largest residual energy. Split additively: c ± δ·σ_donor ⊙ r,
             r random unit vector from a logged seed. Both children get R_donor/√2, N_donor/2.
             Log (seq, j_dead, j_donor, seed) to the transcript. Monitor sees a known jump; it is attributed.

FINALIZE     none required. State is always a valid estimator. For an Attested snapshot, freeze at seq n
             and run one Lloyd M-step over the window → D0 codebook.
```

### 6.3 What each line fixes

| line | fixes | from |
|---|---|---|
| metric mux default M | anisotropy, feature scale (ARI 0.12→0.47, 0.40→0.91) | companion §3.2–3.3; HELICON Law V |
| Hartigan weight | high-$d$ fixed-point pathology | Lederman et al. 2026; HELICON Law VI |
| Givens update, no inverse | exactness in narrow precision; PSD by construction; unimodular / reversible | Exp A; QARE §Unimodularity |
| constant $\lambda$, drift flag | tracking under drift (2.2× better than cosine EMA) | companion §3.4 |
| deterministic tie-break, logged seeds | replay; Estimated → Attested | companion §3.5; QARE §Direction Record |
| mass / first-moment / PSD monitor | per-result attestation at zero cost | HELICON Law IV; Exp C |
| additive, shape-aware split from harvested $\sigma_j$ | zero-coordinate revival artifact; fixed-direction artifact | companion §3.1 |
| loss-aware revival criterion | merged-cluster blindness (0 revivals in 5/6 failed runs) | companion Exp 5 |
| $N_0 = 1$, $S_0 = c_0$ | fixed-point hostability of the state | Exp B; HELICON Law II |

### 6.4 Cost and placement

Per sample: $d$ vectoring + $d(d+1)/2$ rotation + $kd(d+1)/2$ MAC + $k$ divides. At $d = 96$, $k = 1024$: $4.7\times10^3$ rotations on the rotation engine, $4.8\times10^6$ MACs on the multiply engine. The rotation engine is never the bottleneck; the assignment is, and it is GEMM-shaped, so it goes to the matmul co-processor with the rotation engine owning the state — Topological Rotation's prefill/decode split, applied to clustering. Memory $k(d^2/2 + 2d + 2)$ words: 4.9 M at $(96, 1024)$. Regime: $k \lesssim 10^4$, $d \lesssim 10^3$, streaming or drifting data, any precision down to float16/Q-format. Above that, e-means at D2.

---

## 7. Corrections, consolidated

Each item names the document, the section, the problem, and the fix.

| # | document | section | problem | fix |
|---|---|---|---|---|
| 1 | Topological Rotation | II.3 | $\kappa(F) = \varphi$ "master identity" has no derivation and contradicts measured $\kappa(F) = \kappa(\Sigma_j)$ | replace with the half-exponent bit-width law of §4.2 |
| 2 | Topological Rotation | I.2 vs II.3 | contraction constant stated as 0.5 and as $1/\varphi^2$ | keep 0.5 ($\lvert z_n\rvert \le 2^{-n}$ is the theorem) |
| 3 | Topological Rotation | II.3 | TaAs $\Delta E \sim \varphi E_0$, Fermi-arc count $\sim F(n)$: uncited | cite or remove |
| 4 | Topological Rotation | P7 | bit-exact arithmetic ⇒ reproducible training | add: ∧ (order-invariant algorithm ∨ logged transcript) — §3.5, §4.5 |
| 5 | Topological Rotation | III.2 workloads | clustering / quantization / codebook learning absent | add GYRE (§6) as a rotation-native workload; e-means as the matmul-engine baseline |
| 6 | HELICON | §13 | "determinism" conflates arithmetic and algorithmic determinism | state both; HELICON satisfies both because BSM is a function |
| 7 | HELICON | Law II | not applied to any state with time-varying Jacobian | add the $1/N_j$ example (Exp B) as the canonical non-stationary case |
| 8 | HELICON | P1 | CORDIC survives only where DSPs are contended | add: the Givens factor update is rotation-only and has *no* polynomial substitute at equal numerical properties (PSD, unimodular); it survives regardless of DSP contention |
| 9 | QARE | tier table | no clustering placement | add §5.3 |
| 10 | QARE | Q-F12 | Estimated-vs-Attested applied only to quantum outputs | extend to any stochastic estimator; e-means codebooks are Estimated |
| 11 | e-means (Faiss contrib) | API | centroids returned without transcript | return (perm seed, revival log, dtype, finalization) with the codebook |
| 12 | e-means (paper) | §3, revival | multiplicative BF16-$\varepsilon$ split; count-based criterion | additive $\sigma_j$-scaled split; loss-aware criterion (§6.2) |
| 13 | e-means (paper) | Eq. 6–7 | laws clip to $\alpha_0=1$ outside fitted range | publish the fitted range as a guard; fall back to Lloyd++ outside it |

---

## 8. Objections, answered

**"The Givens tracker is $O(d^2)$ per sample; e-means is $O(kd)$. You are comparing different costs."** Yes, and the point is which engine pays. Per sample the tracker touches one cluster: $d(d+1)/2$ rotations, independent of $k$. e-means' $O(kd)$ is the assignment, which the tracker *also* pays (as $kd(d+1)/2$ MACs with the Fisher metric, or $kd$ with the Euclidean mux setting). The second moment adds $O(d^2)$ per sample and $O(kd^2)$ memory, and §6.4 states exactly where that stops being affordable ($k \gtrsim 10^4$ or $d \gtrsim 10^3$). Inside that envelope the tracker is cheaper than one e-means assignment pass.

**"bfloat16 broke the Givens factor too (35–58% error)."** It did, and the table says so. Eight mantissa bits are not enough for a $\lambda = 0.99$ forgetting factor in either form, and the check against the effective rounded $\lambda$ did not rescue it. The claim is not "Givens is accurate in bf16"; it is "Givens degrades boundedly and stays PSD, SM goes indefinite and diverges, and the asymmetry is range vs mantissa." Float16 (10 bits, no range) makes the asymmetry unambiguous: 1.5–3.7% vs overflow at step 0.

**"The frameworks are about options, quantum finance and chip roadmaps, not clustering."** They are about *rotation as the primitive and determinism as the product*. Clustering is where a 2026 state-of-the-art algorithm (e-means) makes the opposite bet — multiplication as the primitive, speed as the product — and the comparison is therefore the cleanest available test of the frameworks' thesis. The thesis survives the test (§4.1) and three of its constants do not (§4.2).

**"You have not run GYRE end to end."** Correct. §6 is a design with costed components, each of which is measured separately in this folder or the companion. An end-to-end implementation on the Volder-1 generator is the natural next experiment; the falsifier would be a measured per-sample rotation count exceeding $d(d+1)/2 + d$ on the touched cluster, or a loss of PSD on any Q24.47 run.

---

## 9. Reproduction

```
python3 -I experiments.py             # companion: e-means failure modes (results.json)
python3 -I addendum.py                # companion: Lloyd++ / e-means-with-k-means++ controls
python3 -I rotation_experiments.py    # this document: Exp A–D (rotation_results.json)
```

`rotation_experiments.py` contains: `givens`, `qr_update` (square-root RLS with forgetting, dtype-emulated), `sm_update` (Sherman–Morrison, dtype-emulated), a bfloat16 round-to-nearest-even emulator with float32 range, the Q24.40 ratio analysis, the e-means mass/first-moment invariant check with a revival, and the Lorentz-centroid EMA on the hyperboloid.

Not shown and not claimed: GYRE end-to-end; Q-format (as opposed to float16) emulation of the Givens sweep; any statement about e-means at $n \ge 10^6$, where the paper's results stand.

---

## 10. Sources

Framework documents (by the author of this folder)
- *Topological Rotation: The Weyl-CORDIC-Semimetal Unified Framework for Next-Generation Chip Architecture*, v2.0, June 2026. github.com/ericrenone/Topological-Rotation-August-2026
- *HELICON: Hyperbolic · Euclidean · Linear Iterative CORDIC Option Network*. github.com/ericrenone/HELICON
- *QUADRANT + QARE: Reversible CORDIC Ladder with Hardware-Accelerated SRAM/DMA State Preparation for Coupled-Path Amplitude Estimation*. github.com/ericrenone/QARE-Reversible-CORDIC-Ladder-with-Hardware-Accelerated-SRAM

Subject
- Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou. *e-means: outperforming k-means (in one-tenth the epochs).* arXiv:2611.09501 (Nov 2026). Faiss `contrib/e_means`.

Rotation-native least squares (the bridge)
- Gentleman, Kung. *Matrix triangularization by systolic arrays.* Proc. SPIE 298 (1981).
- McWhirter. *Recursive least-squares minimization using a systolic array.* Proc. SPIE 431 (1983).
- Haykin. *Adaptive Filter Theory*, ch. on QR-RLS and inverse QR-RLS (square-root forms).
- Volder (1959); Walther (1971) — CORDIC vectoring and rotation modes, as cited in the frameworks.

Clustering theory and metric
- Lederman, Silva-Sánchez, Chen, Mordant, Balanov, Bendory. *The Catastrophic Failure of the k-Means Algorithm in High Dimensions, and How Hartigan's Algorithm Avoids It.* arXiv:2602.09936 (ICML 2026).
- Chen, Bayrooti, Morad. *Learning Mixture Density via Natural Gradient Expectation Maximization.* arXiv:2602.10602 (Feb 2026).
- Armah, Melnykov. *K-Means Clustering With Incomplete Data with the Use of Mahalanobis Distances.* arXiv:2411.00870.
- Dasgupta, Mahajan, So. *Convergence of online k-means.* arXiv:2202.10640 (AISTATS 2022).
- Law, Liao, Snell, Zemel. *Lorentzian Distance Learning for Hyperbolic Representations.* ICML 2019 (Lorentz centroid as normalized sum).

Online precision / rank-1
- Pinto, Engel. *A Fast Incremental Gaussian Mixture Model.* arXiv:1506.04422; *Scalable and Incremental Learning of GMMs.* arXiv:1701.03940.
- *Rank-k inverse update strategies (DI vs iterated SM vs Woodbury) for streaming outlier detection.* arXiv:2603.16697 (Mar 2026).
- Mitz, Shkolnisky. *ROIPCA.* arXiv:1911.11049.

Streaming / drift
- Iglesias, Konzett, Zseby, Bifet. *Stream Clustering Robust to Concept Drift.* IJCNN 2025.
- Cossu et al. *A Practical Guide to Streaming Continual Learning.* arXiv:2603.01677 (2026).
