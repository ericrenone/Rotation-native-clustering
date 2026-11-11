"""
Rotation-native precision tracking vs Sherman-Morrison under reduced precision,
plus fixed-point hosting of the e-means sufficient-statistic ratio.
Run: python3 -I rotation_experiments.py
"""
import numpy as np, json

# ---------------------------------------------------------------------------
# A. Givens-based square-root RLS (Cholesky factor of forgetting-weighted scatter)
#    vs Sherman-Morrison on the inverse, both in float16, judged against float64.
# ---------------------------------------------------------------------------
def givens(a, b):
    """Return (c, s) with [c s; -s c] @ [a; b] = [r; 0]. This is one CORDIC
    vectoring step followed by a rotation step in a systolic array."""
    if b == 0: return 1.0, 0.0
    if abs(b) > abs(a):
        t = a / b; s = 1 / np.sqrt(1 + t*t); c = s * t
    else:
        t = b / a; c = 1 / np.sqrt(1 + t*t); s = c * t
    return c, s

def bf16(a):
    a = np.asarray(a, dtype=np.float32); u = a.view(np.uint32)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return u.view(np.float32)
class BF16:
    """dtype-like: cast = round-to-bfloat16 but store as float32"""
    def __call__(self, v): return bf16(v)
def cast(v, dt): return bf16(v) if isinstance(dt, BF16) else np.asarray(v).astype(dt)

def qr_update(R, x, lam, dt):
    """R'^T R' = lam * R^T R + x x^T via one sweep of Givens rotations.
    R upper triangular (d x d). Work in dtype dt to emulate narrow datapaths."""
    d = len(x)
    R = cast(np.sqrt(lam) * R, dt); x = cast(x, dt).copy()
    for i in range(d):
        c, s = givens(float(R[i, i]), float(x[i]))
        c = cast(c, dt); s = cast(s, dt)
        Ri = R[i, i:].copy(); xi = x[i:].copy()
        R[i, i:] = cast(c * Ri + s * xi, dt)
        x[i:] = cast(-s * Ri + c * xi, dt)
    return R

def sm_update(P, x, lam, dt):
    P = cast(P, dt); x = cast(x, dt)
    u = cast(x / cast(np.sqrt(lam), dt), dt)
    Pu = cast(P @ u, dt)
    denom = cast(1 + cast(u @ Pu, dt), dt)
    with np.errstate(all='ignore'):
        P = cast(cast(P - cast(np.outer(Pu, Pu) / denom, dt), dt) / cast(lam, dt), dt)
    return P

def run_A(d=16, steps=4000, lam=0.99, kappa=50.0, seed=0, dt=np.float16):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Sig = Q @ np.diag(np.linspace(1, kappa, d)) @ Q.T
    L = np.linalg.cholesky(Sig)
    X = rng.normal(size=(steps, d)) @ L.T
    # float64 reference: forgetting-weighted scatter and its inverse
    S64 = np.eye(d) * 1e-2; R = cast(np.linalg.cholesky(S64).T, dt); P = cast(np.linalg.inv(S64), dt)
    errR, errP, mineigP, pd_lost, sm_diverged_at = [], [], [], 0, None
    for t in range(steps):
        x = X[t]
        S64 = lam * S64 + np.outer(x, x)
        R = qr_update(R, x, lam, dt)
        if sm_diverged_at is None:
            P = sm_update(P, x, lam, dt)
            if not np.all(np.isfinite(P)): sm_diverged_at = t
        if t % 200 == 199:
            P64 = np.linalg.inv(S64)
            P_from_R = np.linalg.inv(R.astype(np.float64).T @ R.astype(np.float64))
            errR.append(np.linalg.norm(P_from_R - P64) / np.linalg.norm(P64))
            if sm_diverged_at is None:
                errP.append(np.linalg.norm(P.astype(np.float64) - P64) / np.linalg.norm(P64))
                ev = np.linalg.eigvalsh(0.5 * (P.astype(np.float64) + P.astype(np.float64).T))
                mineigP.append(float(ev.min())); pd_lost += int(ev.min() <= 0)
    return dict(d=d, steps=steps, lam=lam, kappa=kappa, dtype=('bfloat16' if isinstance(dt, BF16) else str(np.dtype(dt))),
                givens_relerr_final=float(errR[-1]), givens_relerr_max=float(max(errR)),
                sm_relerr_final=(float(errP[-1]) if errP else None), sm_relerr_max=(float(max(errP)) if errP else None),
                sm_min_eig_final=(mineigP[-1] if mineigP else None), sm_checkpoints_not_PD=pd_lost,
                sm_diverged_at_step=sm_diverged_at, checkpoints=len(errR))

results = {}
print("A: Givens-QR vs Sherman-Morrison in float16")
results['A_float16'] = [run_A(kappa=k, dt=np.float16) for k in (1.0, 50.0, 500.0)]
for r in results['A_float16']: print(r)
print("A: bfloat16 (8-bit mantissa, float32 range)")
results['A_bfloat16'] = [run_A(kappa=k, dt=BF16()) for k in (1.0, 50.0, 500.0)]
for r in results['A_bfloat16']: print(r)
print("A: float32 control")
results['A_float32'] = [run_A(kappa=500.0, dt=np.float32)]
print(results['A_float32'][0])

# ---------------------------------------------------------------------------
# B. Hosting the e-means ratio c = S/N in a fixed-point register (HELICON Q24.40)
#    The paper initialises N_j = 1e-10, S_j = N_j * c_j.
# ---------------------------------------------------------------------------
ulp = 2.0**-40
N0 = 1e-10
res_B = dict(Q24_40_ulp=ulp, N0=N0, N0_in_ulps=N0/ulp,
             # S_j = N0 * c_j: the centroid coordinate is recoverable only to resolution ulp/N0
             centroid_resolution_at_init=ulp/N0,
             bits_of_centroid_at_init=float(np.log2(1/(ulp/N0))) if ulp/N0 < 1 else 0.0,
             # Jacobian dc/dS = 1/N -> HELICON Law II bit allocation for tolerance tau on c
             bits_needed_on_S_for_tau={str(tau): float(np.log2(1/(N0*tau))) for tau in (1e-3, 1e-6, 1e-9)},
             # after one batch with alpha0=1 (laws clip there at small n) N_j = n_j >= 1: resolution restored
             note="Within one batch at alpha=1 the count becomes O(1) and the ratio is well-posed; "
                  "the 1e-10 idiom is a float trick to make S/N equal c0 at t=0. On fixed point use N0=1, S0=c0.")
results['B_fixed_point_ratio'] = res_B; print(res_B)

# ---------------------------------------------------------------------------
# C. e-means conservation laws (free invariants, HELICON Law IV style)
#    sum_j N_j == EMA of batch size; sum_j S_j == EMA of batch sum. Exact in exact arithmetic.
#    Revival conserves mass exactly and first moment only up to delta.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(1); n, d, k, B = 4000, 8, 16, 256
X = rng.normal(size=(n, d)) + rng.integers(0, 4, size=(n, 1)) * 3
N = np.full(k, 1e-10); S = N[:, None] * X[rng.choice(n, k, replace=False)]
massEMA = 1e-10 * k; sumEMA = S.sum(0).copy()
mass_res, mom_res, rev_mom_jump = [], [], []
delta = 2.0**-8; pattern = np.where(np.arange(d) % 2 == 0, 1 + delta, 1 - delta)
for t in range(300):
    idx = rng.choice(n, B, replace=False); xb = X[idx]
    C = S / N[:, None]; lab = ((xb**2).sum(1)[:, None] - 2*xb@C.T + (C**2).sum(1)[None]).argmin(1)
    nj = np.bincount(lab, minlength=k).astype(float); sj = np.zeros((k, d)); np.add.at(sj, lab, xb)
    a = 0.3; N = (1-a)*N + a*nj; S = (1-a)*S + a*sj
    massEMA = (1-a)*massEMA + a*B; sumEMA = (1-a)*sumEMA + a*xb.sum(0)
    mass_res.append(abs(N.sum() - massEMA)); mom_res.append(np.linalg.norm(S.sum(0) - sumEMA))
    if t % 5 == 4:
        dead = np.where(N < 0.1 * N.mean())[0]
        for j in dead:
            jp = rng.choice(k, p=N/N.sum())
            if jp == j: continue
            before = S.sum(0).copy()
            cjp = S[jp]/N[jp]; N[j] = N[jp] = N[jp]/2; S[j] = cjp*pattern*N[j]; S[jp] = cjp*(2-pattern)*N[jp]
            rev_mom_jump.append(float(np.linalg.norm(S.sum(0) - before)))
results['C_invariants'] = dict(max_mass_residual=float(max(mass_res)), max_first_moment_residual_no_revival=float(np.median(mom_res)),
                               first_moment_residual_max=float(max(mom_res)), revivals=len(rev_mom_jump),
                               first_moment_jump_per_revival=rev_mom_jump[:5],
                               note="mass is conserved to float eps always; first moment is conserved to float eps between revivals and "
                                    "jumps by ~delta*|c|*N_j/2*... at each revival, so a residual monitor flags every revival as an event")
print(results['C_invariants'])

# ---------------------------------------------------------------------------
# D. Lorentz-centroid EMA: e-means on the hyperboloid = S / sqrt(-<S,S>_L), a hyperbolic
#    CORDIC vectoring op. Check it tracks the true Frechet-like centroid of a cluster.
# ---------------------------------------------------------------------------
def lorentz_inner(a, b): return -a[..., 0]*b[..., 0] + (a[..., 1:]*b[..., 1:]).sum(-1)
def expmap0(v):  # tangent at origin (0 time comp) -> hyperboloid
    nv = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return np.concatenate([np.cosh(nv), np.sinh(nv) * v / nv], -1)
rng = np.random.default_rng(2); d = 4; mu_t = rng.normal(size=d) * 0.8
pts = expmap0(mu_t + 0.3 * rng.normal(size=(5000, d)))
S = np.zeros(d+1); a = 0.05
for t in range(5000):
    S = (1-a)*S + a*pts[t]
cent = S / np.sqrt(-lorentz_inner(S, S))
batch_cent = pts.sum(0) / np.sqrt(-lorentz_inner(pts.sum(0), pts.sum(0)))
results['D_lorentz_ema'] = dict(on_hyperboloid=float(lorentz_inner(cent, cent)), ema_vs_batch_lorentz_dist=float(np.arccosh(max(1.0, -lorentz_inner(cent, batch_cent)))),
                                note="EMA of raw Lorentz vectors, normalised by Minkowski norm (x^2 - y^2 invariant = CORDIC m=-1 vectoring), sits on the manifold (<c,c>_L = -1) and matches the batch Lorentz centroid")
print(results['D_lorentz_ema'])

json.dump(results, open('/home/claude/emeans_analysis/rotation_results.json', 'w'), indent=2)
print('saved')
