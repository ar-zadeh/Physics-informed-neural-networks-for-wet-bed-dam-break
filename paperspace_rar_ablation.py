#!/usr/bin/env python3
"""Self-contained Paperspace/A6000 ablation for the shallow-water PINN.

This file deliberately has no imports from the surrounding research repository.
It installs its small dependency set, builds the network and residuals, and runs
a six-arm, three-seed sampling ablation inspired by the paper:

    J-uniform                 Jacobian (autodiff) residual, uniform points
    J-RAR                     Jacobian residual, Jacobian-indicator RAR
    Roe-uniform               Roe finite-difference residual, uniform points
    Roe-J-RAR                 Roe training, Jacobian-indicator RAR
    Roe-Roe-RAR               Roe training, Roe-indicator RAR
    Roe-hybrid                Roe training, half uniform/half Roe-indicator

The RAR implementation here fixes two easy-to-miss failure modes: the probe
pool is refreshed every round, and anchors are deduplicated across rounds. The
default indicator is sqrt(W_CONT*r_cont**2 + W_MOM*r_mom**2).
Use ``--paper-indicator sum_abs`` to reproduce the dimensionally mixed
|r_cont| + |r_mom| indicator from the original experiment.

Example on a Paperspace A6000 (run from a terminal or notebook cell):

    python paperspace_rar_ablation.py --output runs/ablation --workers 1

All arms start with the same 16,000 random uniform points and add 600 points
per round. Uniform additions and RAR use equal time-bin quotas. This is a new
controlled ablation, not an exact reproduction of the chapter's fixed grid.
To resume, repeat the command with --resume and the same output/configuration.
Atomic checkpoints preserve completed Adam stages and the final L-BFGS stage;
an interrupted stage restarts from its beginning. Models are saved before evaluation.

The default schedule is the paper schedule (4000 + 5 x 1000 Adam and a final
L-BFGS stage). ``--quick`` is available for a smoke test before launching the
full campaign. Each condition/seed is an independent process, so multiple
conditions run concurrently while each process uses one CUDA device.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Installation and imports
# -----------------------------------------------------------------------------

def install_missing() -> None:
    """Install only packages that are absent from the Paperspace image."""
    requirements = {
        "numpy": "numpy>=1.24",
        "scipy": "scipy>=1.10",
        # Paperspace normally already contains the CUDA build. Installing a
        # generic torch wheel is only a fallback for a fresh Python image.
        "torch": "torch>=2.1",
    }
    missing = [spec for module, spec in requirements.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing missing packages:", ", ".join(missing), flush=True)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", *missing
        ])


def lazy_imports():
    """Import heavy libraries after installation (also used by spawned workers)."""
    import numpy as np
    import torch
    import torch.nn as nn
    from scipy.optimize import brentq
    return np, torch, nn, brentq


# -----------------------------------------------------------------------------
# Physics and exact solution
# -----------------------------------------------------------------------------

G = 9.81
HL = 1.0
HR = 0.1
XL = -10.0
XR = 10.0
TEND = 1.25
X0 = 0.0
DX = 0.01
HMIN = 1.0e-4
U_SCALE = 0.5
W_CONT = 1.0
W_MOM = 0.0509684


def stoker_parameters(brentq):
    """Return intermediate depth/velocity and wave speeds for the wet-bed case."""
    c_l = math.sqrt(G * HL)

    def rarefaction_u(hm: float) -> float:
        return 2.0 * (c_l - math.sqrt(G * hm))

    def shock_u(hm: float) -> float:
        return (hm - HR) * math.sqrt(G * (hm + HR) / (2.0 * hm * HR))

    hm = float(brentq(lambda z: rarefaction_u(z) - shock_u(z), HR * (1 + 1e-7), HL))
    um = rarefaction_u(hm)
    cm = math.sqrt(G * hm)
    shock = hm * um / (hm - HR)
    return {
        "hm": hm,
        "um": um,
        "cm": cm,
        "shock_speed": shock,
        "fan_head_speed": -c_l,
        "fan_tail_speed": um - cm,
    }


def exact_solution(x, t: float, params: dict[str, float], np):
    """Vectorized Stoker wet-bed solution."""
    x = np.asarray(x, dtype=float)
    if t <= 0.0:
        return np.where(x < X0, HL, HR), np.zeros_like(x)
    xi = (x - X0) / t
    fan_head = params["fan_head_speed"]
    fan_tail = params["fan_tail_speed"]
    shock = params["shock_speed"]
    c_l = math.sqrt(G * HL)
    hm, um = params["hm"], params["um"]
    left = xi <= fan_head
    fan = (xi > fan_head) & (xi <= fan_tail)
    middle = (xi > fan_tail) & (xi <= shock)
    h = np.full_like(xi, HR, dtype=float)
    u = np.zeros_like(xi, dtype=float)
    h[left], u[left] = HL, 0.0
    c = (2.0 * c_l - xi[fan]) / 3.0
    h[fan], u[fan] = c * c / G, (2.0 * xi[fan] + 2.0 * c_l) / 3.0
    h[middle], u[middle] = hm, um
    return h, u


# -----------------------------------------------------------------------------
# Network and residuals
# -----------------------------------------------------------------------------

def make_network(torch, nn, seed: int):
    torch.manual_seed(seed)
    layers: list[Any] = [nn.Linear(2, 32), nn.Tanh()]
    for _ in range(4):
        layers += [nn.Linear(32, 32), nn.Tanh()]
    layers += [nn.Linear(32, 2)]
    net = nn.Sequential(*layers)
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    class Constrained(nn.Module):
        def __init__(self, body):
            super().__init__()
            self.body = body

        def forward(self, x):
            raw = self.body(x)
            h = nn.functional.softplus(raw[:, 0:1]) + HMIN
            u = U_SCALE * raw[:, 1:2]
            return torch.cat((h, u), dim=1)

    return Constrained(net)


def flux_roe(h_l, u_l, h_r, u_r, torch):
    """Roe flux with the same Harten-Hyman entropy fix as the paper."""
    q_l, q_r = h_l * u_l, h_r * u_r
    fhl, fhr = q_l, q_r
    fql = q_l * u_l + 0.5 * G * h_l * h_l
    fqr = q_r * u_r + 0.5 * G * h_r * h_r
    sl, sr = torch.sqrt(torch.clamp(h_l, min=1e-6)), torch.sqrt(torch.clamp(h_r, min=1e-6))
    uhat = (sl * u_l + sr * u_r) / (sl + sr + 1e-10)
    chat = torch.sqrt(torch.clamp(0.5 * G * (h_l + h_r), min=1e-6))
    lam1, lam2 = uhat - chat, uhat + chat
    delta = 0.1 * chat

    def entropy_abs(lam):
        return torch.where(
            torch.abs(lam) >= delta,
            torch.abs(lam),
            (lam * lam + delta * delta) / (2.0 * torch.clamp(delta, min=1e-8)),
        )

    a1, a2 = entropy_abs(lam1), entropy_abs(lam2)
    two_c = 2.0 * chat
    a00 = (a1 * (uhat + chat) + a2 * (chat - uhat)) / two_c
    a01 = (a2 - a1) / two_c
    a10 = (uhat * uhat - chat * chat) * (a1 - a2) / two_c
    a11 = ((uhat + chat) * a2 - (uhat - chat) * a1) / two_c
    dh, dq = h_r - h_l, q_r - q_l
    fh = 0.5 * (fhl + fhr) - 0.5 * (a00 * dh + a01 * dq)
    fq = 0.5 * (fql + fqr) - 0.5 * (a10 * dh + a11 * dq)
    return fh, fq


def input_grad(value, x, torch, create_graph: bool):
    return torch.autograd.grad(
        value, x, grad_outputs=torch.ones_like(value),
        create_graph=create_graph, retain_graph=True,
    )[0]


def residual(model, x, kind: str, torch, create_graph: bool = True):
    """Return continuity and momentum residuals for a point tensor."""
    y = model(x)
    h, u = y[:, 0:1], y[:, 1:2]
    q = h * u
    q_grad = input_grad(q, x, torch, create_graph)
    if kind == "jacobian":
        h_grad = input_grad(h, x, torch, create_graph)
        f_q = q * u + 0.5 * G * h * h
        fq_x = input_grad(f_q, x, torch, create_graph)
        return h_grad[:, 1:2] + q_grad[:, 0:1], q_grad[:, 1:2] + fq_x[:, 0:1]

    if kind != "roe":
        raise ValueError(f"Unknown residual kind: {kind}")
    h_t = input_grad(h, x, torch, create_graph)
    x_c, t_c = x[:, 0:1], x[:, 1:2]
    x_r, x_l = x_c + DX, x_c - DX
    outside_r, outside_l = x_r > XR, x_l < XL
    y_r = model(torch.cat((torch.clamp(x_r, XL, XR), t_c), dim=1))
    y_l = model(torch.cat((torch.clamp(x_l, XL, XR), t_c), dim=1))
    h_r = torch.where(outside_r, torch.full_like(h, HR), y_r[:, 0:1])
    u_r = torch.where(outside_r, torch.zeros_like(u), y_r[:, 1:2])
    h_l = torch.where(outside_l, torch.full_like(h, HL), y_l[:, 0:1])
    u_l = torch.where(outside_l, torch.zeros_like(u), y_l[:, 1:2])
    fh_r, fq_r = flux_roe(h, u, h_r, u_r, torch)
    fh_l, fq_l = flux_roe(h_l, u_l, h, u, torch)
    return h_t[:, 1:2] + (fh_r - fh_l) / DX, q_grad[:, 1:2] + (fq_r - fq_l) / DX


# -----------------------------------------------------------------------------
# Sampling, RAR, and training
# -----------------------------------------------------------------------------

def make_points(seed: int, n_domain: int, n_ic: int = 1600, n_bc: int = 400):
    """Generate deterministic, nested point sets shared by all arms of a seed."""
    np, _, _, _ = lazy_imports()
    rng = np.random.default_rng(seed + 100_000)
    domain_19k = np.column_stack((rng.uniform(XL, XR, n_domain), rng.uniform(0.0, TEND, n_domain)))
    ic_x = rng.uniform(XL, XR, n_ic)
    ic = np.column_stack((ic_x, np.zeros(n_ic)))
    times = rng.uniform(0.0, TEND, n_bc)
    bc_l = np.column_stack((np.full(n_bc, XL), times))
    bc_r = np.column_stack((np.full(n_bc, XR), times))
    return domain_19k[:n_domain], domain_19k, ic, bc_l, bc_r


def select_stratified(points, scores, n, np, spacing, existing, bins):
    """Give time bins equal quotas; spacing is strict (no silent backfill)."""
    selected = []
    for i in range(bins):
        quota = n // bins + (i < n % bins)
        if not quota:
            continue
        mask = (points[:, 1] >= TEND * i / bins) & (points[:, 1] < TEND * (i + 1) / bins)
        prior = np.concatenate([existing, *selected], axis=0)
        chosen = select_anchors(points[mask], scores[mask], quota, np, spacing, prior)
        # Verify the spacing contract across time-bin boundaries too.
        scale = np.array([XR - XL, TEND])
        for j, p in enumerate(chosen):
            other = np.concatenate((prior, chosen[:j]), axis=0)
            if len(other) and np.any(np.linalg.norm((other - p) / scale, axis=1) < spacing):
                raise ValueError("Anchor spacing exhausted a time-bin quota; reduce --min-dist-norm or increase --probe-points")
        selected.append(chosen)
    return np.concatenate(selected) if selected else np.empty((0, 2), dtype=np.float32)


def score_candidates(model, points, kind: str, torch, np, chunk: int):
    """Evaluate a RAR indicator in chunks without constructing a parameter graph."""
    was_training = model.training
    model.eval()
    old_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    r1_all, r2_all = [], []
    try:
        for start in range(0, len(points), chunk):
            x = torch.as_tensor(points[start:start + chunk], dtype=torch.float32, device=next(model.parameters()).device)
            x.requires_grad_(True)
            with torch.enable_grad():
                r1, r2 = residual(model, x, kind, torch, create_graph=False)
            r1_all.append(r1.detach().cpu().numpy().reshape(-1))
            r2_all.append(r2.detach().cpu().numpy().reshape(-1))
            del x, r1, r2
    finally:
        for p, flag in zip(model.parameters(), old_flags):
            p.requires_grad_(flag)
        model.train(was_training)
    return np.concatenate(r1_all), np.concatenate(r2_all)


def indicator_score(r1, r2, mode: str, np):
    if mode == "sum_abs":
        return np.abs(r1) + np.abs(r2)
    if mode == "loss_aligned":
        return np.sqrt(W_CONT * r1 ** 2 + W_MOM * r2 ** 2)
    if mode != "nondimensional":
        raise ValueError(mode)
    # Continuity scales as c and momentum as c^2. This is dimensionless and
    # follows the same physical scaling used by the loss weight.
    c = math.sqrt(G * HL)
    return np.sqrt((r1 / c) ** 2 + (r2 / (c * c)) ** 2)


def select_anchors(points, scores, n, np, min_dist_normalized: float, existing=None):
    """Greedy top-score selection with cross-round spacing in normalized coordinates."""
    if n == 0:
        return np.empty((0, 2), dtype=np.float32)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Nonfinite residual scores")
    order = np.argsort(scores)[::-1]
    scale = np.array([XR - XL, TEND], dtype=float)
    chosen = []
    existing_norm = (np.asarray(existing, dtype=float) / scale
                     if existing is not None and len(existing) else np.empty((0, 2)))
    for idx in order:
        p = points[idx]
        pn = p / scale
        if len(existing_norm) or chosen:
            parts = [existing_norm]
            if chosen:
                parts.append(np.asarray(chosen) / scale)
            c = np.concatenate(parts, axis=0)
            distances = np.linalg.norm(c - pn, axis=1)
            if np.any(distances == 0) or np.min(distances) < min_dist_normalized:
                continue
        chosen.append(p)
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        raise ValueError("Insufficient spaced candidates: reduce --min-dist-norm or increase --probe-points")
    return np.asarray(chosen[:n], dtype=np.float32)


def loss_components(model, domain, ic, bc_l, bc_r, kind, torch):
    x = torch.as_tensor(domain, dtype=torch.float32, device=next(model.parameters()).device)
    x.requires_grad_(True)
    r1, r2 = residual(model, x, kind, torch, create_graph=True)
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    comps = [torch.mean(r1 * r1), W_MOM * torch.mean(r2 * r2)]
    for pts, target_h, target_u in ((bc_l, HL, 0.0), (bc_r, HR, 0.0)):
        xb = torch.as_tensor(pts, dtype=torch.float32, device=x.device)
        yb = model(xb)
        comps += [torch.mean((yb[:, 0] - target_h) ** 2), torch.mean((yb[:, 1] - target_u) ** 2)]
    xi = torch.as_tensor(ic, dtype=torch.float32, device=x.device)
    yi = model(xi)
    h_target = torch.where(xi[:, 0] < X0, torch.full_like(xi[:, 0], HL), torch.full_like(xi[:, 0], HR))
    comps += [torch.mean((yi[:, 0] - h_target) ** 2), torch.mean(yi[:, 1] ** 2)]
    weights = [W_CONT, 1.0, 10.0, 0.509684, 1.0, 1.0, 1.0, 1.0]
    total = zero
    for w, c in zip(weights, comps):
        total = total + w * c
    return total, comps


def adam_block(model, domain, ic, bc_l, bc_r, kind, steps, lr, torch):
    if steps <= 0:
        return float("nan")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    last = float("nan")
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total, _ = loss_components(model, domain, ic, bc_l, bc_r, kind, torch)
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite Adam loss")
        total.backward()
        optimizer.step()
        last = float(total.detach().cpu())
    return last


def lbfgs_polish(model, domain, ic, bc_l, bc_r, kind, max_iter, torch):
    if max_iter <= 0:
        return float("nan"), 0
    calls = [0]
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max_iter, max_eval=max_iter * 2,
        tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        total, _ = loss_components(model, domain, ic, bc_l, bc_r, kind, torch)
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite L-BFGS loss")
        total.backward()
        calls[0] += 1
        return total

    optimizer.step(closure)
    final, _ = loss_components(model, domain, ic, bc_l, bc_r, kind, torch)
    return float(final.detach().cpu()), calls[0]


def predict(model, x, torch, np, chunk: int = 8192):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            xt = torch.as_tensor(x[start:start + chunk], dtype=torch.float32, device=next(model.parameters()).device)
            out.append(model(xt).cpu().numpy())
    return np.concatenate(out, axis=0)


# -----------------------------------------------------------------------------
# Metrics and one independent run
# -----------------------------------------------------------------------------

def evaluate(model, params, torch, np):
    x = np.linspace(XL, XR, 4001)
    y = predict(model, np.column_stack((x, np.full_like(x, 1.2))), torch, np)
    h_ex, u_ex = exact_solution(x, 1.2, params, np)
    rel_h = float(np.linalg.norm(y[:, 0] - h_ex) / np.linalg.norm(h_ex))
    rel_u = float(np.linalg.norm(y[:, 1] - u_ex) / (np.linalg.norm(u_ex) + 1e-12))
    # A descending half-height crossing must actually exist; do not report a
    # nearest point as a shock when the wave has disappeared.
    level = 0.5 * (params["hm"] + HR)
    crossings = np.flatnonzero((x[:-1] > 0.5) & (y[:-1, 0] >= level) & (y[1:, 0] < level))
    shock_x = float("nan")
    if len(crossings) == 1:
        j = crossings[0]
        shock_x = float(x[j] + (level - y[j, 0]) * (x[j+1] - x[j]) / (y[j+1, 0] - y[j, 0]))
    exact_shock_x = params["shock_speed"] * 1.2
    from scipy.integrate import trapezoid
    mass = float(trapezoid(y[:, 0], x))
    expected_mass = HL * (X0 - XL) + HR * (XR - X0)
    mass_pct = 100.0 * abs(mass - expected_mass) / expected_mass
    # Match the chapter's six evaluation times and 1201-point error grid.
    time_metrics = []
    grid = np.linspace(XL, XR, 1201)
    for fraction in [0.05, 0.10, 0.20, 0.50, 0.80, 0.96]:
        t = round(TEND * fraction, 4)
        pred = predict(model, np.column_stack((grid, np.full_like(grid, t))), torch, np)
        eh, eu = exact_solution(grid, t, params, np)
        rh = float(np.linalg.norm(pred[:, 0] - eh) / np.linalg.norm(eh))
        ru = float(np.linalg.norm(pred[:, 1] - eu) / np.linalg.norm(eu))
        time_metrics.append({"t": t, "relative_l2_h": rh, "relative_l2_u": ru})
    rel_h, rel_u = time_metrics[-1]["relative_l2_h"], time_metrics[-1]["relative_l2_u"]
    combined = float(np.mean([0.5 * (m["relative_l2_h"] + m["relative_l2_u"]) for m in time_metrics]))
    return {
        "relative_l2_h_t1.2": rel_h,
        "relative_l2_u_t1.2": rel_u,
        "combined_error": combined,
        "mean_time_balanced_error": combined,
        "metrics_all_times": time_metrics,
        "shock_location_error_m_t1.2": abs(shock_x - exact_shock_x),
        "mass_error_pct_t1.2": mass_pct,
        "shock_prediction_m_t1.2": shock_x,
        "shock_crossing_count": int(len(crossings)),
        "shock_exact_m_t1.2": exact_shock_x,
    }, x, y


def train_one(task: tuple[str, int, dict[str, Any], str]):
    """Worker entry point. One process owns one model and one CUDA device."""
    condition, seed, cfg, output_root = task
    install_missing()
    np, torch, nn, brentq = lazy_imports()
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        if not cfg["allow_cpu"]:
            raise RuntimeError("CUDA is unavailable. Run this script on the Paperspace A6000 or pass --allow-cpu.")
        device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    params = stoker_parameters(brentq)
    train_domain, _, ic, bc_l, bc_r = make_points(seed, cfg["initial_points"])
    model = make_network(torch, nn, seed).to(device)
    probe_rng = np.random.default_rng(seed + 700_000)
    anchor_points = []
    anchor_counts = []
    out = Path(output_root) / f"{condition}__seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint.pt"
    completed_rounds = 0
    warmed = False
    polished = False
    lbfgs_loss, lbfgs_calls = None, 0
    elapsed_previous = 0.0
    t0 = time.perf_counter()
    if cfg["resume"] and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        expected = {k: v for k, v in cfg.items() if k != "resume"}
        if state["config"] != expected:
            raise ValueError("Checkpoint configuration differs from this run")
        model.load_state_dict(state["model"])
        anchor_points = [np.asarray(a, dtype=np.float32).reshape(-1, 2) for a in state["anchors"]]
        anchor_counts = [len(a) for a in anchor_points]
        if anchor_points:
            train_domain = np.concatenate((train_domain, *anchor_points))
        probe_rng.bit_generator.state = state["rng"]
        completed_rounds, warmed, polished = state["rounds"], state["warmed"], state["polished"]
        lbfgs_loss, lbfgs_calls = state["lbfgs_loss"], state["lbfgs_calls"]
        elapsed_previous = state["elapsed"]

    def save_checkpoint():
        state = {"model": model.state_dict(), "config": {k: v for k, v in cfg.items() if k != "resume"},
                 "anchors": [a.tolist() for a in anchor_points], "rng": probe_rng.bit_generator.state,
                 "rounds": completed_rounds, "warmed": warmed, "polished": polished,
                 "lbfgs_loss": lbfgs_loss, "lbfgs_calls": lbfgs_calls,
                 "elapsed": elapsed_previous + time.perf_counter() - t0}
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(state, temporary)
        os.replace(temporary, checkpoint)

    if not warmed:
        adam_block(model, train_domain, ic, bc_l, bc_r, cfg["train_kind"], cfg["adam_initial"], 1e-3, torch)
        warmed = True
        save_checkpoint()
    for round_id in range(completed_rounds, cfg["rar_rounds"]):
            # Refreshing the pool avoids repeatedly selecting the same finite set.
            probe = np.column_stack((
                probe_rng.uniform(XL, XR, cfg["probe_points"]),
                probe_rng.uniform(0.0, TEND, cfg["probe_points"]),
            )).astype(np.float32)
            n = cfg["anchors_per_round"]
            n_rar = (n // 2 if cfg.get("hybrid") else n) if cfg["adaptive"] else 0
            scores = probe_rng.random(len(probe))
            prior = np.asarray(train_domain, dtype=np.float32)
            # Uniform and hybrid exploration uses the same time quotas as RAR.
            uniform = select_stratified(probe, scores, n - n_rar, np, 0.0, prior, cfg["time_bins"])
            if n_rar:
                r1, r2 = score_candidates(model, probe, cfg["rar_kind"], torch, np, cfg["probe_chunk"])
                scores = indicator_score(r1, r2, cfg["indicator_score"], np)
                # Remove exact exploration points before residual selection.
                keep = ~np.isin(np.ascontiguousarray(probe).view('V8').ravel(), np.ascontiguousarray(uniform).view('V8').ravel())
                existing = np.concatenate(anchor_points + [uniform]) if anchor_points else uniform
                adaptive = select_stratified(probe[keep], scores[keep], n_rar, np, cfg["min_dist_norm"], existing, cfg["time_bins"])
                anchors = np.concatenate((uniform, adaptive))
            else:
                anchors = uniform
            train_domain = np.concatenate((train_domain, anchors), axis=0)
            anchor_points.append(anchors)
            anchor_counts.append(int(len(anchors)))
            adam_block(model, train_domain, ic, bc_l, bc_r, cfg["train_kind"], cfg["adam_round"], 5e-4, torch)
            completed_rounds = round_id + 1
            save_checkpoint()
            print(f"{condition} seed={seed}: round {completed_rounds}/{cfg['rar_rounds']}, points={len(train_domain)}", flush=True)
    if not polished:
        lbfgs_loss, lbfgs_calls = lbfgs_polish(
            model, train_domain, ic, bc_l, bc_r, cfg["train_kind"], cfg["lbfgs_max_iter"], torch
        )
        polished = True
        save_checkpoint()
    metrics, x_eval, y_eval = evaluate(model, params, torch, np)
    elapsed = elapsed_previous + time.perf_counter() - t0
    result = {
        "condition": condition,
        "seed": seed,
        "train_residual": cfg["train_kind"],
        "rar_indicator": cfg["rar_kind"] if cfg["adaptive"] else "none",
        "indicator_score": cfg["indicator_score"] if cfg["adaptive"] else "none",
        "adaptive": bool(cfg["adaptive"]),
        "initial_domain_points": cfg["initial_points"],
        "final_domain_points": int(len(train_domain)),
        "anchors_added": int(sum(anchor_counts)),
        "anchor_counts": anchor_counts,
        "anchor_exact_unique": int(len(np.unique(np.concatenate(anchor_points), axis=0))) if anchor_points else 0,
        "probe_points_per_round": cfg["probe_points"] if cfg["adaptive"] else 0,
        "adam_initial_steps": cfg["adam_initial"],
        "adam_round_steps": cfg["adam_round"],
        "rar_rounds": cfg["rar_rounds"],
        "lbfgs_function_evals": int(lbfgs_calls),
        "lbfgs_final_loss": lbfgs_loss,
        "elapsed_seconds": elapsed,
        **metrics,
    }
    out = Path(output_root) / f"{condition}__seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(out / "prediction_t1.2.npz", x=x_eval, h=y_eval[:, 0], u=y_eval[:, 1])
    if anchor_points:
        np.savez_compressed(out / "rar_anchors.npz", **{
            f"round_{i}": a for i, a in enumerate(anchor_points)
        })
    return result


# -----------------------------------------------------------------------------
# Campaign driver
# -----------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=None, help="Output directory (default: rar_ablation_TIMESTAMP)")
    p.add_argument("--workers", type=int, default=1, help="Concurrent processes sharing CUDA device 0")
    p.add_argument("--resume", action="store_true", help="Resume completed stages from the same output directory")
    p.add_argument("--initial-points", type=int, default=16000)
    p.add_argument("--time-bins", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds (default: 0 1 2)")
    p.add_argument("--probe-points", type=int, default=120_000)
    p.add_argument("--probe-chunk", type=int, default=8_192)
    p.add_argument("--anchors-per-round", type=int, default=600)
    p.add_argument("--rar-rounds", type=int, default=5)
    p.add_argument("--adam-initial", type=int, default=4_000)
    p.add_argument("--adam-round", type=int, default=1_000)
    p.add_argument("--lbfgs-max-iter", type=int, default=15_000)
    p.add_argument("--min-dist-norm", type=float, default=0.001,
                   help="Normalized x/t anchor spacing; prevents cross-round duplicates")
    p.add_argument("--paper-indicator", choices=["loss_aligned", "nondimensional", "sum_abs"], default="loss_aligned",
                   help="RAR score (default: weighted training-residual norm)")
    p.add_argument("--allow-cpu", action="store_true", help="Permit a CPU run when CUDA is unavailable")
    p.add_argument("--quick", action="store_true", help="Smoke test: 500 + 2 x 100 Adam, 2k probes, 100 anchors, L-BFGS 50")
    return p.parse_args()


def main():
    args = parse_args()
    install_missing()
    if args.quick:
        args.probe_points, args.probe_chunk = 2_000, 1_000
        args.anchors_per_round, args.rar_rounds = 100, 2
        args.adam_initial, args.adam_round, args.lbfgs_max_iter = 500, 100, 50
    np, torch, _, _ = lazy_imports()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is unavailable. This campaign targets a Paperspace A6000; use --allow-cpu only for a smoke test.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    for key in ["initial_points", "probe_points", "probe_chunk", "time_bins"]:
        if getattr(args, key) <= 0:
            raise SystemExit(f"{key} must be positive")
    for key in ["anchors_per_round", "rar_rounds", "adam_initial", "adam_round", "lbfgs_max_iter", "min_dist_norm"]:
        if getattr(args, key) < 0:
            raise SystemExit(f"{key} must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("Seeds must be unique")
    if args.resume and args.output is None:
        raise SystemExit("--resume requires --output")
    output = args.output or Path(f"rar_ablation_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit("Output is not empty; choose a new directory or use --resume")
    output.mkdir(parents=True, exist_ok=True)
    cfg_common = {
        "resume": args.resume,
        "initial_points": args.initial_points,
        "time_bins": args.time_bins,
        "probe_points": args.probe_points,
        "probe_chunk": args.probe_chunk,
        "anchors_per_round": args.anchors_per_round,
        "rar_rounds": args.rar_rounds,
        "adam_initial": args.adam_initial,
        "adam_round": args.adam_round,
        "lbfgs_max_iter": args.lbfgs_max_iter,
        "min_dist_norm": args.min_dist_norm,
        "indicator_score": args.paper_indicator,
        "allow_cpu": args.allow_cpu,
    }
    arms = [
        ("jacobian_uniform", "jacobian", None, False),
        ("jacobian_rar", "jacobian", "jacobian", True),
        ("roe_uniform", "roe", None, False),
        ("roe_jacobian_rar", "roe", "jacobian", True),
        ("roe_roe_rar", "roe", "roe", True),
        ("roe_hybrid", "roe", "roe", True),
    ]
    jobs = []
    for name, train_kind, rar_kind, adaptive in arms:
        cfg = dict(cfg_common, train_kind=train_kind, rar_kind=rar_kind, adaptive=adaptive, hybrid=name == "roe_hybrid")
        for seed in args.seeds:
            jobs.append((name, int(seed), cfg, str(output)))
    manifest = {
        "created": dt.datetime.now().isoformat(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "workers": args.workers,
        "versions": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda},
        "arms": [a[0] for a in arms],
        "seeds": args.seeds,
        "config": cfg_common,
        "note": "RAR refreshes probe pool and deduplicates anchors across rounds.",
    }
    if args.resume and (output / "manifest.json").exists():
        old = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        old_cfg = {k: v for k, v in old["config"].items() if k != "resume"}
        new_cfg = {k: v for k, v in cfg_common.items() if k != "resume"}
        if old_cfg != new_cfg or old["seeds"] != args.seeds:
            raise SystemExit("Resume requires the original configuration and seeds")
    with open(output / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Running {len(jobs)} jobs on {manifest['device']} with {args.workers} worker(s).", flush=True)
    results: list[dict[str, Any]] = []
    # spawn is required for CUDA; do not fork an initialized CUDA context.
    mp.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(train_one, job): job[:2] for job in jobs}
        for future in as_completed(futures):
            name, seed = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"DONE {name} seed={seed}: E={result['combined_error']:.5f}, "
                      f"h={100*result['relative_l2_h_t1.2']:.2f}%, "
                      f"u={100*result['relative_l2_u_t1.2']:.2f}%", flush=True)
            except Exception as exc:  # keep the other seeds running
                failure = {"condition": name, "seed": seed, "error": repr(exc)}
                results.append(failure)
                print(f"FAILED {name} seed={seed}: {exc!r}", flush=True)
            write_csv(output / "runs.csv", results)
    results.sort(key=lambda r: (r.get("condition", ""), int(r.get("seed", -1))))
    write_csv(output / "runs.csv", results)
    # Aggregate only successful runs; medians are robust to the known seed variance.
    successful = [r for r in results if "combined_error" in r]
    summary = []
    for condition in sorted({r["condition"] for r in successful}):
        rr = [r for r in successful if r["condition"] == condition]
        vals = [r["combined_error"] for r in rr]
        summary.append({
            "condition": condition,
            "n": len(rr),
            "combined_median": float(np.median(vals)),
            "combined_mean": float(np.mean(vals)),
            "combined_std": float(np.std(vals)),
            "depth_median_pct": 100.0 * float(np.median([r["relative_l2_h_t1.2"] for r in rr])),
            "velocity_median_pct": 100.0 * float(np.median([r["relative_l2_u_t1.2"] for r in rr])),
            "bore_max_cm": (100.0 * max(r["shock_location_error_m_t1.2"] for r in rr)
                            if all(r["shock_crossing_count"] == 1 for r in rr) else float("nan")),
            "invalid_shock_runs": sum(r["shock_crossing_count"] != 1 for r in rr),
        })
    write_csv(output / "ablation_summary.csv", summary)
    with open(output / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Results written to {output.resolve()}", flush=True)
    if any("error" in r for r in results):
        raise SystemExit("Some jobs failed; inspect runs.csv and resume after addressing the error")


if __name__ == "__main__":
    main()
