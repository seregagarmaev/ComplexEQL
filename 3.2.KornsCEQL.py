# parallel_korns_complexeql.py

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import numpy as np

from config.korns_config import BENCH, FEATURE_NAMES, CEQL_TRAIN
from src.korns_core import (
    RunConfig,
    load_korns_hdf5,
    BenchmarkRow,
    init_results_csv,
    append_results_csv_row,
)
from src.metrics import compute_metrics, Metrics


def _worker_run_one_seed(
    *,
    seed: int,
    hdf5_path: str,
    feature_names: List[str],
    drop_pids: Optional[List[str]] = None,
    device: str = "cpu",
) -> List[BenchmarkRow]:
    # Imports inside worker for clean multiprocessing spawn
    import sympy as sp
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    from config.korns_config import CEQL
    from src.ComplexEQL import ComplexEQL
    from src.utils import set_seed, train

    # Avoid CPU oversubscription when running multiple processes
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Force device (recommended: cpu for multiprocessing)
    mcfg = CEQL_TRAIN
    mcfg.device = device

    print(f"[WORKER START] seed={seed} device={device}", flush=True)

    datasets = load_korns_hdf5(hdf5_path)
    if drop_pids:
        for pid in drop_pids:
            datasets.pop(pid, None)

    rows: List[BenchmarkRow] = []
    algo_name = "complexeql"

    pids = [pid for pid, _ in sorted(datasets.items(), key=lambda kv: int(kv[0][1:]))]
    print(f"[WORKER] seed={seed} problems={len(pids)} pids={pids}", flush=True)

    dev = torch.device(device)
    mse_loss = nn.MSELoss()

    # ---- NEW: convergence policy ----
    converge_mse = 1e-4
    max_attempts = 3  # cap to avoid infinite retries

    for pid, rec in sorted(datasets.items(), key=lambda kv: int(kv[0][1:])):
        print(f"[WORKER] seed={seed} PID_START {pid}", flush=True)

        X_train, y_train = rec.X_train, rec.y_train
        X_test, y_test = rec.X_test, rec.y_test

        # Ensure CEQL input dimension matches data
        CEQL.n_input_fields = int(X_train.shape[1])

        Xtr = torch.tensor(np.asarray(X_train, dtype=np.float32), device=dev)
        ytr = torch.tensor(np.asarray(y_train, dtype=np.float32).reshape(-1, 1), device=dev)

        dataset = TensorDataset(Xtr, ytr)

        # Per-problem deterministic RNG that proposes new retry seeds
        pid_int = int(pid[1:])  # "P7" -> 7
        retry_rng = np.random.RandomState(seed * 10_000 + pid_int)

        best = {
            "train_mse": float("inf"),
            "attempt": -1,
            "attempt_seed": None,
            "model_state": None,
            "y_pred_train": None,
            "y_pred_test": None,
            "expr_pred": None,
        }
        converged = False
        seeds_tried: List[int] = []

        for attempt in range(max_attempts):
            attempt_seed = int(retry_rng.randint(0, 2**31 - 1))
            seeds_tried.append(attempt_seed)

            # re-seed, re-init model/optim each attempt
            set_seed(attempt_seed)

            dataloader = DataLoader(
                dataset,
                batch_size=int(getattr(mcfg, "train_batch_size", 2**14)),
                shuffle=True,
                drop_last=False,
            )

            model = ComplexEQL(CEQL).to(dev)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(getattr(mcfg, "lr", 1e-3)))
            scheduler = None
            if getattr(mcfg, "scheduler", None) == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **getattr(mcfg, "schedulerparams", {})
                )

            model, _history = train(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                loss_fn=mse_loss,
                cfg=mcfg,
                device=dev,
                scheduler=scheduler,
            )

            model.eval()
            with torch.no_grad():
                yhat_tr = model(Xtr, op_params=None).real  # (N,1)
                train_mse_val = float(mse_loss(yhat_tr, ytr).item())

                Xte = torch.tensor(np.asarray(X_test, dtype=np.float32), device=dev)
                yhat_te = model(Xte, op_params=None).real  # (N,1)

            # Keep best attempt (even if not converged)
            if train_mse_val < best["train_mse"]:
                best["train_mse"] = train_mse_val
                best["attempt"] = attempt
                best["attempt_seed"] = attempt_seed
                best["model_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best["y_pred_train"] = yhat_tr.squeeze(-1).detach().cpu().numpy()
                best["y_pred_test"] = yhat_te.squeeze(-1).detach().cpu().numpy()

                # Symbolic readout (optional; keep best’s expression)
                try:
                    syms = [sp.Symbol(n) for n in feature_names[: CEQL.n_input_fields]]
                    best["expr_pred"] = model.get_symbolic_expression(syms, rounding_decimals=2)
                except Exception:
                    best["expr_pred"] = None

            if train_mse_val <= converge_mse:
                converged = True
                print(
                    f"[WORKER] seed={seed} {pid} CONVERGED attempt={attempt} "
                    f"attempt_seed={attempt_seed} train_mse={train_mse_val:.3e}",
                    flush=True,
                )
                break
            else:
                print(
                    f"[WORKER] seed={seed} {pid} retry attempt={attempt} "
                    f"attempt_seed={attempt_seed} train_mse={train_mse_val:.3e} > {converge_mse:.1e}",
                    flush=True,
                )

        # Use best attempt’s outputs
        y_pred_train = best["y_pred_train"]
        y_pred_test = best["y_pred_test"]
        expr_pred = best["expr_pred"]
        expr_str = str(expr_pred) if expr_pred is not None else ""

        m: Metrics = compute_metrics(
            y_train_true=y_train,
            y_train_pred=y_pred_train,
            y_test_true=y_test,
            y_test_pred=y_pred_test,
            expr_gt=rec.expr_gt,
            expr_pred=expr_pred,
            feature_names=feature_names,
        )

        extra = {
            "seed": int(seed),
            "converged": bool(converged),
            "converge_mse": float(converge_mse),
            "max_attempts": int(max_attempts),
            "attempt_used": int(best["attempt"]),
            "attempt_seed_used": int(best["attempt_seed"]) if best["attempt_seed"] is not None else -1,
            "attempt_seeds_tried": seeds_tried,
            "best_train_mse": float(best["train_mse"]),
        }

        rows.append(
            BenchmarkRow(
                pid=pid,
                algo=algo_name,
                run_id=int(seed),
                nlse_test=m.nlse_test,
                mse_test=m.mse_test,
                mape_test=m.mape_test,
                nlse_train=m.nlse_train,
                mse_train=m.mse_train,
                mape_train=m.mape_train,
                term_precision=m.term_precision,
                term_recall=m.term_recall,
                term_f1=m.term_f1,
                expr_str=expr_str,
                expr_gt_str=str(rec.expr_gt),
                extra=extra,
            )
        )

        print(f"[WORKER] seed={seed} PID_DONE {pid}", flush=True)

    print(f"[WORKER DONE] seed={seed} rows={len(rows)}", flush=True)
    return rows


def run_parallel_seeds(
    *,
    seeds: List[int],
    results_csv_path: str,
    hdf5_path: str,
    feature_names: List[str],
    drop_pids: Optional[List[str]] = None,
    device: str = "cpu",
    max_workers: Optional[int] = None,
) -> List[BenchmarkRow]:
    init_results_csv(results_csv_path, BenchmarkRow)

    all_rows: List[BenchmarkRow] = []

    if max_workers is None:
        max_workers = min(len(seeds), os.cpu_count() or 1)

    print(
        f"[START] seeds={seeds} max_workers={max_workers} device={device} "
        f"csv='{results_csv_path}' hdf5='{hdf5_path}'",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(
                _worker_run_one_seed,
                seed=s,
                hdf5_path=hdf5_path,
                feature_names=feature_names,
                drop_pids=drop_pids,
                device=device,
            )
            for s in seeds
        ]

        for fut in as_completed(futs):
            rows = fut.result()
            for r in rows:
                append_results_csv_row(results_csv_path, r)
            all_rows.extend(rows)
            print(f"[DONE] appended {len(rows)} rows (one seed-run) -> {results_csv_path}", flush=True)

    print(f"[ALL DONE] total_rows={len(all_rows)} -> {results_csv_path}", flush=True)
    return all_rows


if __name__ == "__main__":
    cfg = RunConfig(
        hdf5_path=BENCH.hdf5_path,
        test_size=BENCH.test_size,
        split_seed=BENCH.split_seed,
        per_problem_seed_offset=BENCH.per_problem_seed_offset,
        algo_seed_offset=BENCH.algo_seed_offset,
        run_seed_offset=BENCH.run_seed_offset,
    )

    drop = ["P4", "P11", "P12", "P13", "P14", "P15"]
    seeds = [0, 1, 2, 3, 4]

    run_parallel_seeds(
        seeds=seeds,
        results_csv_path="korns_complexeql_benchmark_results.csv",
        hdf5_path=cfg.hdf5_path,
        feature_names=FEATURE_NAMES,
        drop_pids=drop,
        device="cpu",
        max_workers=5,
    )
