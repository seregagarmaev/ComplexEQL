from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Dict, Any, List, Optional, Protocol, Tuple

import numpy as np
import sympy as sp
import h5py

from src.metrics import Metrics, compute_metrics


@dataclass(frozen=True)
class DatasetRecord:
    pid: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    expr_gt: sp.Expr


# def load_korns_hdf5(path: str) -> Dict[str, DatasetRecord]:
#     out: Dict[str, DatasetRecord] = {}
#     with h5py.File(path, "r") as f:
#         for pid in f.keys():
#             grp = f[pid]
#             X = np.asarray(grp["X"][:], dtype=np.float64)
#             y = np.asarray(grp["y"][:], dtype=np.float64).reshape(-1)
#             if "expr_srepr" in grp.attrs:
#                 expr_gt = sp.sympify(grp.attrs["expr_srepr"])
#             else:
#                 expr_gt = sp.sympify(grp.attrs["expr_str"])
#             out[pid] = DatasetRecord(pid=pid, X=X, y=y, expr_gt=expr_gt)
#     return out
def load_korns_hdf5(path: str) -> Dict[str, DatasetRecord]:
    out: Dict[str, DatasetRecord] = {}
    with h5py.File(path, "r") as f:
        for pid in f.keys():
            grp = f[pid]

            X_train = np.asarray(grp["X_train"][:], dtype=np.float64)
            y_train = np.asarray(grp["y_train"][:], dtype=np.float64).reshape(-1)

            X_test = np.asarray(grp["X_extrap"][:], dtype=np.float64)
            y_test = np.asarray(grp["y_extrap"][:], dtype=np.float64).reshape(-1)

            # prefer srepr if present; else expr_str
            if "expr_srepr" in grp.attrs:
                expr_gt = sp.sympify(grp.attrs["expr_srepr"])
            else:
                expr_gt = sp.sympify(grp.attrs["expr_str"])

            out[pid] = DatasetRecord(
                pid=pid,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                expr_gt=expr_gt,
            )
    return out


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(test_size * n))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


@dataclass
class SRFitResult:
    expr: Optional[sp.Expr]
    y_pred_train: np.ndarray
    y_pred_test: np.ndarray
    metadata: Optional[Dict[str, Any]]


class SymbolicRegressor(Protocol):
    name: str
    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> SRFitResult:
        ...


@dataclass
class RunConfig:
    hdf5_path: str
    test_size: float
    split_seed: int
    per_problem_seed_offset: int
    algo_seed_offset: int
    run_seed_offset: int


@dataclass
class BenchmarkRow:
    pid: str
    algo: str
    run_id: int

    nlse_test: float
    mse_test: float
    mape_test: float

    nlse_train: float
    mse_train: float
    mape_train: float

    term_precision: float
    term_recall: float
    term_f1: float

    expr_str: str
    expr_gt_str: str
    extra: Optional[Dict[str, Any]] = None


def zlib_crc32(b: bytes) -> int:
    import zlib
    return zlib.crc32(b) & 0xFFFFFFFF


def _get_algo_seed(algo_name: str) -> int:
    return int(np.uint32(zlib_crc32(algo_name.encode("utf-8"))))


# def run_benchmark(
#     datasets: Dict[str, DatasetRecord],
#     algorithms: List[SymbolicRegressor],
#     config: RunConfig,
#     n_runs: int,
#     feature_names: List[str],
# ) -> List[BenchmarkRow]:
#     rows: List[BenchmarkRow] = []

#     for pid, rec in sorted(datasets.items(), key=lambda kv: int(kv[0][1:])):
#         split_seed = config.split_seed + config.per_problem_seed_offset + int(pid[1:])
#         X_train, X_test, y_train, y_test = train_test_split(
#             rec.X, rec.y, test_size=config.test_size, seed=split_seed
#         )

#         print("=" * 50)
#         print(f"[PROBLEM] {pid}")
#         print(f"[GT] {rec.expr_gt}")

#         for algo in algorithms:
#             algo_seed = _get_algo_seed(algo.name)
#             print("=" * 50)
#             print(f"[ALGO] {algo.name}")

#             for run_id in range(n_runs):
#                 run_seed = (
#                     config.split_seed
#                     + config.per_problem_seed_offset * int(pid[1:])
#                     + config.algo_seed_offset * algo_seed
#                     + config.run_seed_offset * run_id
#                 )
#                 print(f"[RUN START] run_id={run_id} seed={run_seed}")

#                 fit_res = algo.fit_predict(X_train, y_train, X_test)
#                 m: Metrics = compute_metrics(
#                     y_true=y_test,
#                     y_pred=fit_res.y_pred_test,
#                     expr_gt=rec.expr_gt,
#                     expr_pred=fit_res.expr,
#                     feature_names=feature_names,
#                 )

#                 expr_pred = str(fit_res.expr) if fit_res.expr is not None else ""
#                 print(f"[PRED] {expr_pred}")

#                 extra = dict(fit_res.metadata) if fit_res.metadata else {}
#                 extra["run_seed"] = int(run_seed)

#                 rows.append(
#                     BenchmarkRow(
#                         pid=pid,
#                         algo=algo.name,
#                         run_id=run_id,
#                         nlse=m.nlse,
#                         term_precision=m.term_precision,
#                         term_recall=m.term_recall,
#                         term_f1=m.term_f1,
#                         expr_str=expr_pred,
#                         expr_gt_str=str(rec.expr_gt),
#                         extra=extra,
#                     )
#                 )

#     return rows


def run_benchmark(
    datasets: Dict[str, DatasetRecord],
    algorithms: List[SymbolicRegressor],
    config: RunConfig,
    n_runs: int,
    feature_names: List[str],
    results_csv_path: Optional[str] = None,
) -> List[BenchmarkRow]:
    rows: List[BenchmarkRow] = []

    if results_csv_path is not None:
        init_results_csv(results_csv_path, BenchmarkRow)

    for pid, rec in sorted(datasets.items(), key=lambda kv: int(kv[0][1:])):
        X_train, y_train = rec.X_train, rec.y_train
        X_test, y_test = rec.X_test, rec.y_test

        print("=" * 50)
        print(f"[PROBLEM] {pid}")
        print(f"[GT] {rec.expr_gt}")

        for algo in algorithms:
            algo_seed = _get_algo_seed(algo.name)
            print("=" * 50)
            print(f"[ALGO] {algo.name}")

            for run_id in range(n_runs):
                run_seed = (
                    config.split_seed
                    + config.per_problem_seed_offset * int(pid[1:])
                    + config.algo_seed_offset * algo_seed
                    + config.run_seed_offset * run_id
                )

                print(f"[RUN START] run_id={run_id} seed={run_seed}")

                fit_res = algo.fit_predict(X_train, y_train, X_test)

                m: Metrics = compute_metrics(
                    y_train_true=y_train,
                    y_train_pred=fit_res.y_pred_train,
                    y_test_true=y_test,
                    y_test_pred=fit_res.y_pred_test,
                    expr_gt=rec.expr_gt,
                    expr_pred=fit_res.expr,
                    feature_names=feature_names,
                )
                
                expr_pred = str(fit_res.expr) if fit_res.expr is not None else ""
                print(f"[PRED] {expr_pred}")
                
                extra = dict(fit_res.metadata) if fit_res.metadata else {}
                extra["run_seed"] = int(run_seed)
                
                row = BenchmarkRow(
                    pid=pid,
                    algo=algo.name,
                    run_id=run_id,
                
                    nlse_test=m.nlse_test,
                    mse_test=m.mse_test,
                    mape_test=m.mape_test,
                
                    nlse_train=m.nlse_train,
                    mse_train=m.mse_train,
                    mape_train=m.mape_train,
                
                    term_precision=m.term_precision,
                    term_recall=m.term_recall,
                    term_f1=m.term_f1,
                
                    expr_str=expr_pred,
                    expr_gt_str=str(rec.expr_gt),
                    extra=extra,
                )

                rows.append(row)

                if results_csv_path is not None:
                    append_results_csv_row(results_csv_path, row)

    return rows

# def save_results_csv(rows: List[BenchmarkRow], path: str) -> None:
#     import csv
#     fieldnames = list(asdict(rows[0]).keys()) if rows else []
#     with open(path, "w", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=fieldnames)
#         w.writeheader()
#         for r in rows:
#             d = asdict(r)
#             if d.get("extra") is not None:
#                 d["extra"] = str(d["extra"])
#             w.writerow(d)


def init_results_csv(path: str, row_type) -> None:
    import csv
    fieldnames = [f.name for f in fields(row_type)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def append_results_csv_row(path: str, row) -> None:
    import csv
    d = asdict(row)
    if d.get("extra") is not None:
        d["extra"] = str(d["extra"])
    fieldnames = list(d.keys())
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(d)