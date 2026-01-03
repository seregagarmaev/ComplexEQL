from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Set, List

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class Metrics:
    nlse: float
    term_precision: float
    term_recall: float
    term_f1: float


def nlse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    var = float(np.var(y_true))
    return float(mse / var) if var > 0 else float("nan")


def _as_add_terms(expr: sp.Expr) -> Tuple[sp.Expr, ...]:
    expr = sp.sympify(expr)
    return sp.Add.make_args(expr) if expr.is_Add else (expr,)


def _strip_numeric_coefficient(term: sp.Expr) -> sp.Expr:
    term = sp.simplify(term)
    if term.is_Number:
        return sp.Integer(1)
    _, rest = term.as_coeff_Mul(rational=False)
    return sp.Integer(1) if rest == 1 else sp.simplify(rest)


def _canonicalize_term(term: sp.Expr) -> sp.Expr:
    t = sp.expand_mul(term)
    t = sp.expand(t)
    t = _strip_numeric_coefficient(t)
    t = sp.together(t)
    t = sp.simplify(t)
    return t


def _round_floats(expr: sp.Expr, decimals: int = 12) -> sp.Expr:
    repl = {}
    for f in expr.atoms(sp.Float):
        repl[f] = sp.Float(round(float(f), decimals))
    return expr.xreplace(repl)


def _canonicalize_symbols(expr: sp.Expr, feature_names: List[str]) -> sp.Expr:
    name_set = set(feature_names)
    repl = {}
    for s in expr.free_symbols:
        if s.name in name_set:
            repl[s] = sp.Symbol(s.name)
    return expr.xreplace(repl)


def _normalize_for_terms(expr: sp.Expr, feature_names: List[str], float_decimals: int = 12) -> sp.Expr:
    e = sp.sympify(expr)
    e = _canonicalize_symbols(e, feature_names)
    e = sp.expand_mul(e)
    e = sp.expand(e)
    e = sp.together(e)
    e = sp.simplify(e)
    e = _round_floats(e, decimals=float_decimals)
    e = sp.expand_mul(e)
    e = sp.expand(e)
    e = sp.simplify(e)
    return e


def extract_term_set(expr: sp.Expr, feature_names: List[str], float_decimals: int = 12) -> Set[sp.Expr]:
    e = _normalize_for_terms(expr, feature_names=feature_names, float_decimals=float_decimals)
    terms = _as_add_terms(e)

    const_sum = sp.Integer(0)
    nonconst_terms: List[sp.Expr] = []
    feat_name_set = set(feature_names)

    for t in terms:
        sym_names = {s.name for s in t.free_symbols}
        if sym_names.isdisjoint(feat_name_set):
            const_sum += t
        else:
            nonconst_terms.append(t)

    out: Set[sp.Expr] = set()
    const_sum = sp.simplify(const_sum)
    if const_sum != 0:
        out.add(sp.Integer(1))

    for t in nonconst_terms:
        ct = _canonicalize_term(t)
        for subt in _as_add_terms(ct):
            out.add(_canonicalize_term(subt))

    return out


def term_precision_recall_f1(
    expr_gt: sp.Expr, expr_pred: Optional[sp.Expr], feature_names: List[str]
) -> Tuple[float, float, float]:
    if expr_pred is None:
        return 0.0, 0.0, 0.0

    gt_terms = extract_term_set(expr_gt, feature_names, 12)
    pr_terms = extract_term_set(expr_pred, feature_names, 12)

    if len(pr_terms) == 0 and len(gt_terms) == 0:
        return 1.0, 1.0, 1.0
    if len(pr_terms) == 0:
        return 0.0, 0.0, 0.0

    inter = gt_terms.intersection(pr_terms)
    precision = len(inter) / len(pr_terms)
    recall = len(inter) / len(gt_terms)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    expr_gt: sp.Expr,
    expr_pred: Optional[sp.Expr],
    feature_names: List[str],
) -> Metrics:
    p, r, f1 = term_precision_recall_f1(expr_gt, expr_pred, feature_names)
    return Metrics(nlse=nlse(y_true, y_pred), term_precision=p, term_recall=r, term_f1=f1)