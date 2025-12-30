import sympy as sp


def filter_imaginary_part(expr: sp.Expr, real_symbols: list[sp.Symbol]) -> sp.Expr:
    """
    Take a SymPy expression built from complex weights, re(), im(), etc.,
    assume the given symbols are real, and return a purely real expression
    in those symbols (no re(), im(), I in the final result).

    Parameters
    ----------
    expr : sympy.Expr
        Input (possibly complex) expression.
    real_symbols : list of sympy.Symbol
        Variables you consider real, e.g. [x, y].

    Returns
    -------
    sympy.Expr
        Real-valued expression in real_symbols only.
    """
    e = sp.simplify(expr)

    # Expand w.r.t. I so products of complex terms are resolved
    e = sp.expand_complex(e)

    # Enforce that our base symbols are real: re(x)->x, im(x)->0, etc.
    base_repl = {}
    for s in real_symbols:
        base_repl[sp.re(s)] = s
        base_repl[sp.im(s)] = 0
    e = e.xreplace(base_repl)

    # Take the real part (drop any genuine imaginary contribution)
    e = sp.re(e)
    e = sp.simplify(e)

    # NOW: Strip ALL remaining re()/im() wrappers:
    #   - re(anything) -> anything
    #   - im(anything) -> 0
    e = e.replace(
        lambda a: a.func is sp.re,
        lambda a: a.args[0]
    )
    e = e.replace(
        lambda a: a.func is sp.im,
        lambda a: sp.Integer(0)
    )

    # Final cleanup
    e = sp.simplify(e)
    return e

# def prune_small_coeff_terms(expr: sp.Expr, decimals: int) -> sp.Expr:
#     """
#     Recursively remove additive / multiplicative terms whose numeric prefactor
#     rounds to zero at the given number of decimals.

#     Works inside nested expressions (sqrt, pow, div, etc.) and correctly handles
#     scientific notation (e.g. 5e-8).

#     Parameters
#     ----------
#     expr : sympy.Expr
#         Input symbolic expression.
#     decimals : int
#         Number of decimal digits to keep. Coefficients that round to 0 at this
#         precision are pruned.

#     Returns
#     -------
#     sympy.Expr
#         Pruned expression.
#     """
#     expr = sp.sympify(expr)

#     # atoms (symbols, numbers)
#     if expr.is_Atom:
#         return expr

#     # ---------- sums ----------
#     if expr.is_Add:
#         kept = []
#         for term in expr.args:
#             t = prune_small_coeff_terms(term, decimals)
#             if t == 0:
#                 continue

#             c, rest = t.as_coeff_Mul()   # t = c * rest
#             if c.is_number:
#                 try:
#                     # round numerically
#                     c_round = round(float(sp.N(c)), decimals)
#                     if c_round == 0.0:
#                         continue
#                     # rebuild term with rounded coefficient
#                     t = sp.Float(c_round) * rest
#                 except Exception:
#                     pass

#             kept.append(t)

#         return sp.Add(*kept) if kept else sp.Integer(0)

#     # ---------- products ----------
#     if expr.is_Mul:
#         factors = [prune_small_coeff_terms(a, decimals) for a in expr.args]
#         if any(f == 0 for f in factors):
#             return sp.Integer(0)

#         new_expr = sp.Mul(*factors)

#         c, rest = new_expr.as_coeff_Mul()
#         if c.is_number:
#             try:
#                 c_round = round(float(sp.N(c)), decimals)
#                 if c_round == 0.0:
#                     return sp.Integer(0)
#                 return sp.Float(c_round) * rest
#             except Exception:
#                 return new_expr

#         return new_expr

#     # ---------- powers ----------
#     if expr.is_Pow:
#         base = prune_small_coeff_terms(expr.base, decimals)
#         exp  = prune_small_coeff_terms(expr.exp, decimals)
#         return sp.Pow(base, exp)

#     # ---------- generic functions (sqrt, sin, log, etc.) ----------
#     if expr.args:
#         new_args = [prune_small_coeff_terms(a, decimals) for a in expr.args]
#         try:
#             return expr.func(*new_args)
#         except Exception:
#             return expr

#     return expr

def prune_small_coeff_terms(expr: sp.Expr, decimals: int) -> sp.Expr:
    """
    Recursively prune terms whose numeric prefactor rounds to 0 at `decimals`.

    Also prunes *after expanding additive structure* locally:
    - For each Add node, expand it (distribute products into sums), then prune.
    This enables pruning of small coefficients hidden inside products like a*(x+y).

    Parameters
    ----------
    expr : sympy.Expr
        Input symbolic expression.
    decimals : int
        Number of decimal digits to keep. Coefficients that round to 0 at this
        precision are pruned.

    Returns
    -------
    sympy.Expr
        Pruned expression.
    """
    expr = sp.sympify(expr)

    # atoms (symbols, numbers)
    if expr.is_Atom:
        return expr

    # ---------- sums ----------
    if expr.is_Add:
        # Expand only this additive level so hidden terms become visible
        expr_exp = sp.expand(expr)

        kept = []
        for term in expr_exp.as_ordered_terms():
            t = prune_small_coeff_terms(term, decimals)
            if t == 0:
                continue

            c, rest = t.as_coeff_Mul()
            if c.is_number:
                try:
                    c_round = round(float(sp.N(c)), decimals)
                    if c_round == 0.0:
                        continue
                    t = sp.Float(c_round) * rest
                except Exception:
                    pass

            kept.append(t)

        return sp.Add(*kept) if kept else sp.Integer(0)

    # ---------- products ----------
    if expr.is_Mul:
        factors = [prune_small_coeff_terms(a, decimals) for a in expr.args]
        if any(f == 0 for f in factors):
            return sp.Integer(0)

        new_expr = sp.Mul(*factors)

        c, rest = new_expr.as_coeff_Mul()
        if c.is_number:
            try:
                c_round = round(float(sp.N(c)), decimals)
                if c_round == 0.0:
                    return sp.Integer(0)
                return sp.Float(c_round) * rest
            except Exception:
                return new_expr

        return new_expr

    # ---------- powers ----------
    if expr.is_Pow:
        base = prune_small_coeff_terms(expr.base, decimals)
        exp  = prune_small_coeff_terms(expr.exp, decimals)
        return sp.Pow(base, exp)

    # ---------- generic functions (sqrt, sin, log, etc.) ----------
    if expr.args:
        new_args = [prune_small_coeff_terms(a, decimals) for a in expr.args]
        try:
            return expr.func(*new_args)
        except Exception:
            return expr

    return expr