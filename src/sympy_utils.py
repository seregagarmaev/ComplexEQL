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