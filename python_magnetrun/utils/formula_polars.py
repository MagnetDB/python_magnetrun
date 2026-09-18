"""Shared arithmetic-formula-to-Polars-expression evaluator.

Translates a ``target = expr`` formula's right-hand side into a
:class:`polars.Expr`, for use by ``addData()`` on both
:class:`~python_magnetrun.magnetdata_polars.PolarsMagnetData` (pupitre) and
:class:`~python_magnetrun.magnetdata_tdms.TdmsMagnetData` (pigbrother) —
factored out here to avoid duplicating the same ~40 lines on both classes.

Supports only ``+ - * /``, unary ``+ -``, parentheses, bare column names,
and numeric literals — the full grammar every real formula in this
codebase's housing-config JSON files actually uses (simple column sums,
e.g. ``"IH = Idcct1 + Idcct2"``, ``"Courants_Alimentations/Référence_GR1 =
Référence_A1 + Référence_A2"``). Function calls (``sqrt``, ``sin``, …) are
intentionally not supported; add them here if a real formula ever needs one.
"""

from __future__ import annotations

import ast
import operator

import polars as pl

_AST_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_AST_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _ast_to_polars_expr(node: ast.AST) -> pl.Expr | int | float:
    """Recursively translate one AST node into a Polars expression or literal."""
    if isinstance(node, ast.Expression):
        return _ast_to_polars_expr(node.body)
    if isinstance(node, ast.BinOp):
        op = _AST_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_ast_to_polars_expr(node.left), _ast_to_polars_expr(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _AST_UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_ast_to_polars_expr(node.operand))
    if isinstance(node, ast.Name):
        return pl.col(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def formula_to_polars_expr(rhs: str) -> pl.Expr:
    """Parse a formula's right-hand side into a Polars expression.

    Parameters
    ----------
    rhs : str
        Right-hand side of a ``target = expr`` formula string.

    Returns
    -------
    polars.Expr
        Expression ready for ``df.with_columns(expr.alias(key))``.

    Raises
    ------
    SyntaxError
        If *rhs* is not valid Python expression syntax.
    ValueError
        If *rhs* uses an unsupported operator, function call, or literal.
    """
    tree = ast.parse(rhs.strip(), mode="eval")
    return _ast_to_polars_expr(tree)


__all__ = ["formula_to_polars_expr"]
