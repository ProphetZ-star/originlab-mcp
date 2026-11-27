"""数据分析类 tools。

让 AI 能利用 Origin 内置的拟合引擎进行数据分析。

包含:
    linear_fit: 线性拟合
    nonlinear_fit: 非线性拟合
    list_fit_functions: 列出可用拟合函数
"""

from __future__ import annotations

import configparser
import math
import re
from contextlib import suppress
from pathlib import Path
from typing import Any

from originlab_mcp.exceptions import (
    FitConvergenceError,
    ToolError,
)
from originlab_mcp.utils.annotations import MUTATING, READ_ONLY
from originlab_mcp.utils.constants import COMMON_FIT_FUNCTIONS
from originlab_mcp.utils.helpers import (
    find_worksheet as _find_worksheet,
)
from originlab_mcp.utils.helpers import (
    resolve_worksheet_name,
    tool_error_handler,
)
from originlab_mcp.utils.validators import (
    success_response,
)

# ---------------------------------------------------------------------------
# LabTalk NLFit fallback
#
# originpro.NLFit(func) reads the function definition through the LabTalk
# function GetFDFAsXML(), which returns an empty string on Origin 2021 (9.8),
# so every built-in function is rejected as "Invalid fitting function". The
# fallback drives the same Origin fitting engine through nlbegin/nlfit/nlend
# and reads parameter names from the function's .FDF file.
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NLFIT_TREE = "_MCP_NLFIT_TREE"
_NLFIT_STATS = ("niter", "dof", "ssr", "chisqr", "cod", "adjr")


def _fit_function_dirs(op: Any) -> list[Path]:
    dirs: list[Path] = []
    for kind in ("u", "e"):  # user functions shadow built-ins, as in Origin
        with suppress(Exception):
            raw = op.path(kind)
            if raw:
                base = Path(raw)
                if base.suffix.lower() == ".exe":
                    base = base.parent
                dirs.append(base / "FitFunc")
    return dirs


def _parse_fdf(path: Path) -> tuple[str, list[str]]:
    """(Function Name, parameter names) from an .FDF (INI) file."""
    # FDFs carry inline "; comments"; section case varies ("[Fitting Parameters]"
    # vs "[FITTING PARAMETERS]"); option names are lower-cased by default.
    parser = configparser.ConfigParser(strict=False, interpolation=None, inline_comment_prefixes=(";",))
    with suppress(configparser.Error):
        parser.read_string(path.read_text(encoding="utf-8", errors="ignore"))

    def section(name: str) -> str | None:
        return next((s for s in parser.sections() if s.strip().lower() == name), None)

    info, params = section("general information"), section("fitting parameters")
    func = parser.get(info, "function name", fallback="").strip() if info else ""
    names = parser.get(params, "names", fallback="") if params else ""
    return func, [n.strip() for n in names.split(",") if n.strip()]


def fdf_param_names(op: Any, function_name: str) -> list[str] | None:
    """Parameter names of a fit function from its .FDF file; None if not found.

    Built-in FDFs often use 8.3 file names (Allometric1 -> ALLOMET1.FDF), so when
    ``<name>.FDF`` does not exist the folder is searched by "Function Name=".
    """
    if not _IDENT_RE.match(function_name):
        return None
    wanted = function_name.lower()
    for folder in _fit_function_dirs(op):
        for candidate in (folder / f"{function_name}.FDF", folder / f"{function_name.upper()}.FDF"):
            if candidate.is_file():
                return _parse_fdf(candidate)[1]
        if folder.is_dir():
            for candidate in sorted(folder.glob("*")):
                if candidate.suffix.lower() == ".fdf" and candidate.is_file():
                    func, names = _parse_fdf(candidate)
                    if func.lower() == wanted:
                        return names
    return None


def _sheet_range(wks: Any) -> str:
    with suppress(Exception):
        rng = wks.lt_range()
        if rng:
            return str(rng)
    return f"[{wks.get_book().name}]{wks.name}"


def _lt_number(op: Any, expr: str) -> float | None:
    with suppress(Exception):
        value = float(op.lt_float(expr))
        if not math.isnan(value):
            return value
    return None


def labtalk_linfit(
    op: Any,
    wks: Any,
    sheet_name: str,
    x_col: int,
    y_col: int,
    fix_slope: float | None,
    fix_intercept: float | None,
) -> dict[str, Any]:
    """Linear fit with the fitlr X-Function (used when originpro.LinearFit fails)."""
    opts = []
    if fix_intercept is not None:
        opts += ["fixint:=1", f"intercept:={float(fix_intercept)}"]
    if fix_slope is not None:
        opts += ["fixslope:=1", f"slope:={float(fix_slope)}"]
    names = {"a": "mcplra", "b": "mcplrb", "aerr": "mcplrea", "berr": "mcplreb", "r": "mcplrr", "N": "mcplrn"}
    # outputs start as missing values (0/0) so a fit that silently did nothing cannot read as 0
    init = " ".join(f"{v}=0/0;" for v in names.values())
    outs = " ".join(f"{k}:={v}" for k, v in names.items())
    cmd = f"{init} fitlr iy:={_sheet_range(wks)}!({x_col + 1},{y_col + 1}) {' '.join(opts)} {outs};"
    if op.lt_exec(cmd) is not True:                  # originpro.lt_exec returns True on success
        raise RuntimeError("LabTalk fitlr 执行失败")
    val = {k: _lt_number(op, v) for k, v in names.items()}
    if val["a"] is None or val["b"] is None:
        raise RuntimeError("fitlr 未返回斜率/截距")
    stats: dict[str, Any] = {}
    if val["r"] is not None:
        stats["R"] = val["r"]
        stats["R2"] = val["r"] ** 2
    if val["N"] is not None:
        stats["N"] = val["N"]
    result: dict[str, Any] = {
        "sheet_name": sheet_name,
        "x_col": x_col,
        "y_col": y_col,
        "method": "linear",
        "engine": "labtalk_fitlr",
        "parameters": {
            "Intercept": {"value": val["a"], "error": val["aerr"]},
            "Slope": {"value": val["b"], "error": val["berr"]},
        },
    }
    if stats:
        result["statistics"] = stats
    if fix_slope is not None:
        result["fixed_slope"] = fix_slope
    if fix_intercept is not None:
        result["fixed_intercept"] = fix_intercept
    return result


def labtalk_nlfit(
    op: Any,
    wks: Any,
    function_name: str,
    param_names: list[str],
    x_col: int,
    y_col: int,
    yerr_col: int | None,
    initial_params: dict | None,
    fixed_params: dict | None,
) -> dict[str, Any]:
    """Fit with LabTalk nlbegin/nlfit and return a flat nltree-like dict."""
    if not param_names:
        raise ToolError(
            f"无法从 '{function_name}' 的 FDF 文件读取参数名",
            error_type="unsupported",
            target="function_name",
            value=function_name,
            hint="请确认该拟合函数的 .FDF 文件包含 [Fitting Parameters] Names=...。",
        )
    known = set(param_names)
    for name in list(initial_params or {}) + list(fixed_params or {}):
        if not _IDENT_RE.match(str(name)) or (known and name not in known):
            raise ToolError(
                f"拟合函数 '{function_name}' 没有参数 '{name}'",
                error_type="invalid_input",
                target="initial_params" if name in (initial_params or {}) else "fixed_params",
                value=name,
                hint=f"可用参数: {param_names}",
            )

    tr = _NLFIT_TREE
    cols = f"{x_col + 1},{y_col + 1}" + (f",{yerr_col + 1}" if yerr_col is not None else "")
    cmds = [f"nlbegin iy:={_sheet_range(wks)}!({cols}) func:={function_name} nltree:={tr}"]
    for name, value in (initial_params or {}).items():
        cmds.append(f"{tr}.{name}={float(value)}")
    for name, value in (fixed_params or {}).items():
        if value is False:
            cmds.append(f"{tr}.f_{name}=0")
        else:
            cmds += [f"{tr}.{name}={float(value)}", f"{tr}.f_{name}=1"]
    cmds.append("nlfit")

    try:
        if op.lt_exec("; ".join(cmds) + ";") is False:
            raise FitConvergenceError(function_name, "LabTalk nlfit 执行失败")
        result: dict[str, Any] = {}
        for name in param_names:
            value = _lt_number(op, f"{tr}.{name}")
            if value is None:
                raise FitConvergenceError(function_name, f"参数 {name} 无有效结果")
            result[name] = value
            err = _lt_number(op, f"{tr}.e_{name}")
            if err is not None:
                result[f"e_{name}"] = err
            fixed = _lt_number(op, f"{tr}.f_{name}")
            result[f"f_{name}"] = int(bool(fixed))
        for stat in _NLFIT_STATS:
            value = _lt_number(op, f"{tr}.{stat}")
            if value is not None:
                result[stat] = value
        return result
    finally:
        with suppress(Exception):
            op.lt_exec(f"nlend; del -vt {tr};")


def register_analysis_tools(mcp: Any, manager: Any) -> None:
    """注册数据分析类 tools 到 MCP Server。

    Args:
        mcp: FastMCP 实例。
        manager: OriginManager 实例（依赖注入）。
    """

    # =================================================================
    # linear_fit
    # =================================================================

    @mcp.tool(annotations=MUTATING)
    @tool_error_handler("线性拟合", "请检查列索引和数据是否有效。")
    def linear_fit(
        x_col: int,
        y_col: int,
        sheet_name: str | None = None,
        yerr_col: int | None = None,
        fix_slope: float | None = None,
        fix_intercept: float | None = None,
        confidence_band: bool = False,
    ) -> dict:
        """Perform linear fit on worksheet data (y = Intercept + Slope * x).

        When to use: For linear regression analysis to obtain slope and intercept.
        When not to use: For clearly non-linear data, use nonlinear_fit.
        Do not use this tool just to plot raw data — use create_plot first
        when you only need a visualization.

        Default behavior:
        - sheet_name omitted: uses current active worksheet
        - Returns slope, intercept with errors, and R² statistics

        Examples:
        - linear_fit(x_col=0, y_col=1)
        - linear_fit(x_col=0, y_col=1, yerr_col=2, confidence_band=True)
        - linear_fit(x_col=0, y_col=1, fix_intercept=0)
        """
        target_name = resolve_worksheet_name(sheet_name, manager)

        def _fit(op: Any) -> dict[str, Any]:
            wks = _find_worksheet(op, target_name)

            lr = op.LinearFit()

            # 设置数据
            if yerr_col is not None:
                lr.set_data(wks, x_col, y_col, yerr_col)
            else:
                lr.set_data(wks, x_col, y_col)

            # 可选：固定斜率或截距
            if fix_slope is not None:
                lr.fix_slope(fix_slope)
            if fix_intercept is not None:
                lr.fix_intercept(fix_intercept)

            # 执行拟合
            try:
                if confidence_band:
                    band = 3  # 同时生成置信带和预测带
                    r, c = lr.report(band)
                    if not r or not c:
                        raise RuntimeError("Origin 未返回拟合报告或曲线范围")
                    result = lr.result()
                    report_info = {
                        "report_range": str(r) if r else None,
                        "curves_range": str(c) if c else None,
                    }
                else:
                    result = lr.result()
                    report_info = {}
            except Exception as exc:
                # Origin 2021: originpro cannot parse the fit-result XML ("unclosed
                # token"). Plain fits fall back to the fitlr X-Function.
                if confidence_band or yerr_col is not None:
                    raise
                try:
                    return labtalk_linfit(op, wks, target_name, x_col, y_col, fix_slope, fix_intercept)
                except Exception:
                    raise exc from None

            # 提取拟合参数
            fit_result: dict[str, Any] = {
                "sheet_name": target_name,
                "x_col": x_col,
                "y_col": y_col,
                "method": "linear",
            }

            if not isinstance(result, dict) or not result.get("Parameters"):
                raise RuntimeError("Origin 未返回有效的线性拟合参数")
            if result:
                # 提取 Parameters
                params = result.get("Parameters", {})
                fit_result["parameters"] = {}
                for pname, pinfo in params.items():
                    if isinstance(pinfo, dict):
                        fit_result["parameters"][pname] = {
                            "value": pinfo.get("Value"),
                            "error": pinfo.get("Error"),
                        }

                for name in ("Slope", "Intercept"):
                    if fit_result["parameters"].get(name, {}).get("value") is None:
                        raise RuntimeError(f"Origin 拟合结果缺少 {name}")

                # 提取统计量
                stats = result.get("Statistics", {})
                if stats:
                    fit_result["statistics"] = stats

            if report_info:
                fit_result["report"] = report_info

            if fix_slope is not None:
                fit_result["fixed_slope"] = fix_slope
            if fix_intercept is not None:
                fit_result["fixed_intercept"] = fix_intercept

            return fit_result

        result = manager.execute(_fit)

        # 构建消息
        params = result.get("parameters", {})
        slope_info = params.get("Slope", {})
        intercept_info = params.get("Intercept", {})
        msg_parts = ["线性拟合完成。"]
        if slope_info.get("value") is not None:
            msg_parts.append(
                f"Slope = {slope_info['value']:.6g}"
                + (f" ± {slope_info['error']:.6g}" if slope_info.get('error') else "")
            )
        if intercept_info.get("value") is not None:
            msg_parts.append(
                f"Intercept = {intercept_info['value']:.6g}"
                + (f" ± {intercept_info['error']:.6g}" if intercept_info.get('error') else "")
            )

        return success_response(
            message=" ".join(msg_parts),
            data=result,
            resource=manager.get_resource_context(),
            next_suggestions=[
                "create_plot",
                "export_graph",
                "nonlinear_fit",
            ],
        )

    # =================================================================
    # nonlinear_fit
    # =================================================================

    @mcp.tool(annotations=MUTATING)
    @tool_error_handler("非线性拟合", "请检查函数名、列索引和数据是否有效。调用 list_fit_functions 查看可用函数。")
    def nonlinear_fit(
        function_name: str,
        x_col: int,
        y_col: int,
        sheet_name: str | None = None,
        yerr_col: int | None = None,
        initial_params: dict | None = None,
        fixed_params: dict | None = None,
        generate_report: bool = False,
    ) -> dict:
        """Perform nonlinear curve fitting on worksheet data.

        When to use: To fit data with specific functions (e.g. Gauss, Lorentz, ExpDec1).
        When not to use: For simple linear relationships, use linear_fit.

        Default behavior:
        - sheet_name omitted: uses current active worksheet
        - function_name must be an Origin built-in fit function name

        Parameter notes:
        - function_name: Origin built-in function name (e.g. "Gauss", "Lorentz", "ExpDec1", "Boltzmann")
        - initial_params: optional initial parameter values dict (e.g. {"xc": 0.5, "w": 1.0})
        - fixed_params: optional fixed parameter dict (e.g. {"y0": 0}). Value of false unfixes the parameter

        Examples:
        - nonlinear_fit(function_name="Gauss", x_col=0, y_col=1)
        - nonlinear_fit(function_name="ExpDec1", x_col=0, y_col=1, initial_params={"t1": 5.0})
        - nonlinear_fit(function_name="Gauss", x_col=0, y_col=1, fixed_params={"y0": 0})
        """
        target_name = resolve_worksheet_name(sheet_name, manager)

        def _fit(op: Any) -> dict[str, Any]:
            wks = _find_worksheet(op, target_name)

            try:
                model = op.NLFit(function_name)
            except Exception as e:
                # Origin 2021: originpro.NLFit cannot read FDFs (GetFDFAsXML
                # returns ""). Use LabTalk NLFit when the function exists.
                param_names = fdf_param_names(op, function_name)
                if param_names is None:
                    raise ToolError(
                        f"无法创建拟合模型 '{function_name}': {e}",
                        error_type="not_found",
                        target="function_name",
                        value=function_name,
                        hint="请调用 list_fit_functions 查看常用函数名，或确认 Origin 中已安装该拟合函数。",
                    ) from e
                result = labtalk_nlfit(
                    op, wks, function_name, param_names, x_col, y_col,
                    yerr_col, initial_params, fixed_params,
                )
                return {
                    "sheet_name": target_name,
                    "x_col": x_col,
                    "y_col": y_col,
                    "function_name": function_name,
                    "engine": "labtalk_nlfit",
                    "result": result,
                    "parameters": {
                        name: {"value": result[name], "error": result.get(f"e_{name}")}
                        for name in param_names
                    },
                    **({"report": None} if generate_report else {}),
                }

            # 设置数据
            if yerr_col is not None:
                model.set_data(wks, x_col, y_col, yerr=yerr_col)
            else:
                model.set_data(wks, x_col, y_col)

            # 设置初始参数
            if initial_params:
                for pname, pval in initial_params.items():
                    model.set_param(pname, pval)

            # 固定参数
            if fixed_params:
                for pname, pval in fixed_params.items():
                    model.fix_param(pname, pval)

            # 执行拟合
            try:
                model.fit()
            except Exception as e:
                raise FitConvergenceError(function_name, str(e)) from e

            # 获取结果
            fit_result: dict[str, Any] = {
                "sheet_name": target_name,
                "x_col": x_col,
                "y_col": y_col,
                "function_name": function_name,
            }

            if generate_report:
                r, c = model.report()
                if not r or not c:
                    raise RuntimeError("Origin 未返回拟合报告或曲线范围")
                fit_result["report"] = {
                    "report_range": str(r),
                    "curves_range": str(c),
                }

            # originpro 1.1.15 returns the flat nltree. report must precede
            # result when both are requested. Keep the full tree for custom
            # functions and statistics rather than inventing a nested schema.
            result = model.result()
            if not isinstance(result, dict) or not result:
                raise RuntimeError("Origin 未返回非线性拟合结果")
            fit_result["result"] = result
            known = COMMON_FIT_FUNCTIONS.get(function_name)
            names = set(known.params if known else ())
            names.update(initial_params or {})
            names.update(fixed_params or {})
            names.update(key[2:] for key in result if key.startswith("f_"))
            fit_result["parameters"] = {
                name: {"value": result[name], "error": result.get(f"e_{name}")}
                for name in sorted(names)
                if name in result and isinstance(result[name], (int, float))
            }

            return fit_result

        result = manager.execute(_fit)

        # 构建消息
        params = result.get("parameters", {})
        param_strs = []
        for pname, pinfo in params.items():
            val = pinfo.get("value")
            err = pinfo.get("error")
            if val is not None:
                s = f"{pname} = {val:.6g}"
                if err is not None:
                    s += f" ± {err:.6g}"
                param_strs.append(s)

        msg = f"非线性拟合完成（函数: {function_name}）。"
        if param_strs:
            msg += " " + ", ".join(param_strs[:4])
            if len(param_strs) > 4:
                msg += f"...（共 {len(param_strs)} 个参数）"

        warnings: list[str] = []
        if result.get("engine") == "labtalk_nlfit":
            warnings.append(
                "originpro.NLFit 在此 Origin 版本不可用，已改用 LabTalk nlbegin/nlfit（同一拟合引擎）。"
            )
            if generate_report:
                warnings.append("LabTalk 回退路径不生成拟合报告表；report 为 null。")

        return success_response(
            message=msg,
            data=result,
            resource=manager.get_resource_context(),
            warnings=warnings or None,
            next_suggestions=[
                "create_plot",
                "export_graph",
                "linear_fit",
            ],
        )

    # =================================================================
    # list_fit_functions
    # =================================================================

    @mcp.tool(annotations=READ_ONLY)
    def list_fit_functions() -> dict:
        """List commonly used fit functions with parameter descriptions.

        When to use: When unsure which fit function to use; browse available functions and parameter info.
        When not to use: If the fit function name is already known.

        Examples:
        - list_fit_functions()
        """
        functions = []
        for fname, finfo in COMMON_FIT_FUNCTIONS.items():
            functions.append({
                "name": fname,
                "parameters": list(finfo.params),
                "description": finfo.description,
            })

        return success_response(
            message=f"共有 {len(functions)} 个常用拟合函数可用。Origin 还支持更多内置函数。",
            data={"functions": functions},
            resource=manager.get_resource_context(),
            next_suggestions=["linear_fit", "nonlinear_fit"],
        )
