"""Regressions found driving a live Origin 2021 (9.8) session.

- every COM call must run on one owner thread (Origin COM is thread-bound)
- originpro.NLFit cannot load FDFs on Origin 2021 -> LabTalk NLFit fallback
- legend positions are axis units (log axes included), not percentages
- execute_labtalk must not report success when Origin returns False
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from originlab_mcp.origin_manager import OriginManager
from originlab_mcp.tools.advanced import register_advanced_tools
from originlab_mcp.tools.analysis import fdf_param_names, register_analysis_tools
from originlab_mcp.tools.customize import (
    _position_legend,
    _set_tick_style_commands,
    lt_line_width,
)
from originlab_mcp.utils.com_owner import ComOwner
from tests.fakes import DummyMCP, FakeOrigin, attach_fake_origin


def _live_manager(origin: FakeOrigin | None = None) -> OriginManager:
    """Manager with the production execute() path (no fake dispatch patch)."""
    manager = OriginManager(auto_recover_active=False, idle_timeout=0)
    manager._op = origin or FakeOrigin()
    manager._connected = True
    return manager


class TestComOwnerThread:
    def test_calls_from_many_threads_run_on_one_thread(self) -> None:
        owner = ComOwner()
        seen: set[int] = set()
        lock = threading.Lock()

        def record() -> None:
            with lock:
                seen.add(threading.get_ident())

        callers = [threading.Thread(target=owner.call, args=(record,)) for _ in range(8)]
        for t in callers:
            t.start()
        for t in callers:
            t.join()
        owner.shutdown()
        assert len(seen) == 1
        assert threading.get_ident() not in seen

    def test_call_on_owner_thread_runs_inline(self) -> None:
        owner = ComOwner()
        # A nested call would deadlock on a single worker if it were queued.
        assert owner.call(lambda: owner.call(lambda: 7), timeout=2) == 7
        owner.shutdown()

    @pytest.mark.parametrize("budget", ["off", "30"])
    def test_execute_uses_same_thread_with_and_without_budget(
        self, monkeypatch: pytest.MonkeyPatch, budget: str
    ) -> None:
        monkeypatch.setenv("ORIGINLAB_MCP_DISPATCH_TIMEOUT", budget)
        manager = _live_manager()
        idents = {manager.execute(lambda op: threading.get_ident()) for _ in range(5)}
        idents |= {
            manager.execute(lambda op: threading.get_ident(), dispatch_timeout=0),
            manager.execute(lambda op: threading.get_ident(), dispatch_timeout=10),
        }
        assert len(idents) == 1
        assert idents == {manager._com_owner.thread_ident}
        manager.shutdown()

    def test_idle_timer_detaches_on_owner_thread(self) -> None:
        origin = FakeOrigin()
        detach_threads: list[int] = []
        origin.detach = lambda: detach_threads.append(threading.get_ident())  # type: ignore[method-assign]
        manager = _live_manager(origin)
        owner_ident = manager.execute(lambda op: threading.get_ident())
        manager._on_idle_timeout()  # what threading.Timer calls
        assert detach_threads == [owner_ident]
        assert manager.is_connected is False

    def test_timed_out_call_does_not_move_work_to_another_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORIGINLAB_MCP_DISPATCH_TIMEOUT", "0.05")
        manager = _live_manager()
        first = manager.execute(lambda op: threading.get_ident())
        with pytest.raises(Exception, match="未响应"):
            manager.execute(lambda op: time.sleep(0.3))
        time.sleep(0.35)  # let the wedged call finish
        assert manager.execute(lambda op: threading.get_ident()) == first


FDF_EXPDEC2 = """[General Information]
Function Name=ExpDec2
[Fitting Parameters]
Names=y0,A1,t1,A2,t2
Initial Values=0(V),1(V),1(V),1(V),1(V)
"""


class FallbackOrigin(FakeOrigin):
    """Origin 2021: originpro.NLFit rejects every function."""

    def __init__(self, exe_dir) -> None:
        super().__init__()
        self.exe_path = str(exe_dir) + "\\"
        self.user_path = str(exe_dir / "user") + "\\"
        self.lt_floats.update({
            "_MCP_NLFIT_TREE.y0": 0.0, "_MCP_NLFIT_TREE.f_y0": 1.0,
            "_MCP_NLFIT_TREE.A1": -1.7, "_MCP_NLFIT_TREE.e_A1": 0.1,
            "_MCP_NLFIT_TREE.t1": 1.84, "_MCP_NLFIT_TREE.e_t1": 0.14,
            "_MCP_NLFIT_TREE.A2": -4.2, "_MCP_NLFIT_TREE.e_A2": 0.05,
            "_MCP_NLFIT_TREE.t2": 321.0, "_MCP_NLFIT_TREE.e_t2": 6.0,
            "_MCP_NLFIT_TREE.adjr": 0.997,
        })

    def NLFit(self, function_name: str):  # noqa: N802 - mirrors originpro
        raise ValueError(f"Invalid fitting function: {function_name}")

    def lt_float(self, name: str) -> float:
        return float(self.lt_floats.get(name, float("nan")))

    def lt_exec(self, command: str) -> bool:           # real originpro returns True on success
        self.lt_commands.append(command)
        return True


@pytest.fixture
def fallback_origin(tmp_path, fresh_manager):
    (tmp_path / "FitFunc").mkdir()
    (tmp_path / "FitFunc" / "EXPDEC2.FDF").write_text(FDF_EXPDEC2, encoding="utf-8")
    origin = FallbackOrigin(tmp_path)
    attach_fake_origin(fresh_manager, origin)
    sheet = origin.new_sheet()
    sheet.from_list(0, [0.3, 1, 10, 100])
    sheet.from_list(1, [-5.0, -4.6, -4.0, -3.0])
    fresh_manager.active_worksheet = "[Book1]Sheet1"
    return origin


class TestNonlinearFitFallback:
    def _tools(self, manager):
        mcp = DummyMCP()
        register_analysis_tools(mcp, manager)
        return mcp.tools

    def test_fdf_found_by_function_name_when_file_name_differs(self, tmp_path) -> None:
        # Origin ships Allometric1 as ALLOMET1.FDF, with an inline "; comment".
        (tmp_path / "FitFunc").mkdir()
        (tmp_path / "FitFunc" / "ALLOMET1.FDF").write_text(
            "[General Information]\r\nFunction Name=Allometric1\t; name shown in the dialog\r\n"
            "[Fitting Parameters]\r\nNames=a,b\r\n",
            encoding="utf-8",
        )
        assert fdf_param_names(FallbackOrigin(tmp_path), "Allometric1") == ["a", "b"]
        assert fdf_param_names(FallbackOrigin(tmp_path), "NoSuchFunction") is None

    def test_fdf_section_names_are_case_insensitive(self, tmp_path) -> None:
        # Origin 2021 ships EXPDEC1.FDF with "[FITTING PARAMETERS]".
        (tmp_path / "FitFunc").mkdir()
        (tmp_path / "FitFunc" / "EXPDEC1.FDF").write_text(
            "[GENERAL INFORMATION]\r\nFunction Name=ExpDec1\r\n"
            "[FITTING PARAMETERS]\r\nNames=y0,A1,t1\r\n",
            encoding="utf-8",
        )
        assert fdf_param_names(FallbackOrigin(tmp_path), "ExpDec1") == ["y0", "A1", "t1"]

    def test_labtalk_fallback_fits_builtin_function(self, fresh_manager, fallback_origin):
        result = self._tools(fresh_manager)["nonlinear_fit"](
            function_name="ExpDec2", x_col=0, y_col=1,
            initial_params={"t1": 2, "t2": 300}, fixed_params={"y0": 0},
        )
        assert result["ok"] is True, result
        data = result["data"]
        assert data["engine"] == "labtalk_nlfit"
        assert data["parameters"]["t1"] == {"value": 1.84, "error": 0.14}
        assert data["parameters"]["t2"]["value"] == 321.0
        assert data["result"]["f_y0"] == 1
        assert data["result"]["adjr"] == 0.997
        fit_cmd = next(c for c in fallback_origin.lt_commands if c.startswith("nlbegin"))
        assert "!(1,2) func:=ExpDec2 nltree:=_MCP_NLFIT_TREE" in fit_cmd
        assert "_MCP_NLFIT_TREE.t1=2.0" in fit_cmd
        assert "_MCP_NLFIT_TREE.f_y0=1" in fit_cmd
        assert fit_cmd.endswith("nlfit;")
        assert any(c.startswith("nlend") for c in fallback_origin.lt_commands)
        assert any("LabTalk" in w for w in result["warnings"])

    def test_unknown_function_is_still_not_found(self, fresh_manager, fallback_origin):
        result = self._tools(fresh_manager)["nonlinear_fit"](
            function_name="NotARealFunction", x_col=0, y_col=1,
        )
        assert result["ok"] is False
        assert result["error"]["type"] == "not_found"
        assert result["error"]["target"] == "function_name"

    def test_unknown_parameter_rejected_before_fitting(self, fresh_manager, fallback_origin):
        result = self._tools(fresh_manager)["nonlinear_fit"](
            function_name="ExpDec2", x_col=0, y_col=1, initial_params={"bogus": 1},
        )
        assert result["ok"] is False
        assert result["error"]["type"] == "invalid_input"
        assert not any(c.startswith("nlbegin") for c in fallback_origin.lt_commands)


class _LegendOp:
    def __init__(self, floats: dict[str, float]) -> None:
        self.floats = floats
        self.commands: list[str] = []

    def lt_float(self, expr: str) -> float:
        return self.floats.get(expr, float("nan"))

    def lt_exec(self, command: str) -> None:
        self.commands.append(command)


class TestLegendPlacement:
    def test_log_axis_placement_in_log_space(self) -> None:
        op = _LegendOp({
            "layer.x.from": 0.1, "layer.x.to": 1000.0, "layer.x.type": 2.0,  # log10
            "layer.y.from": -7.0, "layer.y.to": 4.5, "layer.y.type": 1.0,
            # legend spans 10..100 (1 of 4 decades) and 0..1.15 (10 % of y)
            "legend.x": 10 ** 1.5, "legend.x1": 10.0,
            "legend.y": 0.575, "legend.y1": 1.15,
        })
        placed = _position_legend(op, "top_right")
        assert placed is not None
        # right edge 3 % inside: log10 centre = -1 + 4 * (1 - 0.03 - 0.125)
        assert placed["x"] == pytest.approx(10 ** (-1 + 4 * 0.845))
        # top edge 3 % inside: centre = -7 + 11.5 * (1 - 0.03 - 0.05)
        assert placed["y"] == pytest.approx(-7 + 11.5 * 0.92)
        assert op.commands[0].startswith("legend.x = ")

    def test_missing_axis_info_leaves_legend_untouched(self) -> None:
        op = _LegendOp({})
        assert _position_legend(op, "bottom_left") is None
        assert op.commands == []


class TestUnits:
    @pytest.mark.parametrize(
        ("direction", "show_minor", "expected"),
        [("in", True, 5), ("out", True, 10), ("both", True, 15), ("in", False, 1), ("out", False, 2)],
    )
    def test_tick_bit_mask_includes_minor_ticks(self, direction, show_minor, expected) -> None:
        op = _LegendOp({})
        _set_tick_style_commands(op, "Graph1", 0, direction, 3, 1, show_minor)
        assert f"layer.x.ticks = {expected}" in op.commands
        assert f"layer.y.minorticks = {1 if show_minor else 0}" in op.commands
        assert not any(c.startswith("layer.x.minor =") for c in op.commands)

    def test_line_width_points_to_labtalk_units(self) -> None:
        assert lt_line_width(1) == 500
        assert lt_line_width(0.5) == 250
        assert lt_line_width(0.0001) == 1


class TestExecuteLabTalkFailure:
    def test_false_result_is_reported_as_error(self, fresh_manager, fake_origin) -> None:
        mcp = DummyMCP()
        register_advanced_tools(mcp, fresh_manager)
        fake_origin.lt_exec = lambda command: False  # type: ignore[method-assign]
        result = mcp.tools["execute_labtalk"](command="win -a NoSuchGraph")
        assert result["ok"] is False
        assert result["error"]["target"] == "command"

    def test_none_result_still_succeeds(self, fresh_manager, fake_origin) -> None:
        mcp = DummyMCP()
        register_advanced_tools(mcp, fresh_manager)
        assert mcp.tools["execute_labtalk"](command="type -a hi")["ok"] is True


def test_atexit_shutdown_still_detaches_on_owner_thread(tmp_path) -> None:
    """Interpreter-exit shutdown must still reach the owner thread and detach."""
    script = tmp_path / "exit_probe.py"
    script.write_text(
        "import atexit, threading\n"
        "from originlab_mcp.origin_manager import OriginManager\n"
        "from tests.fakes import FakeOrigin\n"
        "m = OriginManager(auto_recover_active=False, idle_timeout=0)\n"
        "o = FakeOrigin()\n"
        "o.detach = lambda: print('detached-on-owner', threading.current_thread().name, flush=True)\n"
        "m._op = o; m._connected = True\n"
        "m.execute(lambda op: None)\n"
        "atexit.register(m.shutdown)\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": f"{root / 'src'}{os.pathsep}{root}"}
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=30, env=env
    )
    assert proc.returncode == 0, proc.stderr
    assert "detached-on-owner origin-com" in proc.stdout
    assert "Traceback" not in proc.stderr


class TestRemoveMissingLabel:
    def test_missing_label_is_not_found(self, fresh_manager, fake_origin) -> None:
        from originlab_mcp.tools.customize import register_customize_tools
        from tests.fakes.origin import FakeLayer

        mcp = DummyMCP()
        register_customize_tools(mcp, fresh_manager)
        fake_origin.new_graph()
        FakeLayer.label = lambda self, name: None          # originpro returns None for unknown names
        try:
            result = mcp.tools["remove_graph_label"](label_name="Legend", graph_name="Graph1")
        finally:
            del FakeLayer.label
        assert result["ok"] is False
        assert result["error"]["type"] == "not_found"


class TestLinearFitFallback:
    def test_fitlr_used_when_linearfit_result_fails(self, tmp_path, fresh_manager) -> None:
        from tests.fakes.origin import FakeLinearFit

        class BrokenLinearFit(FakeLinearFit):
            __slots__ = ()

            def result(self):  # Origin 2021: originpro cannot parse the result XML
                raise ValueError("unclosed token: line 3, column 3946")

        origin = FallbackOrigin(tmp_path)
        origin.LinearFit = lambda: BrokenLinearFit()  # type: ignore[method-assign]
        origin.lt_floats.update({"mcplra": -12.5, "mcplrb": 13.9, "mcplrea": 0.2, "mcplreb": 0.1,
                                 "mcplrr": 0.999, "mcplrn": 9.0})
        attach_fake_origin(fresh_manager, origin)
        sheet = origin.new_sheet()
        sheet.from_list(0, [0.8, 0.85, 0.9])
        sheet.from_list(1, [-1.4, -0.7, 0.0])
        fresh_manager.active_worksheet = "[Book1]Sheet1"
        mcp = DummyMCP()
        register_analysis_tools(mcp, fresh_manager)
        result = mcp.tools["linear_fit"](x_col=0, y_col=1, fix_intercept=0)
        assert result["ok"] is True, result
        assert result["data"]["engine"] == "labtalk_fitlr"
        assert result["data"]["parameters"]["Slope"] == {"value": 13.9, "error": 0.1}
        cmd = next(c for c in origin.lt_commands if "fitlr" in c)
        assert "fixint:=1 intercept:=0.0" in cmd and "!(1,2)" in cmd
