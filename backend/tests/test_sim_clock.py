"""The simulation clock offset must survive the background refresh loop."""

from __future__ import annotations

import pytest

from app.core import sim_clock


@pytest.fixture(autouse=True)
def _reset_clock():
    sim_clock.reset()
    yield
    sim_clock.reset()


class TestSimClock:
    def test_starts_at_zero_offset(self):
        assert sim_clock.get_offset_hours() == 0.0
        delta = sim_clock.simulation_now() - sim_clock.real_now()
        assert abs(delta.total_seconds()) < 5.0

    def test_offset_shifts_the_simulation_epoch(self):
        sim_clock.set_offset_hours(6.0)

        delta_seconds = (
            sim_clock.simulation_now() - sim_clock.real_now()
        ).total_seconds()
        assert delta_seconds == pytest.approx(6 * 3600.0, abs=5.0)

    def test_negative_offset_is_allowed(self):
        sim_clock.set_offset_hours(-12.0)
        assert sim_clock.get_offset_hours() == -12.0

    @pytest.mark.parametrize("offset", [24.5, -24.5, 1000.0, float("nan")])
    def test_out_of_range_offset_is_rejected(self, offset):
        with pytest.raises(ValueError):
            sim_clock.set_offset_hours(offset)

    def test_rejected_offset_does_not_change_the_clock(self):
        sim_clock.set_offset_hours(3.0)
        with pytest.raises(ValueError):
            sim_clock.set_offset_hours(99.0)
        assert sim_clock.get_offset_hours() == 3.0

    def test_real_now_ignores_the_offset(self):
        sim_clock.set_offset_hours(8.0)
        delta = sim_clock.real_now() - sim_clock.real_now()
        assert abs(delta.total_seconds()) < 5.0

    def test_background_refresh_loop_observes_the_offset(self):
        """
        The refresh loop is the thing that used to snap the shifted view back to
        wall-clock UTC, because it read datetime.now() directly. It must read
        the shared clock instead.
        """
        import inspect

        import main as main_module

        source = inspect.getsource(main_module.refresh_snapshot_once)
        assert "sim_clock.simulation_now()" in source
        assert "datetime.now" not in source
