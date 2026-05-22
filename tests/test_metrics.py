"""Tests for runtime.metrics MetricsRegistry and the Prometheus text format."""
from __future__ import annotations

from runtime.metrics import MetricsRegistry


def test_registry_sets_and_renders_plain_gauge():
    r = MetricsRegistry()
    r.set("equity_usd", 1234.5)
    out = r.render()
    assert "equity_usd 1234.5\n" in out


def test_registry_renders_labelled_gauge():
    r = MetricsRegistry()
    r.set("regime_p_bull", 0.83, labels={"coin": "BTC"})
    out = r.render()
    assert 'regime_p_bull{coin="BTC"} 0.83\n' in out


def test_registry_sorts_metric_names():
    r = MetricsRegistry()
    r.set("z_last", 1.0)
    r.set("a_first", 2.0)
    out = r.render()
    assert out.index("a_first") < out.index("z_last")


def test_registry_set_overwrites_previous_value():
    r = MetricsRegistry()
    r.set("g", 1.0)
    r.set("g", 2.0)
    out = r.render()
    assert "g 2.0\n" in out
    assert "g 1.0" not in out


def test_registry_renders_multiple_labels_sorted():
    r = MetricsRegistry()
    r.set("trades", 5.0, labels={"b": "y", "a": "x"})
    out = r.render()
    # Labels must be alphabetised so the output is stable across runs.
    assert 'trades{a="x",b="y"} 5.0\n' in out


def test_empty_registry_renders_empty_string():
    r = MetricsRegistry()
    assert r.render() == ""


def test_registry_escapes_label_value():
    r = MetricsRegistry()
    r.set("g", 1.0, labels={"k": 'a"b'})
    out = r.render()
    assert 'k="a\\"b"' in out
