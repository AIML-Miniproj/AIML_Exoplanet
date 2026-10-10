"""Sanity checks on the synthetic light-curve generators themselves."""

import numpy as np
import pytest


@pytest.mark.parametrize("factory_name", ["planet_factory", "eb_factory",
                                          "variable_factory", "noise_factory"])
def test_generators_produce_multi_sector_collections(request, factory_name):
    make = request.getfixturevalue(factory_name)
    coll = make(n_sectors=3)
    assert len(coll) == 3
    assert [lc.meta["SECTOR"] for lc in coll] == [1, 2, 3]
    for lc in coll:
        t = lc.time.value
        span = t[-1] - t[0]
        assert 25 < span < 27.5
        # mid-sector downlink gap of about a day
        assert np.max(np.diff(t)) > 0.8
        assert np.any(lc["quality"] > 0)
    # raw flux level differs between sectors
    medians = [np.nanmedian(lc.flux.value) for lc in coll]
    assert np.std(medians) > 0


def test_planet_dip_has_requested_depth(planet_factory):
    coll = planet_factory(depth=0.01, noise=1.0, n_sectors=1)
    lc = coll[0]
    rel = lc.flux.value / np.median(lc.flux.value)
    assert rel.min() == pytest.approx(1 - 0.01, abs=5e-4)


def test_eb_has_primary_and_secondary(eb_factory):
    coll = eb_factory(depth1=0.2, depth2=0.06, noise=1.0, n_sectors=1)
    rel = coll[0].flux.value / np.median(coll[0].flux.value)
    assert rel.min() < 0.85          # deep V-shaped primary
    assert np.sum((rel < 0.97) & (rel > 0.93)) > 0   # secondary-depth points exist


def test_variable_kinds(variable_factory):
    for kind in ("sinusoid", "multiperiodic", "spotted"):
        coll = variable_factory(kind=kind, amp=0.01, noise=1.0, n_sectors=1)
        rel = coll[0].flux.value / np.median(coll[0].flux.value)
        assert np.ptp(rel) > 0.01
    with pytest.raises(ValueError):
        variable_factory(kind="nope")


def test_noise_has_spikes(noise_factory):
    coll = noise_factory(n_sectors=1, n_spikes=10)
    rel = coll[0].flux.value / np.median(coll[0].flux.value)
    assert np.sum(np.abs(rel - 1) > 0.008) >= 5
