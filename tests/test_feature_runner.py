"""Time-limited, chunked feature-build runner (no network: fake per-star workers)."""

import json
import os
import sys
import time
from dataclasses import replace

import pytest

from config import CONFIG, TRAINING_PRESET
from src.triage import dataset


def _row(tic):
    return {"tic_id": tic, "label": "EB", "source": "test", "catalog_period": 1.0}


# Workers must live at module level so a spawned process can import them.
def fast_worker(row, preset, cfg, out_dir):
    out = {"tic_id": row["tic_id"], "label": row["label"], "status": "ok", "reason": "",
           "runtime_s": 0.1}
    (out_dir / f"tic_{row['tic_id']}.json").write_text(json.dumps(out), encoding="utf-8")


def stuck_worker(row, preset, cfg, out_dir):
    (out_dir / f"pid_{row['tic_id']}.txt").write_text(str(os.getpid()))
    time.sleep(120)


def crashing_worker(row, preset, cfg, out_dir):
    os._exit(3)


def mixed_worker(row, preset, cfg, out_dir):
    {1: fast_worker, 2: stuck_worker, 3: crashing_worker}[row["tic_id"]](row, preset, cfg,
                                                                          out_dir)


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259                     # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_timeout_crash_and_success_are_all_recorded(tmp_path):
    rows = [_row(1), _row(2), _row(3)]
    t = time.monotonic()
    res = dataset.run_with_timeout(rows, "training", CONFIG, tmp_path, n_jobs=2,
                                   timeout_s=20, target=mixed_worker)
    assert time.monotonic() - t < 60
    by_tic = {r["tic_id"]: r for r in res}
    assert by_tic[1]["status"] == "ok"
    assert by_tic[2]["status"] == "timeout"
    assert by_tic[3]["status"] == "crashed" and "code 3" in by_tic[3]["reason"]
    # every star has a result file, so a resumed run skips them
    assert {p.name for p in tmp_path.glob("tic_*.json")} == {"tic_1.json", "tic_2.json",
                                                             "tic_3.json"}
    # the killed worker is really gone (not just a launcher process)
    pid = int((tmp_path / "pid_2.txt").read_text())
    assert not _alive(pid)


def test_workers_are_capped():
    assert CONFIG.n_jobs == 2
    assert CONFIG.workers() == 2
    assert CONFIG.workers(8) == CONFIG.max_n_jobs == 3
    assert CONFIG.workers(-1) == 1


def test_training_preset_limits():
    cfg = CONFIG.with_preset("training")
    assert cfg.max_sectors == 2 and cfg.period_max_d == 15.0
    assert cfg.search_bin_minutes == 10.0 and not cfg.use_tls and cfg.max_signals == 1
    assert TRAINING_PRESET.name == "training"
    # UI presets keep their own period range
    assert CONFIG.with_preset("quick").period_max_d == CONFIG.period_max_d


def test_chunked_build_reports_progress(tmp_path, monkeypatch, caplog):
    import logging

    import pandas as pd

    cfg = replace(CONFIG, paths=replace(CONFIG.paths, processed=tmp_path), chunk_size=2)
    real = dataset.run_with_timeout
    monkeypatch.setattr(dataset, "run_with_timeout",
                        lambda rows, *a, **k: real(rows, *a, **{**k, "target": fast_worker}))
    labels = pd.DataFrame([_row(i) for i in range(10, 15)])
    with caplog.at_level(logging.INFO, logger="src.triage.dataset"):
        df = dataset.build_features(labels, cfg=cfg)
    progress = [m for m in caplog.messages if m.startswith("Progress:")]
    assert len(progress) == 3                       # chunks of 2, 2, 1
    assert "5/5 done, 0 failed" in progress[-1] and "ETA" in progress[-1]
    assert len(df) == 5 and (df["status"] == "ok").all()


@pytest.mark.parametrize("n,expected", [(0, 0.0), (6, 3.0)])
def test_estimate_minutes_uses_default_without_history(tmp_path, n, expected):
    cfg = replace(CONFIG, paths=replace(CONFIG.paths, processed=tmp_path))
    assert dataset.estimate_minutes(n, 2, cfg, default_s=60) == pytest.approx(expected)


def test_summary_uses_fap_for_variables_and_reports_catalog_match_separately():
    import pandas as pd

    nan = float("nan")
    rows = [
        # variables: recovered = significant periodogram peak; only one has a catalog period
        dict(label="VARIABLE", status="ok", catalog_period=0.63, recovered=True,
             recovered_ratio=nan, period_match_catalog=True, detected=False, runtime_s=30),
        dict(label="VARIABLE", status="ok", catalog_period=nan, recovered=True,
             recovered_ratio=nan, period_match_catalog=None, detected=False, runtime_s=30),
        dict(label="VARIABLE", status="ok", catalog_period=nan, recovered=False,
             recovered_ratio=nan, period_match_catalog=None, detected=False, runtime_s=30),
        dict(label="EB", status="ok", catalog_period=2.0, recovered=True, recovered_ratio=0.5,
             detected=True, runtime_s=30),
        dict(label="EB", status="timeout", reason="no result within 120 s", catalog_period=2.0,
             runtime_s=120),
        dict(label="NOISE", status="ok", catalog_period=nan, recovered=False,
             recovered_ratio=nan, detected=False, runtime_s=30),
    ]
    s = dataset.summarize(pd.DataFrame(rows).assign(reason=lambda d: d["reason"].fillna("")))
    assert s["VARIABLE"]["recovery_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["VARIABLE"]["n_recovery_checked"] == 3
    assert s["VARIABLE"]["catalog_period_match"] == "1/1"
    assert s["EB"]["recovery_rate"] == 1.0 and s["EB"]["n_failed"] == 1
    assert s["EB"]["recovered_at_ratio"] == {0.5: 1}
    assert s["NOISE"]["recovery_rate"] is None


def test_build_stops_when_a_chunk_fails_too_often(tmp_path, monkeypatch):
    import pandas as pd

    cfg = replace(CONFIG, paths=replace(CONFIG.paths, processed=tmp_path), chunk_size=4)
    real = dataset.run_with_timeout
    monkeypatch.setattr(dataset, "run_with_timeout",
                        lambda rows, *a, **k: real(rows, *a, **{**k, "target": crashing_worker}))
    labels = pd.DataFrame([_row(i) for i in range(20, 28)])
    with pytest.raises(RuntimeError, match="failed in one chunk"):
        dataset.build_features(labels, cfg=cfg)
    # stopped after the first chunk; those results are kept so a re-run resumes
    assert len(list(tmp_path.glob("features/tic_*.json"))) == 4
