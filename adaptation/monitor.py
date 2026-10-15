"""
monitor.py -- the Monitor step of the adaptation loop: one row of live
telemetry per second, in the columns the DBN was trained on.

The model (dbn/build_model.py, pool="local") only uses gauges of the three
services: throughput, avg_p_latency, buffer_size, cores and data_quality. The
counters, process and GC metrics of the dumps are never inputs, so no
counter-to-rate conversion is needed here.

Naming follows data_prep/convert_prom_dump.py: <metric>_<i>, where i is the
service. The converter numbers the series of a metric by their sorted label
signature, which for these metrics is the order linked-service-1, -2, -3; here
the number is read from the `instance` label directly
(adaptation/check_monitor.py checks that both give the same names).

The load is buffer_size_1, exposed as `rps` (dbn/data.py, LOAD_SOURCE_COL):
the rate setpoint as seen by service 1.
"""
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dbn"))
import data as D  # noqa: E402

PROM = "http://localhost:9090"
METRICS = ("throughput", "avg_p_latency", "buffer_size", "cores", "data_quality")
QUERY = '{__name__=~"%s", job="iot_service"}' % "|".join(METRICS)

_INSTANCE_RE = re.compile(r"linked-service-(\d+)")


def variable_name(labels):
    """'throughput_2' for the throughput series of linked-service-2; None if not a model input."""
    metric = labels.get("__name__")
    m = _INSTANCE_RE.search(labels.get("instance", ""))
    if metric not in METRICS or m is None:
        return None
    return f"{metric}_{m.group(1)}"


def row_from_samples(samples):
    """
    samples: iterable of (labels, value) for one second, labels including
    __name__ (the shape of a Prometheus instant-query result).
    Returns {column: value} with `rps` in place of buffer_size_1.
    """
    row = {}
    for labels, value in samples:
        name = variable_name(labels)
        if name is not None:
            row[name] = float(value)
    if D.LOAD_SOURCE_COL in row:
        row[D.LOAD_COL] = row.pop(D.LOAD_SOURCE_COL)
    return row


def query_now(prom=PROM, timeout=2):
    """One row from a Prometheus instant query; also returns the sample timestamp."""
    r = requests.get(f"{prom}/api/v1/query", params={"query": QUERY}, timeout=timeout)
    r.raise_for_status()
    result = r.json()["data"]["result"]
    row = row_from_samples((s["metric"], s["value"][1]) for s in result)
    ts = max((float(s["value"][0]) for s in result), default=time.time())
    return ts, row
