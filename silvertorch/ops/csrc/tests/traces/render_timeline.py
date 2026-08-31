"""Draw a per-stream kernel timeline from a Chrome trace exported by
``test_stream_rwlock_e2e.py``.

This is the static counterpart to opening the trace in https://ui.perfetto.dev:
it keeps only the GPU kernel slices of the three StreamRWLock harness kernels,
puts one row per CUDA stream, and labels each row with the role that stream
played. Everything else in the trace -- the host-side ``cudaLaunchKernel`` rows,
the flow arrows, the profiler's own overhead -- is dropped, which is what makes
the read/read overlap and the read/write exclusion easy to see.

Needs matplotlib, which is a dev-only dependency (not in requirements.txt).

Usage:
    python silvertorch/ops/csrc/tests/traces/render_timeline.py \\
        silvertorch/ops/csrc/tests/traces/stream_rwlock_locked.json \\
        silvertorch/ops/csrc/tests/traces/stream_rwlock_unlocked.json \\
        -o silvertorch/ops/csrc/tests/traces/stream_rwlock_timeline.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Kernel name fragment -> (colour, legend label, stream role). The harness
# kernels live in an anonymous namespace, so their trace names are prefixed with
# "(anonymous namespace)::" -- match on the fragment instead.
KERNELS = {
    "windowed_sum_kernel": ("#2e7d32", "read: windowed_sum_kernel", "reader"),
    "swap_kernel": ("#c62828", "write: swap_kernel", "mutation"),
    "publish_pair_kernel": ("#f9a825", "writer input: publish_pair_kernel", "writer"),
}

ROLE_ORDER = ["reader", "writer", "mutation"]

TITLES = {
    "locked": "WITH StreamRWLock — reads overlap each other, never a write",
    "unlocked": "WITHOUT the lock — writes land in the middle of reads (torn sums)",
}


def kernel_slices(trace_path: str) -> list[dict]:
    """GPU kernel slices of the harness kernels, as {ts, dur, stream, kind}."""
    with open(trace_path) as trace_file:
        events = json.load(trace_file)["traceEvents"]

    slices = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        kind = next((fragment for fragment in KERNELS if fragment in event["name"]), None)
        if kind is None:
            continue
        slices.append(
            {
                "ts": event["ts"],
                "dur": event["dur"],
                "stream": event["args"]["stream"],
                "kind": kind,
            }
        )
    if not slices:
        raise SystemExit(f"no harness kernels found in {trace_path}")
    return slices


def title_for(trace_path: str) -> str:
    stem = os.path.splitext(os.path.basename(trace_path))[0]
    return TITLES.get(stem.rsplit("_", 1)[-1], stem)


def draw(ax, slices: list[dict], title: str, span_ms: float) -> None:
    """One panel: a row per stream, kernels as bars, time in milliseconds."""
    start = min(entry["ts"] for entry in slices)
    # Name each row after the roles that ran on it. Under the lock every stream
    # has exactly one; without it a writer stream also runs the mutation itself,
    # which the combined label makes visible.
    roles: dict[int, set[str]] = {}
    for entry in slices:
        roles.setdefault(entry["stream"], set()).add(KERNELS[entry["kind"]][2])
    labels = {
        stream: " + ".join(sorted(stream_roles, key=ROLE_ORDER.index)) for stream, stream_roles in roles.items()
    }
    rows = {stream: row for row, stream in enumerate(sorted(roles))}

    for entry in slices:
        ax.broken_barh(
            [((entry["ts"] - start) / 1000, entry["dur"] / 1000)],
            (rows[entry["stream"]] - 0.38, 0.76),
            facecolors=KERNELS[entry["kind"]][0],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"stream {stream} · {labels[stream]}" for stream in rows], fontsize=8)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("milliseconds", fontsize=9)
    ax.set_xlim(-0.02 * span_ms, 1.02 * span_ms)
    ax.grid(axis="x", alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", help="Chrome traces exported by the e2e test")
    parser.add_argument("-o", "--output", default="stream_rwlock_timeline.png", help="output PNG")
    args = parser.parse_args()

    panels = [(kernel_slices(path), title_for(path)) for path in args.traces]
    # One shared time axis across panels, so the two runs are comparable.
    span_ms = max(
        (max(e["ts"] + e["dur"] for e in slices) - min(e["ts"] for e in slices)) / 1000 for slices, _ in panels
    )

    figure, axes = plt.subplots(len(panels), 1, figsize=(14, 2.5 * len(panels)), squeeze=False)
    for ax, (slices, title) in zip(axes[:, 0], panels):
        draw(ax, slices, title, span_ms)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colour, edgecolor="black") for colour, _, _ in KERNELS.values()]
    figure.legend(
        handles,
        [label for _, label, _ in KERNELS.values()],
        loc="lower center",
        ncol=len(KERNELS),
        fontsize=9,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
