import sys
import os
import json
from pathlib import Path
from datetime import datetime

ROOT = Path("/workspace/birdVision")
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

from scripts.export_all import MANIFEST, _benchmark_engine

rows = []
for spec in MANIFEST:
    for mode in ("single", "batch"):
        ep = Path("engines") / spec.tenant / spec.pipeline / f"{spec.name}_{mode}.engine"
        row = {
            "spec_key": spec.key,
            "mode": mode,
            "engine_exists": ep.exists(),
            "engine_path": str(ep) if ep.exists() else None,
            "engine_size_mb": round(ep.stat().st_size / (1024 * 1024), 1) if ep.exists() else None,
            "parity_passed": None,
            "parity_metrics": {},
            "bench": {},
        }
        pj = Path("reports/parity_real") / spec.tenant / spec.pipeline / f"{spec.name}_{mode}.json"
        if pj.exists():
            try:
                d = json.loads(pj.read_text())
                row["parity_passed"] = d.get("passed")
                row["parity_metrics"] = d.get("metrics", {})
            except Exception:
                pass
        if ep.exists():
            try:
                bs = 1 if mode == "single" else 8
                b = _benchmark_engine(ep, spec.input_shape, batch_size=bs, n_warmup=5, n_iters=30)
                row["bench"] = b
            except Exception as e:
                row["bench"] = {"error": str(e)}
        rows.append(row)
        fps = row["bench"].get("fps", "--") if isinstance(row["bench"], dict) else "--"
        print(f"{spec.key}/{mode}: engine={row['engine_exists']}, parity={row['parity_passed']}, fps={fps}")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
Path(f"reports/export_final_{ts}.json").write_text(
    json.dumps(rows, indent=2, ensure_ascii=False)
)

total = len(rows)
built = sum(1 for r in rows if r["engine_exists"])
passed = sum(1 for r in rows if r["parity_passed"] is True)

lines = [
    "# TensorRT Export — Final Report",
    "",
    f"**Sana:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"**Engines built:** {built}/{total}",
    f"**Parity passed:** {passed}/{total}",
    "",
    "## Summary Table",
    "",
    "| Model | Mode | Built | Size (MB) | Parity | FPS | p50 (ms) | p95 (ms) |",
    "|---|---|---|---|---|---|---|---|",
]
for r in rows:
    built_m = "YES" if r["engine_exists"] else "NO"
    size = r["engine_size_mb"] if r["engine_size_mb"] is not None else "--"
    pm = "--" if r["parity_passed"] is None else ("PASS" if r["parity_passed"] else "FAIL")
    b = r["bench"] if isinstance(r["bench"], dict) else {}
    fps = b.get("fps", "--")
    p50 = b.get("latency_ms_p50", "--")
    p95 = b.get("latency_ms_p95", "--")
    lines.append(
        f"| `{r['spec_key']}` | {r['mode']} | {built_m} | {size} | {pm} | {fps} | {p50} | {p95} |"
    )

not_built = [r for r in rows if not r["engine_exists"]]
if not_built:
    lines += ["", "## Not Built", ""]
    for r in not_built:
        lines.append(f"- `{r['spec_key']}` / {r['mode']}")

parity_fails = [r for r in rows if r["engine_exists"] and r["parity_passed"] is False]
if parity_fails:
    lines += ["", "## Parity Failures", ""]
    for r in parity_fails:
        m = json.dumps(r["parity_metrics"], ensure_ascii=False)
        lines.append(f"- `{r['spec_key']}` / {r['mode']} — metrics: {m}")

md_path = Path(f"reports/export_final_{ts}.md")
md_path.write_text("\n".join(lines))
print(f"\nMD:   {md_path}")
print(f"JSON: reports/export_final_{ts}.json")