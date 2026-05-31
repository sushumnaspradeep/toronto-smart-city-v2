# building_permit.py
"""
Toronto building-permit sub-agent.

Modes:
  python building_permit.py build <permits.csv> <insights.json>
  echo '{"fsa":"M5H","user_question":"..."}' | python building_permit.py run

Programmatic:
  from building_permit import run
  run({"fsa":"M5H","user_question":"expected approval time?"})
"""

import os, sys, json
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
INSIGHTS_PATH = os.environ.get("INSIGHTS_PATH", "insights.json")
MODEL         = os.environ.get("PERMIT_AGENT_MODEL", "gpt-4o-mini")

DATE_COLS = ["APPLICATION_DATE", "ISSUED_DATE", "COMPLETED_DATE"]
NUM_COLS  = ["DWELLING_UNITS_CREATED", "DWELLING_UNITS_LOST", "EST_CONST_COST",
             "ASSEMBLY", "INSTITUTIONAL", "RESIDENTIAL",
             "BUSINESS_AND_PERSONAL_SERVICES", "MERCANTILE", "INDUSTRIAL",
             "INTERIOR_ALTERATIONS", "DEMOLITION"]
MIN_PERMITS_BUILDER = 10


# ============================================================
# OFFLINE: BUILD insights.json
# ============================================================
def _pct(s, p):
    s = s.dropna()
    return float(np.nanpercentile(s, p)) if len(s) else None

def _stats(s):
    s = s.dropna()
    if s.empty: return None
    return {"n": int(s.size), "mean": float(s.mean()), "median": float(s.median()),
            "p25": _pct(s,25), "p75": _pct(s,75),
            "p90": _pct(s,90), "p99": _pct(s,99),
            "min": float(s.min()), "max": float(s.max())}

def _group_stats(frame, by, value="approval_days", min_n=20, top=25):
    out = []
    for key, vals in frame.groupby(by)[value]:
        v = vals.dropna()
        if len(v) < min_n: continue
        out.append({"key": key if not isinstance(key, tuple) else list(key),
                    "n": int(len(v)), "median_days": float(v.median()),
                    "p90_days": _pct(v,90), "mean_days": float(v.mean())})
    out.sort(key=lambda x: -x["n"])
    return out[:top]

def _builder_table(frame, min_n=MIN_PERMITS_BUILDER, top=50, ascending=True):
    rows = []
    for name, vals in frame.groupby("BUILDER_NAME")["approval_days"]:
        v = vals.dropna()
        if len(v) < min_n: continue
        rows.append({"builder": name, "n_permits": int(len(v)),
                     "median_days": float(v.median()),
                     "p90_days": _pct(v,90), "mean_days": float(v.mean())})
    rows.sort(key=lambda r: r["median_days"], reverse=not ascending)
    return rows[:top]

def _builder_delta_table(frame, min_n=MIN_PERMITS_BUILDER, top=25, ascending=True):
    rows = []
    for name, sub in frame.groupby("BUILDER_NAME"):
        v = sub["speed_delta"].dropna()
        if len(v) < min_n: continue
        rows.append({"builder": name, "n_permits": int(len(v)),
                     "median_delta_days": float(v.median()),
                     "median_days_absolute": float(sub["approval_days"].median())})
    rows.sort(key=lambda r: r["median_delta_days"], reverse=not ascending)
    return rows[:top]


def build_insights(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, low_memory=False)
    for c in DATE_COLS: df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in NUM_COLS:  df[c] = pd.to_numeric(df[c], errors="coerce")

    df["FSA"]             = df["POSTAL"].astype(str).str.strip().str.upper().str[:3]
    df["approval_days"]   = (df["ISSUED_DATE"]    - df["APPLICATION_DATE"]).dt.days
    df["completion_days"] = (df["COMPLETED_DATE"] - df["ISSUED_DATE"]).dt.days
    df["app_year"]        = df["APPLICATION_DATE"].dt.year

    issued = df[df["ISSUED_DATE"].notna() & (df["approval_days"] >= 0)].copy()

    b = issued.copy()
    b["BUILDER_NAME"] = b["BUILDER_NAME"].fillna("").str.strip().str.upper()
    b = b[b["BUILDER_NAME"] != ""]
    fsa_median = issued.groupby("FSA")["approval_days"].median()
    b["fsa_baseline"] = b["FSA"].map(fsa_median)
    b["speed_delta"]  = b["approval_days"] - b["fsa_baseline"]

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_file": csv_path,
        "dataset": {
            "rows": int(len(df)),
            "date_range": {
                "application_min": str(df["APPLICATION_DATE"].min()),
                "application_max": str(df["APPLICATION_DATE"].max()),
            },
            "null_rate": {c: float(df[c].isna().mean()) for c in df.columns},
        },
        "status_breakdown":      df["STATUS"].value_counts(dropna=False).to_dict(),
        "permit_type_counts":    df["PERMIT_TYPE"].value_counts().head(30).to_dict(),
        "structure_type_counts": df["STRUCTURE_TYPE"].value_counts().head(30).to_dict(),
        "work_counts":           df["WORK"].value_counts().head(30).to_dict(),

        "approval_time_overall":   _stats(issued["approval_days"]),
        "completion_time_overall": _stats(df["completion_days"]),

        "approval_time_by_permit_type":    _group_stats(issued, "PERMIT_TYPE"),
        "approval_time_by_structure_type": _group_stats(issued, "STRUCTURE_TYPE"),
        "approval_time_by_work":           _group_stats(issued, "WORK"),
        "approval_time_by_fsa":            _group_stats(issued, "FSA", top=200),
        "approval_time_by_ward":           _group_stats(issued, "WARD_GRID", top=50),
        "approval_time_by_year":           _group_stats(issued, "app_year",
                                                        min_n=50, top=50),
        "approval_time_by_type_and_fsa":   _group_stats(issued,
                                                        ["PERMIT_TYPE","FSA"],
                                                        min_n=20, top=500),
        "volume_by_year":  df.groupby("app_year").size().dropna().astype(int).to_dict(),

        "cost": {
            "overall": _stats(df["EST_CONST_COST"]),
            "by_structure_type": {
                k: _stats(g["EST_CONST_COST"])
                for k, g in df.groupby("STRUCTURE_TYPE") if len(g) >= 50
            },
        },
        "dwelling_units": {
            "total_created": int(df["DWELLING_UNITS_CREATED"].sum(skipna=True)),
            "total_lost":    int(df["DWELLING_UNITS_LOST"].sum(skipna=True)),
            "net_added":     int((df["DWELLING_UNITS_CREATED"]
                                  - df["DWELLING_UNITS_LOST"]).sum(skipna=True)),
        },
        "top_builders_by_volume": (
            df["BUILDER_NAME"].dropna().str.strip().str.upper()
              .value_counts().head(25).to_dict()
        ),
        "builder_performance": {
            "min_permits_threshold": MIN_PERMITS_BUILDER,
            "city_median_days": float(issued["approval_days"].median()),
            "fastest_builders":         _builder_table(b, ascending=True,  top=25),
            "slowest_builders":         _builder_table(b, ascending=False, top=25),
            "most_active_builders":     sorted(
                _builder_table(b, top=10_000),
                key=lambda r: -r["n_permits"])[:25],
            "fastest_builders_vs_area": _builder_delta_table(b, ascending=True,  top=25),
            "slowest_builders_vs_area": _builder_delta_table(b, ascending=False, top=25),
            "fastest_by_permit_type": {
                p: _builder_table(b[b["PERMIT_TYPE"]==p], min_n=10, top=10)
                for p in b["PERMIT_TYPE"].value_counts().head(8).index
            },
            "fastest_by_structure_type": {
                s: _builder_table(b[b["STRUCTURE_TYPE"]==s], min_n=10, top=10)
                for s in b["STRUCTURE_TYPE"].value_counts().head(8).index
            },
            "fastest_by_fsa": {
                f: _builder_table(b[b["FSA"]==f], min_n=5, top=10)
                for f in b["FSA"].value_counts().head(25).index
            },
        },
        "pending_backlog": {
            "still_open":  int(df["ISSUED_DATE"].isna().sum()),
            "by_status":   df[df["ISSUED_DATE"].isna()]["STATUS"]
                             .value_counts().to_dict(),
        },
        "data_quality_flags": {
            "negative_approval_days":   int((df["approval_days"] < 0).sum()),
            "missing_postal":           int(df["POSTAL"].isna().sum()),
            "missing_application_date": int(df["APPLICATION_DATE"].isna().sum()),
        },
    }


# ============================================================
# RUNTIME: TOOLS over insights.json
# ============================================================
@lru_cache(maxsize=1)
def _insights():
    with open(INSIGHTS_PATH) as f:
        return json.load(f)


def lookup_approval_time(permit_type=None, fsa=None, ward=None,
                         structure_type=None, min_n=20):
    I = _insights()

    if permit_type and fsa:
        for r in I["approval_time_by_type_and_fsa"]:
            if r["key"] == [permit_type, fsa] and r["n"] >= min_n:
                return {**r, "fallback_level": "permit_type+fsa"}
    if fsa:
        for r in I["approval_time_by_fsa"]:
            if r["key"] == fsa and r["n"] >= min_n:
                return {**r, "fallback_level": "fsa"}
    if ward:
        for r in I["approval_time_by_ward"]:
            if r["key"] == ward and r["n"] >= min_n:
                return {**r, "fallback_level": "ward"}
    if structure_type:
        for r in I["approval_time_by_structure_type"]:
            if r["key"] == structure_type and r["n"] >= min_n:
                return {**r, "fallback_level": "structure_type"}
    if permit_type:
        for r in I["approval_time_by_permit_type"]:
            if r["key"] == permit_type and r["n"] >= min_n:
                return {**r, "fallback_level": "permit_type"}

    o = I["approval_time_overall"]
    return {"n": o["n"], "median_days": o["median"], "p90_days": o["p90"],
            "mean_days": o["mean"], "fallback_level": "overall"}


def lookup_builders(fsa=None, permit_type=None, structure_type=None,
                    mode="vs_area", top=5):
    bp = _insights()["builder_performance"]
    if fsa and fsa in bp["fastest_by_fsa"]:
        return bp["fastest_by_fsa"][fsa][:top]
    if structure_type and structure_type in bp["fastest_by_structure_type"]:
        return bp["fastest_by_structure_type"][structure_type][:top]
    if permit_type and permit_type in bp["fastest_by_permit_type"]:
        return bp["fastest_by_permit_type"][permit_type][:top]
    key = "fastest_builders_vs_area" if mode == "vs_area" else "fastest_builders"
    return bp[key][:top]


def dataset_summary():
    I = _insights()
    return {
        "rows": I["dataset"]["rows"],
        "date_range": I["dataset"]["date_range"],
        "city_median_approval_days": I["builder_performance"]["city_median_days"],
        "open_permits": I["pending_backlog"]["still_open"],
    }


# ============================================================
# RUNTIME: LLM AGENT
# ============================================================
SYSTEM = """You answer Toronto building-permit questions using ONLY the tools provided.

The user's location has already been resolved by an upstream geocoding agent
and is given to you as structured fields (fsa, ward, lat, lon). Do NOT ask
for or attempt to resolve addresses yourself.

Rules:
1. Never invent numbers — every figure must come from a tool result.
2. Always include sample size (n) and fallback_level when citing approval times.
3. Prefer 'vs_area' mode when ranking builders; always cite n_permits.
4. If n < 20 for any stat, warn the user it is low-confidence.
5. Return a concise answer focused on what the user asked.
"""

TOOLS_SCHEMA = [
    {"type":"function","function":{
        "name":"lookup_approval_time",
        "description":"Historical permit approval-time stats for a bucket.",
        "parameters":{"type":"object","properties":{
            "permit_type":{"type":"string"},
            "structure_type":{"type":"string"},
            "fsa":{"type":"string"},
            "ward":{"type":"string"}}}}},
    {"type":"function","function":{
        "name":"lookup_builders",
        "description":"Ranked builders by approval speed for a bucket.",
        "parameters":{"type":"object","properties":{
            "fsa":{"type":"string"},
            "permit_type":{"type":"string"},
            "structure_type":{"type":"string"},
            "mode":{"type":"string","enum":["absolute","vs_area"]},
            "top":{"type":"integer"}}}}},
    {"type":"function","function":{
        "name":"dataset_summary",
        "description":"Dataset coverage and city baselines.",
        "parameters":{"type":"object","properties":{}}}},
]

DISPATCH = {
    "lookup_approval_time": lambda **kw: lookup_approval_time(**kw),
    "lookup_builders":      lambda **kw: lookup_builders(**kw),
    "dataset_summary":      lambda **kw: dataset_summary(),
}


def run(payload: dict, max_iters: int = 5) -> dict:
    """
    payload keys (all optional except user_question + at least fsa or ward):
      address, fsa, ward, lat, lon,
      permit_type, structure_type, intent, user_question
    """
    from openai import OpenAI
    client = OpenAI()

    context = {k: payload.get(k) for k in
               ["address","fsa","ward","lat","lon",
                "permit_type","structure_type","intent"]
               if payload.get(k) is not None}

    user_msg = (
        f"Resolved location and hints from upstream:\n"
        f"{json.dumps(context, indent=2)}\n\n"
        f"User question: {payload.get('user_question','(none)')}"
    )

    messages = [{"role":"system","content":SYSTEM},
                {"role":"user","content":user_msg}]
    trace = []

    for _ in range(max_iters):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages,
            tools=TOOLS_SCHEMA, tool_choice="auto")
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return {"answer": msg.content, "trace": trace,
                    "input_context": payload}

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            for k in ("fsa","ward","permit_type","structure_type"):
                args.setdefault(k, payload.get(k))
            args = {k:v for k,v in args.items() if v is not None}
            result = DISPATCH[call.function.name](**args)
            trace.append({"tool": call.function.name,
                          "args": args, "result": result})
            messages.append({"role":"tool", "tool_call_id": call.id,
                             "content": json.dumps(result, default=str)})

    return {"answer":"iteration limit reached", "trace": trace}


# ============================================================
# CLI
# ============================================================
def _main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "build":
        csv_path, out_path = sys.argv[2], sys.argv[3]
        data = build_insights(csv_path)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"wrote {out_path}", file=sys.stderr)

    elif cmd == "run":
        payload = json.load(sys.stdin)
        print(json.dumps(run(payload), indent=2, default=str))

    else:
        print(f"unknown command: {cmd}"); sys.exit(1)


if __name__ == "__main__":
    _main()
