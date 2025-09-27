# symptom_recommend.py
import pandas as pd, numpy as np, re, json, math
from collections import Counter

DELIMS = r"[;,|/•・·\n]+"
TRAIL_PUNCT = r"^[\s\-\–\—\·\•\:]+|[\s\.\,\;\:\!\?]+$"
SEX_NORMALIZE = {"m":"male","male":"male","ชาย":"male","boy":"male","f":"female","female":"female","หญิง":"female","girl":"female"}

def canon_text(s: str) -> str:
    import re
    t = re.sub(TRAIL_PUNCT, "", str(s).strip(), flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).lower()
    return t

def unique_preserve(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def to_list_from_cell(cell):
    import json, re, pandas as pd
    if pd.isna(cell): return []
    s = str(cell).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            out = [canon_text(x) for x in arr if str(x).strip()]
            return unique_preserve(out)
        except Exception:
            pass
    parts = [canon_text(p) for p in re.split(DELIMS, s) if str(p).strip()]
    parts = [p for p in parts if p and p not in ("nan","none","null")]
    return unique_preserve(parts)

def make_age_band(age, width=10):
    import math, numpy as np
    try:
        a = float(age)
    except Exception:
        return np.nan
    if np.isnan(a): return np.nan
    start = int(math.floor(a/width)*width)
    return f"{start}-{start+width-1}"

def detect_columns(df: pd.DataFrame):
    cols_lower = {c: c.lower() for c in df.columns}
    def find_col(patterns):
        for c, low in cols_lower.items():
            if any(p in low for p in patterns):
                return c
        return None
    pid = find_col(["patient_id","hn","mrn","id","patientid","visit","encounter"])
    age = find_col(["age","อายุ"])
    sex = find_col(["sex","gender","เพศ"])
    sym_text = None
    for patt in ["summary","symptom","symptoms","chief","complaint","sx","อาการ","สรุป"]:
        sym_text = find_col([patt])
        if sym_text: break
    wide = [c for c in df.columns if c.lower().startswith("symptom_")]
    if not wide:
        for c in df.columns:
            low = c.lower()
            if low in [cols_lower.get(pid,""), cols_lower.get(age,""), cols_lower.get(sex,"")]:
                continue
            if df[c].nunique(dropna=True) <= 5 and str(df[c].dtype) in ["int64","float64","bool","int32"]:
                vals = set([v for v in df[c].dropna().unique().tolist()])
                if vals.issubset({0,1,True,False}) and len(vals) >= 1:
                    wide.append(c)
    return pid, age, sex, sym_text, wide

def explode_symptoms(df: pd.DataFrame, pid_col, sym_text_col, wide_cols):
    import numpy as np
    if not pid_col:
        df = df.reset_index().rename(columns={"index":"patient_id"}); pid_col = "patient_id"
    rows = []
    if sym_text_col and df[sym_text_col].notna().any():
        for _, r in df.iterrows():
            sx_list = to_list_from_cell(r.get(sym_text_col, ""))
            for s in sx_list:
                rows.append({"patient_id": r[pid_col], "symptom": s})
    if wide_cols:
        for _, r in df.iterrows():
            for c in wide_cols:
                v = r.get(c, np.nan)
                is_pos = False
                if isinstance(v, str):
                    is_pos = v.strip().lower() in ["1","true","yes","y","t"]
                elif isinstance(v, (int,float,np.integer,np.floating)):
                    is_pos = (v == 1)
                elif isinstance(v, (bool, np.bool_)):
                    is_pos = bool(v)
                if is_pos:
                    sname = canon_text(c.replace("symptom_","").replace("_"," ").strip())
                    rows.append({"patient_id": r[pid_col], "symptom": sname})
    long = pd.DataFrame(rows).dropna()
    if not long.empty:
        long["symptom"] = long["symptom"].astype(str).str.strip().str.lower()
        long = long[long["symptom"] != ""].drop_duplicates()
    return long, pid_col

def prepare_profiles(long_df: pd.DataFrame, raw_df: pd.DataFrame, pid_col, age_col, sex_col):
    import numpy as np
    if long_df.empty:
        return pd.DataFrame(columns=["patient_id","symptom_list","sex_norm","age_band"])
    sx_by_pt = long_df.groupby("patient_id")["symptom"].apply(lambda s: unique_preserve(list(s))).reset_index(name="symptom_list")
    if sex_col or age_col:
        demo_cols = [pid_col] + ([sex_col] if sex_col else []) + ([age_col] if age_col else [])
        demo = raw_df[demo_cols].drop_duplicates()
    else:
        demo = raw_df[[pid_col]].drop_duplicates()
    if sex_col:
        demo["sex_norm"] = demo[sex_col].astype(str).str.lower().map(SEX_NORMALIZE).fillna(demo[sex_col].astype(str).str.lower())
    else:
        demo["sex_norm"] = np.nan
    if age_col:
        ages = pd.to_numeric(demo[age_col], errors="coerce")
        demo["age_band"] = [make_age_band(a) for a in ages]
    else:
        demo["age_band"] = np.nan
    demo = demo.rename(columns={pid_col:"patient_id"})
    profiles = sx_by_pt.merge(demo[["patient_id","sex_norm","age_band"]], on="patient_id", how="left")
    return profiles

def score_cooccurrence(profiles: pd.DataFrame, selected_symptoms, age=None, sex=None, top_n=5, require_all=True, add_pmi=True):
    from collections import Counter
    import numpy as np
    if profiles.empty:
        return pd.DataFrame(columns=["symptom","support","confidence","lift","pmi"] if add_pmi else ["symptom","support","confidence","lift"])
    sel = unique_preserve([canon_text(s) for s in (selected_symptoms or []) if str(s).strip()])
    pf = profiles.copy()
    if age is not None:
        band = make_age_band(age)
        if band is not None: pf = pf[pf["age_band"] == band]
    if sex is not None:
        sex_norm = str(sex).lower()
        pf = pf[pf["sex_norm"] == sex_norm]
    if pf.empty:
        return pd.DataFrame(columns=["symptom","support","confidence","lift","pmi"] if add_pmi else ["symptom","support","confidence","lift"])
    baskets = pf["symptom_list"].tolist()
    def contains_all(b): return all(s in b for s in sel)
    def contains_any(b): return any(s in b for s in sel) if sel else True
    base = baskets if not sel else [b for b in baskets if (contains_all(b) if require_all else contains_any(b))]
    if not base:
        return pd.DataFrame(columns=["symptom","support","confidence","lift","pmi"] if add_pmi else ["symptom","support","confidence","lift"])
    total = len(baskets); base_n = len(base)
    overall = Counter([sx for b in baskets for sx in b])
    co = Counter([sx for b in base for sx in b if sx not in sel])
    rows = []
    for sx, cnt in co.items():
        support = cnt/total
        confidence = cnt/base_n
        p_sx = overall[sx]/total if total else 0.0
        lift = confidence/p_sx if p_sx>0 else np.nan
        if add_pmi:
            p_xy = cnt/total; p_x = base_n/total; p_y = p_sx
            pmi = float(np.log((p_xy/(p_x*p_y)))) if (p_x>0 and p_y>0 and p_xy>0) else float("nan")
            rows.append({"symptom": sx, "support": support, "confidence": confidence, "lift": lift, "pmi": pmi})
        else:
            rows.append({"symptom": sx, "support": support, "confidence": confidence, "lift": lift})
    recs = pd.DataFrame(rows)
    if recs.empty: return recs
    sort_cols = ["pmi","lift","confidence","support"] if add_pmi else ["lift","confidence","support"]
    return recs.sort_values(sort_cols, ascending=False).head(top_n).reset_index(drop=True)
