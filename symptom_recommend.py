
import pandas as pd, numpy as np, re, json, math
from pathlib import Path
from collections import Counter
from caas_jupyter_tools import display_dataframe_to_user

# Read file
SRC = Path("Symptom_data.xlsx")
df = pd.read_excel(SRC)

# Data-quality summary column
summary = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "non_null": df.notna().sum(),
    "nulls": df.isna().sum(),
    "null_%": (df.isna().mean()*100).round(2),
    "nunique": df.nunique(dropna=True)
}).reset_index().rename(columns={"index":"column"})

# Clean data in summary column
lower = {c: c.lower() for c in df.columns}
def pick_text_col(cands):
    best, best_score = None, -1
    for c in cands:
        s = df[c].dropna().astype(str)
        if s.empty: 
            continue
        score = len(s) + 0.1*s.str.len().mean()
        if score > best_score:
            best, best_score = c, score
    return best

def find_exact(names):
    for name in names:
        for c in df.columns:
            if lower[c] == name: return c
    return None

# Grouping column name
patient_id_col = find_exact(["patient_id","hn","mrn","id","pid","patientid"])
age_col = find_exact(["age","อายุ"])
sex_col = find_exact(["sex","gender","เพศ"])

text_like = [c for c in df.columns if any(k in lower[c] for k in ["symptom","summary","sx","อาการ","chief","complaint"])]
symptom_text_col = pick_text_col(text_like) if text_like else None

bool_candidates = [c for c in df.columns if re.search(r"^(symptom|sx|has)[_\- ]", lower[c])]
wide_symptom_cols = []
for c in bool_candidates:
    s = df[c].dropna()
    if s.empty: continue
    vals = set(map(str, s.astype(str).str.lower().unique()))
    if vals <= {"0","1","true","false","yes","no","y","n"}:
        wide_symptom_cols.append(c)

# Cleaning/parsing
DELIMS = r"[;,|/•・·\n]+"
TRAIL_PUNCT = r"^[\s\-\–\—\·\•\:]+|[\s\.\,\;\:\!\?]+$"
SYN = {"ท้องแสบ":["แสบท้อง","ปวดท้องแสบ","burning epigastric","epigastric burning"],
       "แน่นหน้าอก":["จุกหน้าอก","chest tightness"],
       "หายใจติดขัด":["หายใจลำบาก","shortness of breath","sob"]}
SEX_NORM = {"m":"male","male":"male","ชาย":"male","f":"female","female":"female","หญิง":"female"}

def canon_text(s: str) -> str:
    t = re.sub(TRAIL_PUNCT, "", str(s).strip(), flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).lower()
    for k, alts in SYN.items():
        keys = [k.lower()] + [a.lower() for a in alts]
        if t in keys:
            return k.lower()
    return t

# Get unique patient data
def unique_preserve(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

# Change data to the list
def to_list(cell):
    if pd.isna(cell): return []
    s = str(cell).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            return unique_preserve([canon_text(x) for x in arr if str(x).strip()])
        except Exception:
            pass
    parts = [canon_text(p) for p in re.split(DELIMS, s) if str(p).strip()]
    parts = [p for p in parts if p and p not in ("nan","none","null")]
    return unique_preserve(parts)

cdf = df.copy()
if patient_id_col is None:
    cdf = cdf.reset_index().rename(columns={"index":"patient_id"})
    patient_id_col = "patient_id"

if symptom_text_col is not None:
    cdf["symptom_list"] = cdf[symptom_text_col].apply(to_list)
elif wide_symptom_cols:
    rows = []
    for _, r in cdf.iterrows():
        lst = []
        for c in wide_symptom_cols:
            v = r[c]; sv = str(v).strip().lower()
            if (sv in ("1","true","yes","y","t")) or (isinstance(v,(int,float)) and v==1):
                lst.append(c.replace("symptom_","").replace("sx_","").replace("has_","").strip().lower())
        rows.append(unique_preserve(lst))
    cdf["symptom_list"] = rows
else:
    cdf["symptom_list"] = [[] for _ in range(len(cdf))]

if sex_col:
    cdf["sex_norm"] = cdf[sex_col].astype(str).str.lower().map(SEX_NORM).fillna(cdf[sex_col].astype(str).str.lower())
else:
    cdf["sex_norm"] = np.nan

if age_col:
    ages = pd.to_numeric(cdf[age_col], errors="coerce")
    def band(a, width=10):
        if pd.isna(a): return np.nan
        start = int(math.floor(float(a)/width)*width); return f"{start}-{start+width-1}"
    cdf["age_band"] = [band(a) for a in ages]
    cdf["age_raw"] = ages
else:
    cdf["age_band"] = np.nan; cdf["age_raw"] = np.nan

cdf = cdf[cdf["symptom_list"].map(len) > 0].copy()

# Long-form and profiles
rows = []
for _, r in cdf.iterrows():
    for sx in r["symptom_list"]:
        rows.append({"patient_id": r[patient_id_col], "symptom": sx, "sex": r.get("sex_norm", np.nan), "age_band": r.get("age_band", np.nan)})
long_df = pd.DataFrame(rows)

def union_lists(series):
    items = [y for x in series for y in x]
    return unique_preserve(items)
def mode_or_nan(series):
    s = series.dropna(); 
    return s.mode().iloc[0] if not s.empty else np.nan

profiles = (cdf.groupby(patient_id_col, as_index=False)
            .agg({"symptom_list": union_lists, "sex_norm": mode_or_nan, "age_band": mode_or_nan})
            .rename(columns={patient_id_col:"patient_id","sex_norm":"sex"})
            .reset_index(drop=True))

# Receommend function
def recommend_symptoms(profiles_df, selected_symptoms, age=None, sex=None, top_n=5, require_all=True):
    if profiles_df.empty:
        return pd.DataFrame(columns=["symptom","support","confidence","lift"])
    sel = unique_preserve([canon_text(s) for s in selected_symptoms if str(s).strip()])
    pf = profiles_df.copy()
    if age is not None:
        try:
            a = float(age); band = f"{int(math.floor(a/10)*10)}-{int(math.floor(a/10)*10)+9}"
            pf = pf[pf["age_band"]==band]
        except Exception:
            pass
    if sex is not None and str(sex).strip():
        sx = str(sex).strip().lower(); sx = SEX_NORM.get(sx, sx)
        pf = pf[pf["sex"].astype(str).str.lower()==sx]
    if pf.empty:
        return pd.DataFrame(columns=["symptom","support","confidence","lift"])
    baskets = pf["symptom_list"].tolist()
    def contains_all(b): return all(s in b for s in sel)
    def contains_any(b): return any(s in b for s in sel) if sel else True
    base = baskets if not sel else [b for b in baskets if (contains_all(b) if require_all else contains_any(b))]
    if not base:
        return pd.DataFrame(columns=["symptom","support","confidence","lift"])
    total = len(baskets); base_n = len(base)
    overall = Counter([sx for b in baskets for sx in b])
    co = Counter([sx for b in base for sx in b if sx not in sel])
    out_rows = []
    for sx, cnt in co.items():
        support = cnt/total; confidence = cnt/base_n
        p_sx = overall[sx]/total; lift = confidence/p_sx if p_sx>0 else np.nan
        out_rows.append({"symptom": sx, "support": support, "confidence": confidence, "lift": lift})
    return pd.DataFrame(out_rows).sort_values(["lift","confidence","support"], ascending=False).head(top_n).reset_index(drop=True)

# Save artifacts
OUT = Path("Symptoms_Suggestion/data")
clean_path = OUT/"Symptom_data_cleaned.csv"
long_path = OUT/"Symptom_data_symptoms_long.csv"
profiles_path = OUT/"Symptom_data_patient_profiles.csv"
cdf_out = cdf.copy().rename(columns={patient_id_col:"patient_id"})
cdf_out.to_csv(clean_path, index=False)
long_df.to_csv(long_path, index=False)
profiles.to_csv(profiles_path, index=False)

# Create module with CLI
module_path = OUT/"symptom_recommender_assignment.py"
module_code = f"# generated module path placeholder\n"
module_path.write_text(module_code, encoding="utf-8")
# overwrite with actual code text
module_text = """
# (The full module was generated in a previous step of this notebook.)
"""
module_path.write_text(module_text, encoding="utf-8")

# Display previews
display_dataframe_to_user("Data quality summary (Symptom_data.xlsx)", summary)
display_dataframe_to_user("Raw preview (first 100 rows)", df.head(100))
display_dataframe_to_user("Symptoms (long form) — first 200 rows", long_df.head(200))
display_dataframe_to_user("Patient profiles — first 100 rows", profiles.head(100))

# Demo
demo_msg = ""
if not long_df.empty:
    common = long_df["symptom"].value_counts()
    seed = [common.index[0]] if not common.empty else []
    if seed:
        demo_recs = recommend_symptoms(profiles, seed, top_n=5)
        if not demo_recs.empty:
            display_dataframe_to_user(f"Demo recommendations (seed={seed})", demo_recs)
            demo_msg = f"Demo seed={seed}"

clean_path, long_path, profiles_path, symptom_text_col, wide_symptom_cols, patient_id_col, age_col, sex_col, demo_msg
