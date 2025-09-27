
import pandas as pd
from typing import Dict, List

def _is_binary_like(series: pd.Series) -> bool:
    s = series.dropna().astype(str).str.strip().str.lower().replace({"true":"1","false":"0","yes":"1","no":"0"})
    uniq = set(s.unique())
    return uniq.issubset({"0","1"}) and len(uniq) <= 2

def _build_symptom_matrix(df: pd.DataFrame, binary_cols: List[str], text_cols: List[str]) -> pd.DataFrame:
    mats = []
    if binary_cols:
        mats.append(df[binary_cols].apply(lambda x: pd.to_numeric(x, errors="coerce")).fillna(0).clip(0,1).astype(int))
    for c in text_cols:
        tokens = df[c].fillna("").astype(str).str.lower().str.split(",")
        all_tokens = tokens.explode().str.strip()
        top = all_tokens[all_tokens!=""].value_counts().head(200).index.tolist()
        mat = pd.DataFrame(0, index=df.index, columns=top, dtype=int)
        for i, lst in tokens.items():
            for t in lst:
                t = t.strip().lower()
                if t in mat.columns and len(t)>0:
                    mat.at[i, t] = 1
        mats.append(mat)
    if not mats:
        return pd.DataFrame(index=df.index)
    return pd.concat(mats, axis=1).astype(int)

class SymptomRecommender:
    def __init__(self, df: pd.DataFrame):
        lower_cols = {c: str(c).lower() for c in df.columns}
        patient_keywords = ["age","sex","gender","bmi","weight","height","smoker","comorbidity","diagnosis","patient","id","bp","pressure","diabetes","htn","asthma"]
        patient_cols = [c for c in df.columns if any(k in lower_cols[c] for k in patient_keywords)]
        for c in df.columns:
            if c not in patient_cols:
                s = df[c]
                if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 3:
                    patient_cols.append(c)
        binary_symptom_cols = [c for c in df.columns if c not in patient_cols and _is_binary_like(df[c])]
        text_symptom_cols = [c for c in df.columns 
                             if c not in patient_cols and c not in binary_symptom_cols 
                             and df[c].dtype == object and df[c].astype(str).str.contains(",", na=False).mean() > 0.2]
        self.df = df
        self.patient_cols = patient_cols
        self.binary_symptom_cols = binary_symptom_cols
        self.text_symptom_cols = text_symptom_cols
        self.symptom_mat = _build_symptom_matrix(df, binary_symptom_cols, text_symptom_cols)

    def recommend(self, selected_symptom: str, patient_filters: Dict[str, object] = None, top_k: int = 10) -> pd.DataFrame:
        if self.symptom_mat.empty:
            raise ValueError("No symptom columns detected.")
        cols_lower = {c.lower(): c for c in self.symptom_mat.columns}
        sel_key = selected_symptom.lower().strip()
        matched = cols_lower.get(sel_key, None)
        if matched is None:
            for low, orig in cols_lower.items():
                if sel_key in low or low in sel_key:
                    matched = orig
                    break
        if matched is None:
            raise ValueError(f"Selected symptom '{selected_symptom}' not found in known symptoms.")
        mask = self.symptom_mat[matched] == 1
        if patient_filters:
            for k, v in patient_filters.items():
                if k in self.df.columns:
                    col = self.df[k]
                    if pd.api.types.is_numeric_dtype(col) and isinstance(v, (list, tuple)) and len(v)==2:
                        lo, hi = v
                        mask &= col.between(lo, hi)
                    else:
                        mask &= (col.astype(str).str.lower() == str(v).lower())
        subset = self.symptom_mat[mask]
        if subset.empty:
            return pd.DataFrame(columns=["symptom","probability"])
        co_counts = subset.sum().astype(int)
        co_counts = co_counts.drop(index=matched, errors="ignore")
        denom = len(subset)
        cond_prob = (co_counts / max(denom,1)).sort_values(ascending=False)
        out = cond_prob.head(top_k).reset_index()
        out.columns = ["symptom","probability"]
        out.insert(0, "selected_symptom", matched)
        return out
