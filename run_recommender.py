# run_recommender.py
import argparse, re, pathlib
import pandas as pd
from symptom_recommend import detect_columns, explode_symptoms, prepare_profiles, score_cooccurrence

def main():
    ap = argparse.ArgumentParser(description="Symptom recommender (co-occurrence)")
    ap.add_argument("--file", default="Symptom_data.xlsx", help="Path to Excel/CSV file")
    ap.add_argument("--sheet", default=None, help="Excel sheet name (optional)")
    ap.add_argument("--symptoms", default="", help="Seed symptoms, e.g., 'fever, cough'")
    ap.add_argument("--age", type=float, default=None, help="Age in years (optional)")
    ap.add_argument("--sex", default=None, help="male/female/ชาย/หญิง (optional)")
    ap.add_argument("--top_n", type=int, default=5, help="Number of suggestions")
    ap.add_argument("--any", action="store_true", help="Use any-of instead of all-of")
    ap.add_argument("--export", default=None, help="CSV path to save results")
    args = ap.parse_args()

    p = pathlib.Path(args.file)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_excel(p, sheet_name=args.sheet) if p.suffix.lower() in (".xlsx",".xls") else pd.read_csv(p)

    pid_col, age_col, sex_col, sym_text_col, wide_cols = detect_columns(df)
    long_df, pid_col = explode_symptoms(df, pid_col, sym_text_col, wide_cols)
    profiles = prepare_profiles(long_df, df, pid_col, age_col, sex_col)

    seeds = [s.strip() for s in re.split(r"[;,]", args.symptoms) if s.strip()]

    recs = score_cooccurrence(
        profiles=profiles,
        selected_symptoms=seeds,
        age=args.age,
        sex=args.sex,
        top_n=args.top_n,
        require_all=(not args.any),
        add_pmi=True
    )

    print("Seeds:", seeds or ["(none)"], "| Filters:", f"age={args.age}", f"sex={args.sex}")
    if recs.empty:
        print("No recommendations with current filters.")
    else:
        print(recs.to_string(index=False, formatters={
            "support":"{:.3f}".format, "confidence":"{:.3f}".format, "lift":"{:.3f}".format, "pmi":"{:.3f}".format
        }))
    if args.export:
        recs.to_csv(args.export, index=False)
        print("Saved ->", args.export)

if __name__ == "__main__":
    main()
