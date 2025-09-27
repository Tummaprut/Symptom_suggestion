# Symptom_suggestion
Analysis symptom and suggest patients like a doctor

# Assignment: Symptom Recommender

## Deliverables
- **Core module**: `symptom_recommender.py`
- **CLI runner**: `run_recommender.py`
- **Derived data**: `profiles.csv`, `symptoms_long.csv`
- **Input**: `Symptom_data.xlsx`

## Quick Start
```bash
pip install pandas numpy openpyxl
python run_recommender.py --file Symptom_data.xlsx --symptoms "fever, cough" --top_n 5
python run_recommender.py --file Symptom_data.xlsx --symptoms "fever, cough" --sex male --age 30
python run_recommender.py --file Symptom_data.xlsx --symptoms "fever, cough" --any
python run_recommender.py --file Symptom_data.xlsx --symptoms "fever" --export recs.csv
```

## How it works
1. Column detection (patient_id/age/sex/symptom text/wide booleans)
2. Parsing text symptoms or wide booleans into long form
3. Profiles = unique symptom basket per patient + sex_norm, age_band
4. Recommender ranks by support, confidence, lift, PMI
5. Optional demographic filters

Seed used for demo: ['{"diseases": []']
