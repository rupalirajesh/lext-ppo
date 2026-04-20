import os
import pandas as pd

REF_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "references.csv")

def save_to_references(row_data):
    df = pd.DataFrame([row_data])

    write_header = not os.path.exists(REF_FILE)

    df.to_csv(
        REF_FILE,
        mode='a',          # ✅ append instead of overwrite
        header=write_header,
        index=False
    )