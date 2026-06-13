import pandas as pd
import os

# ==================================================
# DATASET CONFIGURATION
# ==================================================

DATASETS = {
    "Dataset 1": {
        "file": "fraud_dataset_1.csv"
    },

    "Dataset 2": {
        "file": "fraud_dataset_2.csv"
    },

    "Dataset 3": {
        "file": "fraud_dataset_3.csv"
    }
}

# ==================================================
# POSSIBLE TARGET COLUMNS
# ==================================================

POSSIBLE_TARGETS = [

    "isFraud",
    "IsFraud",

    "Fraudulent",
    "fraudulent",

    "Fraud_Label",
    "fraud_label",

    "Class",
    "class",

    "Fraud",
    "fraud",

    "Label",
    "label",

    "Target",
    "target"
]

# ==================================================
# LOAD DATASET
# ==================================================

def load_dataset(dataset_name):

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not found."
        )

    config = DATASETS[dataset_name]

    # Project Root Folder
    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    # Dataset Path
    file_path = os.path.join(
        base_dir,
        "datasets",
        config["file"]
    )

    # Check File Exists
    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"""
Dataset file not found:

{file_path}

Please verify:
1. datasets folder exists
2. filename is correct
3. file is inside datasets folder
"""
        )

    # Read CSV
    df = pd.read_csv(file_path)

    # Remove Extra Spaces
    df.columns = df.columns.str.strip()

    # Auto Detect Target Column
    target = None

    for column in POSSIBLE_TARGETS:

        if column in df.columns:
            target = column
            break

    # If Target Not Found
    if target is None:

        raise ValueError(
            f"""
No fraud target column detected.

Available Columns:

{list(df.columns)}

Expected One Of:

{POSSIBLE_TARGETS}

Update POSSIBLE_TARGETS list
or rename your fraud column.
"""
        )

    return df, target

# ==================================================
# DATASET INFORMATION
# ==================================================

def get_dataset_info(dataset_name):

    df, target = load_dataset(dataset_name)

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target": target,
        "column_names": list(df.columns)
    }