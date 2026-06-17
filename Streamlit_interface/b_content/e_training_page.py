import streamlit as st
import os
import pickle
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mypackage.paths import DATA_DIR, IMAGES_DIR
from mypackage.preprocessing import cat_identify
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression

import catboost
from catboost import CatBoostClassifier
import random
random.seed(42)
random_seeds = [random.randint(0, 1000) for _ in range(50)] # Random seeds for reproducibility
# print("Random seeds:", random_seeds)

SESSION_PREFIX = "training_page"
ACTION_LABELS = {
    "drop": "drop",
    "fill_value": "replace with a value",
    "fill_max": "fill with max",
    "fill_min": "fill with min",
    "fill_median": "fill with median",
    "fill_mean": "fill with average",
    "fill_mode": "fill with mode",
}
LABEL_SCAN_MAX_UNIQUE = 10
LABEL_SCAN_MAX_UNIQUE_RATIO = 0.2
LABEL_SCAN_MAX_CANDIDATES = 20
LABEL_SCAN_MAX_ROWS = 2000
LABEL_SCAN_MIN_ROWS = 30
LABEL_SCAN_MIN_CLASS_COUNT = 2

def check_missing_data(df, show_plot=False, show_info=False):
    missing_ratio = df.isna().mean()
    denominator = max(df.shape[1], 1)

    if show_info:
        # st.markdown(f"""
        # <div style="line-height:1.4;">
        # Total number of features: {df.shape[1]-1}.<br>
        # Completed features (without missing data): {len(missing_ratio[missing_ratio == 0])} ({np.round(len(missing_ratio[missing_ratio == 0])/(df.shape[1]-1)*100,2)} %).<br>
        # Number of features with missing data: {len(missing_ratio[missing_ratio > 0])} ({np.round(len(missing_ratio[missing_ratio > 0])/(df.shape[1]-1)*100,2)}%).<br>
        #     _ Number of features with >20% missing data: {len(missing_ratio[missing_ratio > 0.2])} ({np.round(len(missing_ratio[missing_ratio > 0.2])/(df.shape[1]-1)*100,2)}%).<br>
        #     _ Number of features with >50% missing data: {len(missing_ratio[missing_ratio > 0.5])} ({np.round(len(missing_ratio[missing_ratio > 0.5])/(df.shape[1]-1)*100,2)}%).<br>
        #     _ Number of features with >80% missing data: {len(missing_ratio[missing_ratio > 0.8])} ({np.round(len(missing_ratio[missing_ratio > 0.8])/(df.shape[1]-1)*100,2)}%).
        # </div>
        # """, unsafe_allow_html=True)  
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                        <div style="line-height:1.4;">
                        Completed columns (without missing data):<br>
                        <span style="background-color:#e6f2ff; padding:4px 8px; border-radius:6px;">{len(missing_ratio[missing_ratio == 0])} ({np.round(len(missing_ratio[missing_ratio == 0])/denominator*100,2)} %)</span>.<br>
                        """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                        <div style="line-height:1.4;">
                        Number of columns with missing data:<br>
                        <span style="background-color:#e6f2ff; padding:4px 8px; border-radius:6px;">{len(missing_ratio[missing_ratio > 0])} ({np.round(len(missing_ratio[missing_ratio > 0])/denominator*100,2)}%)</span>.<br>
                        </div>
                        """, unsafe_allow_html=True)

        # st.markdown(f"""
        # <div style="line-height:2;">
        #     _ Number of columns with >20% missing data: <span style="background-color:#e6f2ff; padding:4px 8px; border-radius:6px;">{len(missing_ratio[missing_ratio > 0.2])} ({np.round(len(missing_ratio[missing_ratio > 0.2])/(df.shape[1]-1)*100,2)}%)</span>.<br>
        #     _ Number of columns with >50% missing data: <span style="background-color:#e6f2ff; padding:4px 8px; border-radius:6px;">{len(missing_ratio[missing_ratio > 0.5])} ({np.round(len(missing_ratio[missing_ratio > 0.5])/(df.shape[1]-1)*100,2)}%)</span>.<br>
        #     _ Number of columns with >80% missing data: <span style="background-color:#e6f2ff; padding:4px 8px; border-radius:6px;">{len(missing_ratio[missing_ratio > 0.8])} ({np.round(len(missing_ratio[missing_ratio > 0.8])/(df.shape[1]-1)*100,2)}%)</span>.
        # </div>
        # """, unsafe_allow_html=True)  
        
        
        # col1, col2, col3 = st.columns(3)
        # col1.metric("Columns with >20% missing data", f"{len(missing_ratio[missing_ratio > 0.2])} ({np.round(len(missing_ratio[missing_ratio > 0.2])/(df.shape[1]-1)*100,2)}%)")       
        # col2.metric("Columns with >50% missing data", f"{len(missing_ratio[missing_ratio > 0.5])} ({np.round(len(missing_ratio[missing_ratio > 0.5])/(df.shape[1]-1)*100,2)}%)")       
        # col3.metric("Columns with >80% missing data", f"{len(missing_ratio[missing_ratio > 0.8])} ({np.round(len(missing_ratio[missing_ratio > 0.8])/(df.shape[1]-1)*100,2)}%)")       
            # st.markdown("""
            #     <div style="font-size:14px; line-height:1.5;">
            #     •   missing >20%: <b>{}</b> ({:.2f}%)<br> 
            #     •   missing >50%: <b>{}</b> ({:.2f}%)<br>
            #     •   missing >80%: <b>{}</b> ({:.2f}%)
            #     </div>
            #     """.format(
            #         len(missing_ratio[missing_ratio > 0.2]),
            #         np.round(len(missing_ratio[missing_ratio > 0.2])/(df.shape[1]-1)*100,2),
            #         len(missing_ratio[missing_ratio > 0.5]),
            #         np.round(len(missing_ratio[missing_ratio > 0.5])/(df.shape[1]-1)*100,2),
            #         len(missing_ratio[missing_ratio > 0.8]),
            #         np.round(len(missing_ratio[missing_ratio > 0.8])/(df.shape[1]-1)*100,2)
            #     ), unsafe_allow_html=True)

    missing_df = missing_ratio.reset_index()
    missing_df.columns = ["Feature", "MissingRatio"]
    missing_df["Count"] = df.isna().sum().values
    missing_df = missing_df[missing_df["MissingRatio"] > 0]
    missing_df = missing_df.sort_values(by="MissingRatio", ascending=False)

    # st.dataframe(missing_df)

    colors = []
    for r in missing_df["MissingRatio"]:
        if r > 0.8:
            colors.append("red")
        elif r > 0.5:
            colors.append("salmon")
        elif r > 0.2:
            colors.append("orange")  
        else:
            colors.append("lightgreen")

    if show_plot:
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(figsize=(20,6))
        if missing_df.empty:
            ax.text(0.5, 0.5, "No missing data in selected columns", ha="center", va="center")
            ax.set_axis_off()
        else:
            ax.bar(missing_df["Feature"], missing_df["MissingRatio"]*100, color=colors)
            x1, x2 = ax.get_xlim()
                    
            ax.plot([x1, x2], [80, 80], color='darkred', linestyle='--', label='>80%')
            ax.plot([x1, x2], [50, 50], color='orange', linestyle='--', label='>50%')
            ax.plot([x1, x2], [20, 20], color='green', linestyle='--', label='>20%')

            ax.set_xticks(range(len(missing_df["Feature"])))
            ax.set_xticklabels(missing_df["Feature"], rotation=90)
            ax.legend()
            ax.set_ylim(0,100)
            ax.set_xlim(x1, x2)
        ax.set_ylabel("Missing data ratio (%)")
        ax.set_xlabel("Feature")
        # ax.set_title("")
        plt.tight_layout()
        st.pyplot(fig)
        # plt.show()
    else:
        fig, ax = plt.subplots()

    return missing_df, fig, ax


def describe_dataframe(df):
    missing_ratio = df.isna().mean()
    completed_columns = missing_ratio[missing_ratio == 0].index.tolist()
    missing_columns = missing_ratio[missing_ratio > 0].index.tolist()
    missing_80_columns = missing_ratio[missing_ratio > 0.8].index.tolist()
    missing_50_columns = missing_ratio[missing_ratio > 0.5].index.tolist()
    missing_25_columns = missing_ratio[missing_ratio > 0.25].index.tolist()
    float_columns = df.select_dtypes(include=["float"]).columns.tolist()
    integer_columns = df.select_dtypes(include=["integer"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    high_cardinality_categorical = [
        col for col in categorical_columns
        if df[col].nunique(dropna=True) > 10
    ]

    rows = [
        {
            "Metric": "Data shape",
            "Value": f"{df.shape[0]} x {df.shape[1]}",
            "Description": "Current dataframe size (rows x columns).",
            "Column names": "",
        },
        {
            "Metric": "Number of columns",
            "Value": df.shape[1],
            "Description": "Current number of variables.",
            "Column names": "",
        },
        {
            "Metric": "Number of rows",
            "Value": df.shape[0],
            "Description": "Current number of observations.",
            "Column names": "",
        },
        {
            "Metric": "Duplicated rows",
            "Value": int(df.duplicated().sum()),
            "Description": "Rows with duplicated values across all columns.",
            "Column names": "",
        },
        {
            "Metric": "Duplicated columns",
            "Value": int(df.T.duplicated().sum()),
            "Description": "Columns with duplicated values across all rows.",
            "Column names": "",
        },
        {
            "Metric": "Completed columns",
            "Value": f"{len(completed_columns)} ({len(completed_columns) / max(df.shape[1], 1) * 100:.2f}%)",
            "Description": "Columns with no missing values.",
            "Column names": ", ".join(completed_columns) if completed_columns else "None",
        },
        {
            "Metric": "Columns with missing data",
            "Value": f"{len(missing_columns)} ({len(missing_columns) / max(df.shape[1], 1) * 100:.2f}%)",
            "Description": "Columns containing at least one missing value.",
            "Column names": ", ".join(missing_columns) if missing_columns else "None",
        },
        {
            "Metric": "Columns with >80% missing data",
            "Value": f"{len(missing_80_columns)} ({len(missing_80_columns) / max(df.shape[1], 1) * 100:.2f}%)",
            "Description": "Columns where more than 80% of values are missing.",
            "Column names": ", ".join(missing_80_columns) if missing_80_columns else "None",
        },
        {
            "Metric": "Columns with >50% missing data",
            "Value": f"{len(missing_50_columns)} ({len(missing_50_columns) / max(df.shape[1], 1) * 100:.2f}%)",
            "Description": "Columns where more than 50% of values are missing.",
            "Column names": ", ".join(missing_50_columns) if missing_50_columns else "None",
        },
        {
            "Metric": "Columns with >25% missing data",
            "Value": f"{len(missing_25_columns)} ({len(missing_25_columns) / max(df.shape[1], 1) * 100:.2f}%)",
            "Description": "Columns where more than 25% of values are missing.",
            "Column names": ", ".join(missing_25_columns) if missing_25_columns else "None",
        },
        {
            "Metric": "Numeric columns (float)",
            "Value": len(float_columns),
            "Description": "Floating-point numeric columns.",
            "Column names": ", ".join(float_columns) if float_columns else "None",
        },
        {
            "Metric": "Numeric columns (integer)",
            "Value": len(integer_columns),
            "Description": "Integer numeric columns.",
            "Column names": ", ".join(integer_columns) if integer_columns else "None",
        },
        {
            "Metric": "Categorical columns",
            "Value": len(categorical_columns),
            "Description": "Object, category, or boolean columns.",
            "Column names": ", ".join(categorical_columns) if categorical_columns else "None",
        },
        {
            "Metric": "High-cardinality categorical columns (>10 values)",
            "Value": len(high_cardinality_categorical),
            "Description": "Categorical columns with more than 10 unique non-missing values.",
            "Column names": ", ".join(high_cardinality_categorical) if high_cardinality_categorical else "None",
        },
    ]

    return pd.DataFrame(rows)

def train_model(df_X_origin, df_y, seed=42):
    if df_y.nunique() != 2:
        raise ValueError("CatBoost training currently supports binary labels only.")

    X_train, X_test, y_train, y_test = train_test_split(df_X_origin, 
                                                            df_y, 
                                                            test_size=0.2, 
                                                            stratify=df_y, 
                                                            random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.25,
                                                            stratify=y_train,
                                                            random_state=seed)
    
    model = CatBoostClassifier(random_seed=seed,
                                early_stopping_rounds=100,
                                use_best_model=True,
                                verbose=0,
                                task_type='CPU',
                                devices="0"
                                )
    cat_cols = cat_identify(df_X_origin)
    train_pool = catboost.Pool(X_train,y_train, 
                                cat_features=cat_cols
                                )
    eval_pool = catboost.Pool(X_valid,y_valid, 
                                cat_features=cat_cols
                                )
    model.fit(train_pool, eval_set=eval_pool)

    adjusted_threshold = sum(y_train.values)/len(y_train)
    
    return model, X_test, y_test, adjusted_threshold 

def threshold_check(y_test, y_test_proba, model_name=None, task_label=None):
    tn_list       = []
    fp_list       = []
    fn_list       = []
    tp_list       = []
    sensitivity_list     = []
    specificity_list= []
    npv_list        = []
    f1_list         = []
    mcc_list        = []
    # plt.figure(figsize=[50,50])
    for i in np.arange(0,101,2):
        y_pred = np.where(y_test_proba >= i/100, 1, 0)
        cm, sensitivity, specificity, npv, f1, mcc = evaluation_metric(y_test, y_pred)
        # plt.subplot(10,10,i+1)
        # plt.title(f"{np.round(i/100, 2)}")
        # plt.suptitle(f"{model_name} {task_label}")
        # cm = value_to_percentage(cm)
        # plot_confusion_matrix(cm, classes=["normal", "apnea"])
        # acc_list.append(accuracy)
        # pre_list.append(precision)
        tn, fp, fn, tp = [int(value) for value in cm.ravel()]
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        tp_list.append(tp)
        specificity_list.append(specificity)
        npv_list.append(npv)
        sensitivity_list.append(sensitivity)
        f1_list.append(f1)
        mcc_list.append(mcc)
    return sensitivity_list, specificity_list, npv_list, f1_list, mcc_list, tn_list, fp_list, fn_list, tp_list

def evaluation_metric(y_test,y_test_pred):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score, f1_score, matthews_corrcoef
    cm = confusion_matrix(y_test,y_test_pred)
    ## --------------------- for binary class ------------------ ###
    tn, fp, fn, tp = [int(i) for i in cm.ravel()]
    try:
        sensitivity      = recall_score(y_test, y_test_pred)           #tp / (tp + fn)  # Sensitivity
    except ZeroDivisionError:
        sensitivity      = np.nan

    try:
        specificity = (tn / (tn + fp))
    except ZeroDivisionError:
        specificity = np.nan

    try:
        npv         = (tn / (tn + fn))
    except ZeroDivisionError:
        npv         = np.nan

    try:
        f1          = f1_score(y_test, y_test_pred)
    except ZeroDivisionError:
        f1          = np.nan

    try:
        mcc         = matthews_corrcoef(y_test, y_test_pred)
    except ZeroDivisionError:
        mcc         = np.nan

    return cm, sensitivity, specificity, npv, f1, mcc

def apply_pipeline(df, pipeline):
    df = df.copy()
    for step in pipeline:
        cols = step["column"]
        action = step["action"]

        if not cols:
            continue  # skip empty selection

        for col in cols:
            if col not in df.columns:
                continue

            if action == "drop":
                df = df.drop(columns=[col])

            elif action == "fill_value":
                fill_value = step["value"]
                if isinstance(fill_value, dict):
                    fill_value = fill_value.get(col)
                df[col] = df[col].fillna(fill_value)

            elif action == "fill_max":
                if df[col].dtype != "object":
                    df[col] = df[col].fillna(df[col].max())

            elif action == "fill_min":
                if df[col].dtype != "object":
                    df[col] = df[col].fillna(df[col].min())

            elif action == "fill_median":
                if df[col].dtype != "object":
                    df[col] = df[col].fillna(df[col].median())

            elif action == "fill_mean":
                if df[col].dtype != "object":
                    df[col] = df[col].fillna(df[col].mean())

            elif action == "fill_mode":
                mode_values = df[col].mode(dropna=True)
                if not mode_values.empty:
                    df[col] = df[col].fillna(mode_values.iloc[0])

    return df


def prepare_binary_label(series):
    labels = series.dropna().unique()
    if len(labels) != 2:
        raise ValueError("Please select a binary label column with exactly 2 classes after missing rows are removed.")

    sorted_labels = sorted(labels, key=str)
    label_mapping = {sorted_labels[0]: 0, sorted_labels[1]: 1}
    return series.map(label_mapping), label_mapping


def find_candidate_label_columns(
    df,
    max_unique=LABEL_SCAN_MAX_UNIQUE,
    max_unique_ratio=LABEL_SCAN_MAX_UNIQUE_RATIO,
):
    candidate_rows = []
    row_count = len(df)

    for col in df.columns:
        non_missing = df[col].dropna()
        non_missing_count = len(non_missing)
        unique_count = non_missing.nunique(dropna=True)

        if non_missing_count < LABEL_SCAN_MIN_ROWS:
            continue
        if unique_count < 2 or unique_count > max_unique:
            continue
        if row_count > 0 and unique_count / row_count > max_unique_ratio:
            continue

        candidate_rows.append({
            "Column": col,
            "Unique values": int(unique_count),
            "Non-missing rows": int(non_missing_count),
            "Missing %": float((row_count - non_missing_count) / max(row_count, 1) * 100),
        })

    if not candidate_rows:
        return pd.DataFrame(columns=["Column", "Unique values", "Non-missing rows", "Missing %"])

    return pd.DataFrame(candidate_rows).sort_values(
        by=["Unique values", "Missing %", "Column"],
        ascending=[True, True, True],
    )


def make_label_scan_features(df, label_col, excluded_label_columns):
    excluded_columns = [
        col for col in excluded_label_columns
        if col in df.columns and col != label_col
    ]
    feature_df = df.drop(columns=[label_col] + excluded_columns).copy()
    numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_columns:
        median_value = feature_df[col].median()
        if pd.isna(median_value):
            median_value = 0
        feature_df[col] = feature_df[col].fillna(median_value)

    for col in feature_df.columns:
        if col not in numeric_columns:
            feature_df[col] = (
                feature_df[col]
                .astype("object")
                .where(feature_df[col].notna(), "__missing__")
                .astype(str)
            )
    feature_df = pd.get_dummies(feature_df, dummy_na=False)
    return feature_df


def format_class_counts(class_counts):
    return ", ".join(
        f"{label}: {count}"
        for label, count in class_counts.items()
    )


def macro_specificity_score(y_true, y_pred):
    labels = sorted(pd.Series(y_true).dropna().unique(), key=str)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity_values = []

    for index, _ in enumerate(labels):
        tp = cm[index, index]
        fp = cm[:, index].sum() - tp
        fn = cm[index, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        denominator = tn + fp
        if denominator > 0:
            specificity_values.append(tn / denominator)

    return float(np.mean(specificity_values)) if specificity_values else np.nan


def scan_label_candidate(df, label_col, excluded_label_columns, seed=42):
    scan_df = df.dropna(subset=[label_col]).copy()
    if len(scan_df) > LABEL_SCAN_MAX_ROWS:
        scan_df = scan_df.sample(n=LABEL_SCAN_MAX_ROWS, random_state=seed)

    y = scan_df[label_col]
    class_counts = y.value_counts(dropna=True)
    if len(class_counts) < 2:
        raise ValueError("fewer than 2 classes")
    if class_counts.min() < LABEL_SCAN_MIN_CLASS_COUNT:
        raise ValueError(f"smallest class has fewer than {LABEL_SCAN_MIN_CLASS_COUNT} rows")

    X = make_label_scan_features(scan_df, label_col, excluded_label_columns)
    if X.shape[1] == 0:
        raise ValueError("no feature columns remain")

    test_size = max(0.2, len(class_counts) / len(y))
    if test_size >= 0.5:
        raise ValueError("not enough rows per class for a holdout split")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.ravel(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    recall_macro = recall_score(y_test, y_pred, average="macro")
    specificity_macro = macro_specificity_score(y_test, y_pred)
    if accuracy >= 0.99 or f1_macro >= 0.99:
        flag = "Perfect / possible leakage"
    elif accuracy >= 0.95 or f1_macro >= 0.95:
        flag = "Near-perfect"
    else:
        flag = ""

    return {
        "Column": label_col,
        "Accuracy": accuracy,
        "F1 macro": f1_macro,
        "Recall": recall_macro,
        "Specificity": specificity_macro,
        "Test size [rows x columns]": f"{X_test.shape[0]} x {X_test.shape[1]}",
        "Number of each class": format_class_counts(class_counts),
        "Flag": flag,
        "Error": "",
    }


def scan_candidate_labels(df, candidate_columns, excluded_label_columns=None, seed=42):
    if excluded_label_columns is None:
        excluded_label_columns = candidate_columns

    rows = []
    for label_col in candidate_columns:
        try:
            rows.append(scan_label_candidate(df, label_col, excluded_label_columns, seed=seed))
        except Exception as error:
            excluded_feature_columns = [
                col for col in excluded_label_columns
                if col in df.columns and col != label_col
            ]
            rows.append({
                "Column": label_col,
                "Accuracy": np.nan,
                "F1 macro": np.nan,
                "Recall": np.nan,
                "Specificity": np.nan,
                "Test size [rows x columns]": "",
                "Number of each class": (
                    format_class_counts(df[label_col].value_counts(dropna=True))
                    if label_col in df.columns else ""
                ),
                "Flag": "Skipped",
                "Error": str(error),
            })

    return pd.DataFrame(rows).sort_values(
        by=["F1 macro", "Accuracy"],
        ascending=[False, False],
        na_position="last",
    )


def find_id_columns(df):
    id_columns = []
    row_count = len(df)

    for col in df.columns:
        normalized_name = str(col).lower().replace("-", "_").replace(" ", "_")
        looks_like_id = (
            normalized_name == "id"
            or normalized_name.endswith("_id")
            or normalized_name.startswith("id_")
            or "subject_id" in normalized_name
            or "participant_id" in normalized_name
        )
        mostly_unique = row_count > 0 and df[col].nunique(dropna=True) / row_count >= 0.95

        if looks_like_id or (mostly_unique and "id" in normalized_name):
            id_columns.append(col)

    return id_columns


def default_preprocessing_pipeline(id_columns, missing_50_columns):
    return [
        make_remove_id_step(id_columns),
        make_remove_missing_step(missing_50_columns, 50),
    ]


def make_remove_id_step(id_columns):
    return {
        "id": next_pipeline_step_id("remove_id"),
        "name": "Remove detected ID columns",
        "description": "Automatically detects likely ID columns, such as id, subject_id, participant_id, or columns ending with _id, and removes them before model training.",
        "column": id_columns,
        "action": "drop",
        "source": "detected_id",
        "value": "",
    }


def make_remove_missing_step(missing_columns, threshold_percent=None):
    if threshold_percent is None:
        name = "Remove manually selected missing-data columns"
        description = "Removes the missing-data columns manually selected by the user."
        id_prefix = "remove_missing_manual"
        source = "manual_missing"
    else:
        name = f"Remove columns with >{threshold_percent}% missing data"
        description = f"Removes features where more than {threshold_percent}% of the values are missing. This step can be removed if you prefer to impute or keep these columns."
        id_prefix = f"remove_missing_{int(threshold_percent)}"
        source = "missing_threshold"

    return {
        "id": next_pipeline_step_id(id_prefix),
        "name": name,
        "description": description,
        "column": missing_columns,
        "action": "drop",
        "source": source,
        "threshold_percent": threshold_percent,
        "value": "",
    }


def make_custom_pipeline_step(columns, action="drop", value=""):
    if isinstance(columns, str):
        columns = [columns]

    return {
        "id": next_pipeline_step_id("custom"),
        "column": columns,
        "action": action,
        "source": "manual",
        "value": value,
    }


def missing_columns_above_threshold(missing_df, threshold_percent):
    threshold_ratio = threshold_percent / 100
    return [
        row["Feature"]
        for _, row in missing_df.iterrows()
        if row["MissingRatio"] > threshold_ratio
    ]


def describe_pipeline_action(step):
    action = step.get("action", "drop")

    if action == "drop":
        source = step.get("source", "")
        if source == "detected_id":
            return "Drop (detected ID columns)"
        if source == "missing_threshold":
            return f"Drop (missing data threshold >{step.get('threshold_percent')}%)"
        if source == "manual_missing":
            return "Drop (manual missing-data selection)"
        if source == "manual":
            return "Drop (user manual selection)"
        return "Drop"
    if action == "fill_value":
        return f"Replace missing data with {step.get('value', '')}"
    if action == "fill_max":
        return "Replace missing data with maximum value"
    if action == "fill_min":
        return "Replace missing data with minimum value"
    if action == "fill_median":
        return "Replace missing data with median value"
    if action == "fill_mean":
        return "Replace missing data with average value"
    if action == "fill_mode":
        return "Replace missing data with mode value"

    return ACTION_LABELS.get(action, action)


def format_fill_value(value):
    if pd.isna(value):
        return "NaN"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def is_categorical_column(df, col):
    return col in df.columns and (
        df[col].dtype == "object"
        or str(df[col].dtype) == "category"
        or df[col].dtype == "bool"
    )


def has_categorical_columns(df, columns):
    return any(is_categorical_column(df, col) for col in columns)


def computed_fill_values(df, columns, action):
    values = {}
    for col in columns:
        if col not in df.columns or df[col].dtype == "object":
            continue

        if action == "fill_max":
            values[col] = df[col].max()
        elif action == "fill_min":
            values[col] = df[col].min()
        elif action == "fill_median":
            values[col] = df[col].median()
        elif action == "fill_mean":
            values[col] = df[col].mean()

    return values


def describe_affected_columns(step, df):
    columns = step.get("column", [])
    action = step.get("action", "drop")

    if not columns:
        return "(none selected)"

    if action == "fill_value":
        fill_value = step.get("value", "")
        if isinstance(fill_value, dict):
            return ", ".join(
                f"[{col}: {format_fill_value(fill_value.get(col, ''))}]"
                for col in columns
            )

        return ", ".join(
            f"[{col}: {format_fill_value(fill_value)}]"
            for col in columns
        )

    if action in ["fill_max", "fill_min", "fill_median", "fill_mean"]:
        fill_values = computed_fill_values(df, columns, action)
        return ", ".join(
            f"[{col}: {format_fill_value(fill_values[col])}]"
            if col in fill_values else f"[{col}: unavailable]"
            for col in columns
        )

    return ", ".join(columns)


def pipeline_to_dataframe(pipeline, df):
    rows = []
    for index, step in enumerate(pipeline, start=1):
        rows.append({
            "Step ID": index,
            "Action": describe_pipeline_action(step),
            "Affected columns": describe_affected_columns(step, df),
        })
    return pd.DataFrame(rows)


def parse_step_selection(selection_text, max_step):
    selected_steps = set()

    for part in selection_text.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start > end:
                start, end = end, start
            selected_steps.update(range(start, end + 1))
        else:
            selected_steps.add(int(part))

    return sorted(step for step in selected_steps if 1 <= step <= max_step)


def ensure_pipeline_step_ids(pipeline):
    for index, step in enumerate(pipeline):
        step.setdefault("id", f"pipeline_step_{index}")
    return pipeline


def next_pipeline_step_id(prefix="custom"):
    counter_key = f"{SESSION_PREFIX}_pipeline_step_counter"
    st.session_state[counter_key] = st.session_state.get(counter_key, 0) + 1
    return f"{prefix}_step_{st.session_state[counter_key]}"
# class TrainModelPage:
#     def __init__(self, df_x, df_y):
#         # self.dataset_name = dataset_name
#         self.df_x = df_x
#         self.df_y = df_y

#     def display_training_page(self): 
#         ### ------------------------- Display model information ------------------------------ ###
#         with st.expander("Upload your dataset here."):
#             st.markdown(f"<h4>Model training page.</h4>", unsafe_allow_html=True)
            
#             uploaded_file = st.file_uploader("Upload your data file", type=["csv"])

#             if uploaded_file is not None:


#                 df_uploaded = pd.read_csv(uploaded_file)
#                 st.write("Uploaded Data:")
#                 st.dataframe(df_uploaded)

#                 check_missing_data(df_uploaded)

#                 label = st.selectbox(
#                         "Select label column:",
#                         options=df_uploaded.columns,
#                         index=len(df_uploaded.columns)-1
#                     )
                
#                 all_features = list(df_uploaded.columns)        
#                 all_features.remove(label)

#                 # Initialize
#                 if "selected_features" not in st.session_state:
#                     st.session_state.selected_features = all_features.copy()
#                 # else:
#                 #     st.session_state.selected_features = all_features.copy()

#                 # Selection box for features
#                 selected = st.multiselect(
#                     "Selected features:",
#                     all_features,
#                     default=st.session_state.selected_features
#                 )

#                 # Save selection
#                 st.session_state.selected_features = selected
#                 removed_features = list(set(all_features) - set(selected))

#                 # Display removed features (disabled)
#                 st.multiselect(
#                     "Removed features:",
#                     removed_features,
#                     default=removed_features,
#                     disabled=True
#                 )

#         # cols = st.columns(2)
#         # with cols[0]:
#             if st.button("Confirm features"):
#                 # st.markdown("### Selected features")
#                 # st.markdown(
#                 #     " ".join([f"`{f}`" for f in st.session_state.selected_features])
#                 # )
#                 df_uploaded_dropped = df_uploaded.dropna(subset=st.session_state.selected_features + [label])
#                 st.session_state.df_x_uploaded = df_uploaded_dropped[st.session_state.selected_features]
#                 st.session_state.df_y_uploaded = df_uploaded_dropped[label]
#         # with cols[1]:
#         if "df_x_uploaded" not in st.session_state:
#             st.warning("Please confirm features first.")
#         else:
#             st.success("Features confirmed!")

#         if st.button("Train model"):
#             if "df_x_uploaded" not in st.session_state:
#                 a=0
#             else:
#                 with st.spinner("Training model..."):
#                     model, X_test, y_test, adjusted_threshold = train_model(
#                         st.session_state.df_x_uploaded,
#                         st.session_state.df_y_uploaded,
#                         seed=random_seeds[0]
#                     )
#                     st.success("Model trained!")
#                     # st.write(f"Adjusted threshold: {np.round(adjusted_threshold, 2)}")

import streamlit as st
import pandas as pd
import numpy as np

class TrainModelPage:
    def __init__(self, df_x, df_y):
        self.df_x = df_x
        self.df_y = df_y

    def display_training_page(self):

        # st.markdown("## 🧠 Model Training")
        with st.expander("Upload your dataset here."):
            # ---------------- Upload ----------------
            uploaded_file = st.file_uploader("", type=["csv"])
            use_demo_dataset_key = f"{SESSION_PREFIX}_use_demo_dataset"

            if uploaded_file is None:
                st.info("You can use default dataset for demonstration. Please upload your own datset if you want to train the model for yourself")
                if st.button("Load default demo dataset", use_container_width=True):
                    st.session_state[use_demo_dataset_key] = True
                if not st.session_state.get(use_demo_dataset_key, False):
                    return
            else:
                st.session_state[use_demo_dataset_key] = False

            if uploaded_file is not None:
                df_uploaded = pd.read_csv(uploaded_file)
                uploaded_signature = (uploaded_file.name, df_uploaded.shape)
            else:
                demo_dataset_path = os.path.join(DATA_DIR, "df_test.csv")
                df_uploaded = pd.read_csv(demo_dataset_path)
                uploaded_signature = ("df_test.csv", df_uploaded.shape)

            st.dataframe(df_uploaded)
            if st.session_state.get(f"{SESSION_PREFIX}_uploaded_signature") != uploaded_signature:
                st.session_state[f"{SESSION_PREFIX}_uploaded_signature"] = uploaded_signature
                st.session_state.pop(f"{SESSION_PREFIX}_df_processed", None)
                st.session_state.pop(f"{SESSION_PREFIX}_selected_features", None)
                st.session_state.pop(f"{SESSION_PREFIX}_confirmed", None)
                st.session_state.pop(f"{SESSION_PREFIX}_df_x_uploaded", None)
                st.session_state.pop(f"{SESSION_PREFIX}_df_y_uploaded", None)
                st.session_state.pop(f"{SESSION_PREFIX}_label_mapping", None)
                st.session_state.pop(f"{SESSION_PREFIX}_pipeline", None)
                st.session_state.pop(f"{SESSION_PREFIX}_pipeline_defaults_signature", None)
                st.session_state.pop(f"{SESSION_PREFIX}_pipeline_step_counter", None)
                st.session_state.pop(f"{SESSION_PREFIX}_show_missing_step_config", None)
                st.session_state.pop(f"{SESSION_PREFIX}_show_other_step_config", None)
                st.session_state.pop(f"{SESSION_PREFIX}_manual_missing_columns", None)
                st.session_state.pop(f"{SESSION_PREFIX}_missing_reset_counter", None)
                st.session_state.pop(f"{SESSION_PREFIX}_other_reset_counter", None)
                st.session_state.pop(f"{SESSION_PREFIX}_label_scan_results", None)
                st.session_state.pop(f"{SESSION_PREFIX}_flagged_feature_columns", None)
                st.session_state.pop(f"{SESSION_PREFIX}_flagged_feature_version", None)
                st.session_state.pop(f"{SESSION_PREFIX}_flagged_feature_applied_version", None)

        df_processed = df_uploaded.copy()
        original_missing, _, _ = check_missing_data(df_processed)
        list_missing_50 = [row["Feature"] for _, row in original_missing.iterrows() if row["MissingRatio"] > 0.5]
        list_id_columns = find_id_columns(df_processed)

        # ---------------- Quick data preprocessing ----------------
        # Initialize pipeline
        pipeline_key = f"{SESSION_PREFIX}_pipeline"
        pipeline_defaults_key = f"{SESSION_PREFIX}_pipeline_defaults_signature"
        default_pipeline_signature = (
            st.session_state.get(f"{SESSION_PREFIX}_uploaded_signature"),
            tuple(list_id_columns),
            tuple(list_missing_50),
        )
        if (
            pipeline_key not in st.session_state
            or pipeline_defaults_key not in st.session_state
        ):
            st.session_state[pipeline_key] = default_preprocessing_pipeline(
                list_id_columns,
                list_missing_50
            )
            st.session_state[pipeline_defaults_key] = default_pipeline_signature
        st.session_state[pipeline_key] = ensure_pipeline_step_ids(st.session_state[pipeline_key])

        st.markdown("#### Quick preprocessing")
        st.markdown("""This interface supports several quick and simple data cleaning functions, 
                    including dropping missing values, replacing them with specified values, 
                    and imputing with median (for numerical features) or mode (for categorical features).""")
        st.caption(
            "The cleaning pipeline starts with two default steps: remove detected ID columns "
            "and remove columns with more than 50% missing data. You can remove either step with the X button."
        )
        
        columns = df_uploaded.columns.tolist()
        with st.expander("Data preprocessing options", expanded=True):
            missing_reset_counter = st.session_state.get(f"{SESSION_PREFIX}_missing_reset_counter", 0)
            other_reset_counter = st.session_state.get(f"{SESSION_PREFIX}_other_reset_counter", 0)

            tab_remove_id, tab_remove_columns, tab_other = st.tabs([
                "Remove ID column (if detected)",
                "Remove columns",
                "+ Other",
            ])

            with tab_remove_id:
                st.info(
                    "Use this option to remove likely identifier columns before model training. "
                    "Detected ID columns are based on common names such as id, subject_id, participant_id, or columns ending with _id."
                )
                if list_id_columns:
                    st.multiselect(
                        "Detected ID columns",
                        list_id_columns,
                        default=list_id_columns,
                        disabled=True,
                        key=f"{SESSION_PREFIX}_detected_id_preview"
                    )
                else:
                    st.caption("No likely ID columns were detected in the uploaded dataset.")

                if st.button("Add ID removal step", use_container_width=True):
                    st.session_state[pipeline_key].append(make_remove_id_step(list_id_columns))
                    st.rerun()

            with tab_remove_columns:
                st.info(
                    "You can auto-select columns automatically by defining a missing-data percentage threshold, or manually pick columns from the uploaded dataset."
                    "Use this option when you want to remove columns entirely before model training. "
                    "If you want to impute missing values instead of removing columns, please add a custom step in the + Other tab and choose the appropriate action."
                )

                missing_mode = st.radio(
                    "How do you want to choose columns?",
                    ["Use missing threshold", "Pick columns manually"],
                    horizontal=True,
                    key=f"{SESSION_PREFIX}_missing_step_mode_{missing_reset_counter}"
                )

                if missing_mode == "Use missing threshold":
                    missing_threshold_percent = st.number_input(
                        "Missing threshold (%)",
                        min_value=0,
                        max_value=100,
                        value=90,
                        step=5,
                        key=f"{SESSION_PREFIX}_missing_threshold_percent_{missing_reset_counter}"
                    )
                    selected_missing_columns = missing_columns_above_threshold(
                        original_missing,
                        missing_threshold_percent
                    )
                    
                    if selected_missing_columns:
                        selected_missing_columns = st.multiselect(
                            f"{len(selected_missing_columns)} columns match this threshold and will be removed (Unselect any column names from this list if you want to keep those columns.)",
                            selected_missing_columns,
                            default=selected_missing_columns,
                            key=f"{SESSION_PREFIX}_threshold_missing_preview_{missing_reset_counter}_{missing_threshold_percent}"
                        )
                    else:
                        st.info("No columns match the current missing-data threshold.")

                    if st.button("Confirm dropping above columns", use_container_width=True):
                        st.session_state[pipeline_key].append(
                            make_remove_missing_step(selected_missing_columns, missing_threshold_percent)
                        )
                        st.session_state[f"{SESSION_PREFIX}_missing_reset_counter"] = missing_reset_counter + 1
                        st.rerun()
                else:
                    missing_feature_options = original_missing["Feature"].tolist()
                    selected_missing_columns = st.multiselect(
                        "Pick columns to remove",
                        missing_feature_options,
                        key=f"{SESSION_PREFIX}_manual_missing_columns_{missing_reset_counter}"
                    )
                    if st.button("Add manual missing-data step"):
                        st.session_state[pipeline_key].append(
                            make_remove_missing_step(selected_missing_columns)
                        )
                        st.session_state[f"{SESSION_PREFIX}_missing_reset_counter"] = missing_reset_counter + 1
                        st.rerun()

            with tab_other:
                st.info(
                    "Use this option to add a custom cleaning step. Select one or more columns, choose the action, "
                    "then add the step to Current pipeline. If selected columns include categorical data, only drop or replace with a value are available."
                )
                other_columns = st.multiselect(
                    "Columns",
                    columns,
                    key=f"{SESSION_PREFIX}_other_columns_{other_reset_counter}"
                )
                contains_categorical_columns = has_categorical_columns(df_processed, other_columns)
                other_action_options = ["fill_value", "drop"] if contains_categorical_columns else [
                    "fill_value", "drop", "fill_max", "fill_min", "fill_median", "fill_mean"
                ]
                if contains_categorical_columns:
                    st.caption("Categorical columns selected: only drop or replace with a value are available.")

                col_other_action, col_other_number = st.columns(2)
                with col_other_action:
                    other_action = st.selectbox(
                        "Action",
                        other_action_options,
                        format_func=lambda action: ACTION_LABELS[action],
                        key=f"{SESSION_PREFIX}_other_action_{other_reset_counter}_{contains_categorical_columns}"
                )
                other_value = ""
                with col_other_number:
                    if other_action == "fill_value":
                        categorical_columns = [
                            col for col in other_columns
                            if is_categorical_column(df_processed, col)
                        ]
                        numeric_columns = [
                            col for col in other_columns
                            if col not in categorical_columns
                        ]

                        fill_values = {}
                        if numeric_columns:
                            numeric_fill_value = st.number_input(
                                "Number for numeric columns",
                                value=0.0,
                                key=f"{SESSION_PREFIX}_other_fill_number_{other_reset_counter}"
                            )
                            fill_values.update({
                                col: numeric_fill_value
                                for col in numeric_columns
                            })

                        for col in categorical_columns:
                            category_options = df_processed[col].dropna().unique().tolist()
                            if category_options:
                                fill_values[col] = st.selectbox(
                                    f"Category for {col}",
                                    category_options,
                                    key=f"{SESSION_PREFIX}_other_fill_category_{other_reset_counter}_{col}"
                                )
                            else:
                                st.caption(f"{col} has no non-missing category values to choose from.")

                        other_value = fill_values if fill_values else ""
                    elif other_action in ["fill_max", "fill_min", "fill_median", "fill_mean"]:
                        st.write("Value")
                        fill_values = computed_fill_values(df_processed, other_columns, other_action)
                        if fill_values:
                            st.dataframe(
                                pd.DataFrame(
                                    fill_values.items(),
                                    columns=["Column", "Value"]
                                ),
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.caption("Select numeric columns to preview computed values.")
                    else:
                        st.write("")

                if st.button("Add preprocessing step", disabled=len(other_columns) == 0):
                    st.session_state[pipeline_key].append(
                        make_custom_pipeline_step(other_columns, other_action, other_value)
                    )
                    st.session_state[f"{SESSION_PREFIX}_other_reset_counter"] = other_reset_counter + 1
                    st.rerun()

        with st.expander("Current pipeline", expanded=False):
            if st.session_state[pipeline_key]:
                step_number_to_id = {
                    index: step["id"]
                    for index, step in enumerate(st.session_state[pipeline_key], start=1)
                }
                tab_remove_step, tab_reorder_step = st.tabs([
                    "Remove a pipeline step",
                    "Re-order pipeline step",
                ])

                with tab_remove_step:
                    remove_step_ids = []
                    col_remove_mode, col_remove_value, col_button = st.columns(
                        3,
                        vertical_alignment="bottom",
                    )
                    with col_remove_mode:
                        remove_mode = st.selectbox(
                            "Step to remove",
                            ["Single step", "Remove all", "Custom"],
                            key=f"{SESSION_PREFIX}_remove_pipeline_step_mode"
                        )

                    with col_remove_value:
                        if remove_mode == "Single step":
                            remove_step_number = st.selectbox(
                                "Select one step",
                                list(step_number_to_id.keys()),
                                format_func=lambda step_number: f"Step {step_number}",
                                key=f"{SESSION_PREFIX}_remove_pipeline_step_id"
                            )
                            remove_step_ids = [step_number_to_id[remove_step_number]]
                        elif remove_mode == "Remove all":
                            st.warning("This will remove every preprocessing step from the pipeline.")
                            remove_step_ids = list(step_number_to_id.values())
                        else:
                            custom_selection = st.text_input(
                                "Custom steps",
                                placeholder="Example: 1,2,5-8",
                                key=f"{SESSION_PREFIX}_remove_pipeline_custom_steps"
                            )
                            try:
                                custom_step_numbers = parse_step_selection(
                                    custom_selection,
                                    max(step_number_to_id)
                                )
                                remove_step_ids = [
                                    step_number_to_id[step_number]
                                    for step_number in custom_step_numbers
                                ]
                                # if custom_selection:
                                #     st.caption(f"Selected steps: {', '.join(map(str, custom_step_numbers)) or 'none'}")
                            except ValueError:
                                remove_step_ids = []
                                st.error("Use numbers and ranges only, for example: 1,2,5-8")

                    with col_button:
                        if st.button(
                            "Remove selected step",
                            disabled=len(remove_step_ids) == 0,
                            use_container_width=True,
                        ):
                            st.session_state[pipeline_key] = [
                                pipeline_step
                                for pipeline_step in st.session_state[pipeline_key]
                                if pipeline_step["id"] not in remove_step_ids
                            ]
                            st.rerun()

                with tab_reorder_step:
                    col_move_from, col_move_to, col_button = st.columns(
                        3,
                        vertical_alignment="bottom",
                    )
                    with col_move_from:
                        move_step_number = st.selectbox(
                            "Step to move",
                            list(step_number_to_id.keys()),
                            format_func=lambda step_number: f"Step {step_number}",
                            key=f"{SESSION_PREFIX}_move_pipeline_step_id"
                        )
                    with col_move_to:
                        target_step_number = st.selectbox(
                            "New position",
                            list(step_number_to_id.keys()),
                            format_func=lambda step_number: f"Position {step_number}",
                            key=f"{SESSION_PREFIX}_move_pipeline_target_position"
                        )

                    with col_button:
                        if st.button("Move selected step", use_container_width=True):
                            pipeline = st.session_state[pipeline_key]
                            step_to_move = pipeline.pop(move_step_number - 1)
                            pipeline.insert(target_step_number - 1, step_to_move)
                            st.session_state[pipeline_key] = pipeline
                            st.rerun()
            else:
                st.info("No preprocessing steps are currently in the pipeline.")

            pipeline_df = pipeline_to_dataframe(st.session_state[pipeline_key], df_processed)
            st.dataframe(
                pipeline_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Step ID": st.column_config.NumberColumn(
                        "Step ID",
                        width="small",
                    ),
                    "Action": st.column_config.TextColumn(
                        "Action",
                        width="medium",
                    ),
                    "Affected columns": st.column_config.TextColumn(
                        "Affected columns",
                        width="large",
                    ),
                }
            )
            if st.button("✅ Apply preprocessing pipeline"):
                df_processed_test = apply_pipeline(df_processed.copy(), st.session_state[pipeline_key])
                st.session_state[f"{SESSION_PREFIX}_df_processed"] = df_processed_test
                st.session_state.pop(f"{SESSION_PREFIX}_selected_features", None)
                st.session_state.pop(f"{SESSION_PREFIX}_confirmed", None)
                st.session_state.pop(f"{SESSION_PREFIX}_df_x_uploaded", None)
                st.session_state.pop(f"{SESSION_PREFIX}_df_y_uploaded", None)
                st.session_state.pop(f"{SESSION_PREFIX}_label_mapping", None)
                st.session_state.pop(f"{SESSION_PREFIX}_label_scan_results", None)
                st.session_state.pop(f"{SESSION_PREFIX}_flagged_feature_columns", None)
                st.session_state.pop(f"{SESSION_PREFIX}_flagged_feature_version", None)
                st.session_state.pop(f"{SESSION_PREFIX}_flagged_feature_applied_version", None)
                st.success("Pipeline applied successfully!")

        df_afterprocessed = st.session_state.get(f"{SESSION_PREFIX}_df_processed", df_processed.copy())

        st.markdown("#### Dataset summary")
        st.dataframe(
            describe_dataframe(df_afterprocessed),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="small"),
                "Description": st.column_config.TextColumn("Description", width="large"),
                "Column names": st.column_config.TextColumn("Column names", width="large"),
            }
        )

        # ------------------ Missing data check ----------------
        st.markdown("#### Checking columns with missing data")
        df_missing, _, _ = check_missing_data(df_afterprocessed)
        list_missing_80 = [row["Feature"] for _, row in df_missing.iterrows() if row["MissingRatio"] > 0.8]
        list_missing_50_current = [row["Feature"] for _, row in df_missing.iterrows() if row["MissingRatio"] > 0.5]
        list_missing_20 = [row["Feature"] for _, row in df_missing.iterrows() if row["MissingRatio"] > 0.2]

        missing_label = [f"All columns", f"missing >20% data", f"missing >50% data", f"missing >80% data"]
        select_missing = st.radio(
                                    "",
                                    options=missing_label,
                                    index=0,
                                    label_visibility="collapsed",
                                    key=f"{SESSION_PREFIX}_missing_filter",
                                    horizontal=True
                                )
        if select_missing == missing_label[0]:
            check_missing_data(df_afterprocessed, show_plot=True)
        elif select_missing == missing_label[3]:
            check_missing_data(df_afterprocessed[list_missing_80], show_plot=True)
        elif select_missing == missing_label[2]:
            check_missing_data(df_afterprocessed[list_missing_50_current], show_plot=True)
        elif select_missing == missing_label[1]:
            check_missing_data(df_afterprocessed[list_missing_20], show_plot=True)

        with st.expander("Feature selection", expanded=False):
            candidate_label_df = find_candidate_label_columns(df_afterprocessed)
            st.markdown("##### Candidate label check")
            st.caption(
                "This checks low-cardinality columns as possible labels by training quick logistic regression models. "
                "Other candidate-label columns are excluded from the features during this check. "
                "Perfect or near-perfect scores can indicate an easy label, but they can also mean target leakage."
            )
            if candidate_label_df.empty:
                st.info("No candidate label columns were found. Candidate labels need 2-10 unique non-missing values.")
            else:
                candidate_label_options = candidate_label_df["Column"].head(LABEL_SCAN_MAX_CANDIDATES).tolist()
                if st.button(
                    "Run candidate label check",
                    disabled=len(candidate_label_options) == 0,
                    use_container_width=True,
                ):
                    with st.spinner("Testing candidate labels..."):
                        label_scan_results = scan_candidate_labels(
                            df_afterprocessed,
                            candidate_label_options,
                            excluded_label_columns=candidate_label_df["Column"].tolist(),
                            seed=random_seeds[0],
                        )
                        flagged_feature_columns = label_scan_results.loc[
                            label_scan_results["Flag"].fillna("") != "",
                            "Column",
                        ].tolist()
                        st.session_state[f"{SESSION_PREFIX}_label_scan_results"] = label_scan_results
                        st.session_state[f"{SESSION_PREFIX}_flagged_feature_columns"] = flagged_feature_columns
                        st.session_state[f"{SESSION_PREFIX}_flagged_feature_version"] = (
                            st.session_state.get(f"{SESSION_PREFIX}_flagged_feature_version", 0) + 1
                        )

                label_scan_results = st.session_state.get(f"{SESSION_PREFIX}_label_scan_results")
                if label_scan_results is not None:
                    display_results = label_scan_results[
                        label_scan_results["Flag"].fillna("") != ""
                    ].copy()
                    if display_results.empty:
                        st.info("No candidate labels were flagged.")
                    else:
                        for metric_col in ["Accuracy", "F1 macro", "Recall", "Specificity"]:
                            display_results[metric_col] = display_results[metric_col].map(
                                lambda value: "" if pd.isna(value) else f"{value:.3f}"
                            )
                        st.dataframe(
                            display_results,
                            use_container_width=True,
                            hide_index=True,
                        )

            # ---------------- Label selection ----------------
            label = st.selectbox(
                "Select label column (by default, SLEAP selected the last column as label):",
                options=df_afterprocessed.columns,
                index=len(df_afterprocessed.columns) - 1
            )

            # ---------------- Feature selection ----------------
    
            all_features = [col for col in df_afterprocessed.columns if col != label]

            selected_features_key = f"{SESSION_PREFIX}_selected_features"
            flagged_feature_columns = st.session_state.get(f"{SESSION_PREFIX}_flagged_feature_columns", [])
            flagged_feature_version = st.session_state.get(f"{SESSION_PREFIX}_flagged_feature_version", 0)
            flagged_feature_applied_version = st.session_state.get(f"{SESSION_PREFIX}_flagged_feature_applied_version", 0)
            apply_flagged_feature_defaults = flagged_feature_version != flagged_feature_applied_version

            if selected_features_key not in st.session_state:
                if apply_flagged_feature_defaults:
                    st.session_state[selected_features_key] = [
                        feature for feature in all_features
                        if feature not in flagged_feature_columns
                    ]
                    st.session_state[f"{SESSION_PREFIX}_flagged_feature_applied_version"] = flagged_feature_version
                else:
                    st.session_state[selected_features_key] = all_features.copy()
            else:
                st.session_state[selected_features_key] = [
                    feature for feature in st.session_state[selected_features_key]
                    if feature in all_features
                ]
                if apply_flagged_feature_defaults:
                    st.session_state[selected_features_key] = [
                        feature for feature in st.session_state[selected_features_key]
                        if feature not in flagged_feature_columns
                    ]
                    st.session_state[f"{SESSION_PREFIX}_flagged_feature_applied_version"] = flagged_feature_version

            selected = st.multiselect(
                "Selected features:",
                all_features,
                key=selected_features_key
            )

            removed_features = list(set(all_features) - set(selected))

            st.multiselect(
                "Unselected features:",
                removed_features,
                default=removed_features,
                disabled=True
            )

            # st.markdown("---")

            # ---------------- Confirm ----------------
            confirm_disabled = len(selected) == 0

        if st.button("✅ Confirm features", disabled=confirm_disabled):
            df_uploaded_dropped = df_afterprocessed.dropna(subset=selected + [label])
            if df_uploaded_dropped.empty:
                st.error("No rows remain after dropping missing values for the selected features and label.")
                st.session_state[f"{SESSION_PREFIX}_confirmed"] = False
            else:
                try:
                    y_binary, label_mapping = prepare_binary_label(df_uploaded_dropped[label])
                except ValueError as error:
                    st.error(str(error))
                    st.session_state[f"{SESSION_PREFIX}_confirmed"] = False
                else:
                    st.session_state[f"{SESSION_PREFIX}_df_x_uploaded"] = df_uploaded_dropped[selected]
                    st.session_state[f"{SESSION_PREFIX}_df_y_uploaded"] = y_binary
                    st.session_state[f"{SESSION_PREFIX}_label_mapping"] = label_mapping
                    st.session_state[f"{SESSION_PREFIX}_confirmed"] = True

        # Show confirmation
        if st.session_state.get(f"{SESSION_PREFIX}_confirmed", False):
            st.success(f"{len(selected)} features confirmed")
            st.caption(f"Label mapping: {st.session_state[f'{SESSION_PREFIX}_label_mapping']}")

        # ---------------- Train ----------------
        train_disabled = not st.session_state.get(f"{SESSION_PREFIX}_confirmed", False)

        if st.button("🚀 Train model", disabled=train_disabled):

            with st.spinner("Training model..."):
                try:
                    model, X_test, y_test, adjusted_threshold = train_model(
                        st.session_state[f"{SESSION_PREFIX}_df_x_uploaded"],
                        st.session_state[f"{SESSION_PREFIX}_df_y_uploaded"],
                        seed=random_seeds[0]
                    )
                except ValueError as error:
                    st.error(str(error))
                    st.stop()

                st.success("Model trained successfully!")
                # st.write(f"Adjusted threshold: {np.round(adjusted_threshold, 2)}")
                y_test_proba = model.predict_proba(X_test)[:, 1]
                y_test_pred = np.where(y_test_proba >= adjusted_threshold, 1, 0)
                cm, sensitivity, specificity, npv, f1, mcc = evaluation_metric(y_test, y_test_pred)
                # Compute ROC
                fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
                roc_auc = auc(fpr, tpr)

                st.markdown("#### Performance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Sensitivity", f"{sensitivity*100:.2f}%")
                col2.metric("Specificity", f"{specificity*100:.2f}%")
                col3.metric("NPV", f"{npv*100:.2f}%")

                col4, col5, col6 = st.columns(3)
                col4.metric("F1 Score", f"{f1:.2f}")
                col5.metric("MCC", f"{mcc:.2f}")
                col6.metric("AUC", f"{roc_auc:.2f}")
                
                cols = st.columns(2)
                with cols[0]:
                    
                    st.markdown("#### Confusion Matrix")
                    fig, ax = plt.subplots()
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(ax=ax)
                    st.pyplot(fig)

                    # cm_df = pd.DataFrame(cm, 
                    #     index=["Actual 0", "Actual 1"], 
                    #     columns=["Pred 0", "Pred 1"])
                    # st.dataframe(cm_df)
                
                with cols[1]:
                    st.markdown("#### ROC Curve")

                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                    ax.plot([0, 1], [0, 1], linestyle="--")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend()

                    st.pyplot(fig)

                # # Serialize
                # model_bytes = pickle.dumps(model)

                # # Download
                # st.download_button(
                #     label="💾 Download model",
                #     data=model_bytes,
                #     file_name="sleep_apnea_model.pkl",
                #     mime="application/octet-stream"
                # )

                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".cbm") as tmp:
                    model.save_model(tmp.name)

                    with open(tmp.name, "rb") as f:
                        model_bytes = f.read()
                os.unlink(tmp.name)

                # Download button
                st.download_button(
                    label="💾 Download CatBoost model (.cbm)",
                    data=model_bytes,
                    file_name="sleep_apnea_model.cbm",
                    mime="application/octet-stream"
                )
                
