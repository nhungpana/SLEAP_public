import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mypackage.paths import DATA_DIR, IMAGES_DIR
from mypackage.preprocessing import cat_identify
from sklearn.model_selection import train_test_split

import catboost
from catboost import CatBoostClassifier
import random
random.seed(42)
random_seeds = [random.randint(0, 1000) for _ in range(50)] # Random seeds for reproducibility
# print("Random seeds:", random_seeds)

def train_model(df_X_origin, df_y, seed=42):
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
    
    model = model = CatBoostClassifier(random_seed=seed,
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

class LoadModelPage:
    def __init__(self, df_x, df_y):
        # self.dataset_name = dataset_name
        self.df_x = df_x
        self.df_y = df_y

    def display_model_page(self): 
        ### ------------------------- Display model information ------------------------------ ###
        st.markdown(f"<h4>Model information page.</h4>", unsafe_allow_html=True)
        
        tab_feature, tab_processing, tab_feature_importance, tab_threshold = st.tabs(
            ["Feature description", "Processing pipeline", "Feature importance",
             "Threshold tuning"]
            )

        with tab_feature:
            st.write("Questionnaire features:")
            path_questionnaire_features = os.path.join(IMAGES_DIR, "Skip_logic2.png")
            st.image(path_questionnaire_features, caption="Example of questionnaire features", use_column_width=True)


        with tab_processing:
            st.write("""
                    For questionnaire features, missing values are handled by mean imputation. " \
                    "Categorical variables are one-hot encoded. " \
                    "Numerical features are standardized using z-score normalization.
                     """)
            path_questionnaire = os.path.join(IMAGES_DIR, "Flow_chart_questionnaire.png")
            st.image(path_questionnaire, caption="Questionnaire feature processing pipeline", use_column_width=True)

            st.write("""
                     The original SpO$_2$ signals in the SHHS dataset are sampled at 1 Hz. 
                     The multi-scale feature engineering process, inspired by existing studies,
                     begins with coarse-graining the signals at various timescales, 
                     ranging from 2 to 10 s with an increment of 1 s, from 15 to 60 s with an increment of 5 s, 
                     and from 120 to 600 s with an increment of 60 s. 
                     This allows us to examine the predictive power of features at a wide range of timescales. 
                     Multiple coarse-grained timescales are constructed by averaging the data points within non-overlapping windows of 
                     increasing length.
                     """)
            path_grain = os.path.join(IMAGES_DIR, "Coarse_graining_process.png")
            st.image(path_grain, caption="Illustration for coarse-graining procedure, for timescales of 2, 5, and 10 s.", use_column_width=True)

            path_spo2 = os.path.join(IMAGES_DIR, "Flow_chart_signal.png")
            st.image(path_spo2, caption="SpO2 feature processing pipeline", use_column_width=True)

        # with tab_model_performance:
        #     st.write("Model performance content goes here.")

        with tab_feature_importance:
            path_SHAP_questionnaire = os.path.join(IMAGES_DIR, "SHAP.png")
            st.image(path_SHAP_questionnaire, caption="SHAP values for questionnaire features", use_column_width=True)
            path_spo2_importance = os.path.join(IMAGES_DIR, "SHAP_signal.png")
            st.image(path_spo2_importance, caption="SHAP values for SpO2 features", use_column_width=True)
            path_multiscale = os.path.join(IMAGES_DIR, "Scatter_plot1.png")
            st.image(path_multiscale, caption="Important trends observered from multiscale features", use_column_width=True)

        with tab_threshold:
            # model.load_model("/media/littlelab/disk1/0_Project/SleepQuestMining/Code/Streamlit_interface/Model_export/SHHS_model_15_seed654.cbm")
            biClass = "cut_off_5"
            cols = [i for i in self.df_x.columns if i[i.rfind("_"):] in ["_common","_s1"] ]
            df_features = self.df_x[cols]
            df_features.dropna(inplace=True)
            df_label = self.df_y["cut_off_5_AHI1"]
            df_total = pd.concat([df_features, df_label], axis=1)
            df_total.dropna(inplace=True)

            # st.dataframe(df_total)
            df_y = df_total["cut_off_5_AHI1"]
            df_X = df_total[cols]
            
            df_recall = pd.DataFrame()
            df_specificity = pd.DataFrame()
            df_npv = pd.DataFrame()
            df_f1 = pd.DataFrame()
            df_mcc = pd.DataFrame() 
            df_tn = pd.DataFrame()
            df_fp = pd.DataFrame()
            df_fn = pd.DataFrame()
            df_tp = pd.DataFrame()

            for seed in random_seeds[:2]:
                model, X_test, y_test, adjusted_threshold = train_model(df_X, df_y, seed)   

                # # model.save_model(os.path.join(PROJECT_ROOT, f"Code/Streamlit_interface/Model_export/{internal_name[:-1]}_model_{biClass[biClass.rfind('_')+1:]}_seed{seed}.cbm"))     
                
                y_test_pred = model.predict(X_test)
                y_test_pred_proba = model.predict_proba(X_test)[:, 1]
                # list_threshold.append(adjusted_threshold)
                y_test_pred = [0 if i < adjusted_threshold else 1 for i in y_test_pred_proba]
                
                
                recall_list, specificity_list, npv_list, f1_list, mcc_list, tn_list, fp_list, fn_list, tp_list = threshold_check(y_test, y_test_pred_proba)
                df_recall = pd.concat([df_recall, pd.DataFrame(recall_list)], axis=1)
                df_specificity = pd.concat([df_specificity, pd.DataFrame(specificity_list)], axis=1)
                df_npv = pd.concat([df_npv, pd.DataFrame(npv_list)], axis=1)
                df_f1 = pd.concat([df_f1, pd.DataFrame(f1_list)], axis=1)
                df_mcc = pd.concat([df_mcc, pd.DataFrame(mcc_list)], axis=1)
                df_tn = pd.concat([df_tn, pd.DataFrame(tn_list)], axis=1)
                df_fp = pd.concat([df_fp, pd.DataFrame(fp_list)], axis=1)
                df_fn = pd.concat([df_fn, pd.DataFrame(fn_list)], axis=1)
                df_tp = pd.concat([df_tp, pd.DataFrame(tp_list)], axis=1)

            print("done")
            plt.rcParams.update({'font.size': 20})
            fig, ax = plt.subplots(figsize=(20, 10), facecolor='white')

            # Boxplot settings
            boxprops1 = dict(facecolor='salmon', alpha=0.7)
            boxprops2 = dict(facecolor='skyblue', alpha=0.7)
            boxprops3 = dict(facecolor='palegreen', alpha=0.7)
            boxprops4 = dict(facecolor='orange', alpha=0.7)
            boxprops5 = dict(facecolor='purple', alpha=0.7)

            medianprops = dict(color='black', linewidth=1.5)

            # Plot: transpose so each row is a box
            b1 = ax.boxplot(df_recall.values.T, patch_artist=True, boxprops=boxprops1,
                    medianprops=medianprops, showfliers=False)
            b2 = ax.boxplot(df_specificity.values.T, patch_artist=True, boxprops=boxprops2,
                    medianprops=medianprops, showfliers=False)
            b3 = ax.boxplot(df_npv.values.T, patch_artist=True, boxprops=boxprops3,
                    medianprops=medianprops, showfliers=False)
            b4 = ax.boxplot(df_f1.values.T, patch_artist=True, boxprops=boxprops4,
                    medianprops=medianprops, showfliers=False)
            b5 = ax.boxplot(df_mcc.values.T, patch_artist=True, boxprops=boxprops5,
                    medianprops=medianprops, showfliers=False)
            y1, y2 = ax.get_ylim()
            ax.plot([26, 26], [y1, y2], 
                color='black', linestyle='--', label='Adjusted Threshold')
            ax.plot([int(adjusted_threshold*50)+1, int(adjusted_threshold*50)+1], [y1, y2], 
                    color='red', linestyle='--', label='Adjusted Threshold')


            # Customize axis labels
            ax.set_xticks(range(1, len(df_recall)+1))
            ax.set_xticklabels(np.round(np.arange(0,1.01,0.02),2), rotation=45, fontsize=20)
            ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize=20)
            ax.set_xlabel("Decision Thresholds", fontsize=20)
            ax.set_ylabel("Values", fontsize=20)
            ax.set_title(f"Evaluation Metrics vs. Decision Thresholds for {biClass}", fontsize=20)
            ax.grid(True)
            # Add legend manually
            ax.legend([b1["boxes"][0], b2["boxes"][0], b3["boxes"][0], b4["boxes"][0], b5["boxes"][0]],
                    ["Recall", "Specificity", "NPV", "F1", "MCC"],
                    loc='best', fontsize=20)

            plt.tight_layout()
            st.pyplot(fig)

            
            