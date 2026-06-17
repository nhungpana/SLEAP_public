import numpy as np

from sklearn import metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
                                )

def collect_metrics(**kwargs):
    return kwargs

def model_eval(y_test, y_test_pred, y_test_pred_proba, print_metrics=False, print_cm = False):
    

    if len(np.unique(y_test)) < 3:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        if print_cm:
            print("Confusion Matrix:")
            print(cm)

        ## --------------------- for binary class ------------------ ###
        tn, fp, fn, tp = [int(i) for i in cm.ravel()]

        # Evaluation metrics
        accuracy    = accuracy_score(y_test, y_test_pred)*100
        precision   = precision_score(y_test, y_test_pred)*100        #tp / (tp + fp)  # PPV
        recall      = recall_score(y_test, y_test_pred)*100           #tp / (tp + fn)  # Sensitivity
        specificity = (tn / (tn + fp))*100
        npv         = (tn / (tn + fn))*100
        f1          = f1_score(y_test, y_test_pred)
        mcc         = matthews_corrcoef(y_test, y_test_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred_proba)
        auc_model   = metrics.auc(fpr, tpr)

        # Print metrics
        if print_metrics:
            print(f"Accuracy                    : {accuracy:.4f}")
            print(f"Precision (PPV)             : {precision:.4f}")
            print(f"Sensitivity (Recall, TPR)   : {recall:.4f}")
            print(f"Specificity (TNR)           : {specificity:.4f}")
            print(f"Negative Predictive Value (NPV): {npv:.4f}")
            print(f"F1 Score                    : {f1:.4f}")
            print(f"AUC                         : {auc_model:.4f}")
            print(f"MCC                         : {mcc:.4f}")
            # print(f"fpr                         : {fpr:.4f}")  
            # print(f"tpr                         : {tpr:.4f}") 
            # print(f"thresholds                  : {thresholds:.4f}")
        
        all_metrics = collect_metrics(true_positive=tp, false_negative=fn, 
                                        false_positive=fp, true_negative=tn, 
                                        accuracy=accuracy, precision=precision,
                                        recall=recall, specificity=specificity,
                                        negative_predictive_value = npv,
                                        f1_score=f1, 
                                        auc_model=auc_model, 
                                        mcc=mcc,
                                        # fpr=fpr,
                                        # tpr=tpr,
                                        # thresholds=thresholds
                                        )

    else:
        # ------------------- Confusion Matrix ------------------- #
        cm = confusion_matrix(y_test, y_test_pred)
        if print_cm:
            print("Confusion Matrix:")
            print(cm)

        # ------------------- Metrics ------------------- #
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='macro')  # or 'weighted'
        recall = recall_score(y_test, y_test_pred, average='macro')
        f1 = f1_score(y_test, y_test_pred, average='macro')
        mcc = matthews_corrcoef(y_test, y_test_pred)

        # ------------------- Optional: Class-wise Metrics ------------------- #
        for i, label in enumerate(np.unique(y_test)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            print(f"\nClass {label} Metrics:")
            print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print(f"  Specificity={specificity:.4f}, NPV={npv:.4f}")
    

        # Print metrics
        if print_metrics:
            print(f"Accuracy                    : {accuracy:.4f}")
            print(f"Precision (PPV)             : {precision:.4f}")
            print(f"Sensitivity (Recall, TPR)   : {recall:.4f}")
            print(f"Specificity (TNR)           : {specificity:.4f}")
            print(f"Negative Predictive Value (NPV): {npv:.4f}")
            print(f"F1 Score                    : {f1:.4f}")
            print(f"MCC                         : {mcc:.4f}")
            if auc is not None:
                print(f"AUC                         : {auc_model:.4f}")
            else:
                auc = np.nan
        
        all_metrics = collect_metrics(true_positive=tp,false_negative=fn, 
                                      true_negative=tn, false_positive=fp, 
                                      accuracy=accuracy, 
                                        sensitivity=recall, specificity=specificity,
                                        positive_predictive_value=precision,
                                        negative_predictive_value=npv,
                                        f1_score=f1, 
                                        auc_model=auc_model, 
                                        mcc=mcc)
    return all_metrics