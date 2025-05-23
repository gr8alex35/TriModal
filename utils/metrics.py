import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

#custom modules
from utils.models import *
from utils.dataset import *

# ============================
# Evaluation functions
# ============================

def compute_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # FNR = FN / (FN + TP), FPR = FP / (FP + TN)
    fnr_list = []
    fpr_list = []

    for i in range(len(cm)):
        TP = cm[i][i]
        FN = sum(cm[i]) - TP
        FP = sum([cm[j][i] for j in range(len(cm))]) - TP
        TN = cm.sum() - TP - FN - FP

        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        fnr_list.append(fnr)
        fpr_list.append(fpr)

    print("🔍 Classification Report")
    print(classification_report(y_true, y_pred, digits=4))

    print("📉 Average FNR: {:.4f}, Average FPR: {:.4f}".format(np.mean(fnr_list), np.mean(fpr_list)))


def report_fnr_fpr(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fnr_list = []
    fpr_list = []

    for i in range(len(cm)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

        fnr_list.append(fnr)
        fpr_list.append(fpr)

    df = pd.DataFrame({
        "Class": class_names,
        "FNR": np.round(fnr_list, 4),
        "FPR": np.round(fpr_list, 4)
    })

    print("\n📌 Class FNR / FPR")
    print(df.to_string(index=False))
    return df


def evaluate(model, loader, device, detailed=False, logf=None):
    model.eval()
    all_preds = []
    all_labels = []
    misclassified_folders = []

    with torch.no_grad():
        for batch in loader:
            start = batch["start"]
            end = batch["end"]
            image = batch["image"]
            label = batch["label"]
            folders = batch["folder"]  # list of strings

            inputs = {}
            if isinstance(start, dict):
                inputs["start_text"] = start["text_emb"].to(device)
                inputs["start_feat"] = start["feature"].to(device)
            if end is not None:
                inputs["end_text"] = end.to(device)
            if image is not None:
                inputs["image_emb"] = image.to(device)

            labels = label.to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Name of False predicted labels
            for label, pred, folder in zip(labels, preds, folders):
                if label.item() != pred.item():
                    misclassified_folders.append([folder, pred])


    # accuracy
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    if logf:
        logf.write("\n❌ Misclassified folders:\n")
        for folder in misclassified_folders:
            logf.write(f"label: {folder[0]}, pred: {folder[1]}\n")
        logf.flush()
    
    if detailed:
        class_names = ["Normal", "Gambling", "Others"]
        print("\n📊 Classification Report")

        # precision, recall, f1, support
        prec, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1, 2])
        total = np.sum(support)

        # macro avg
        macro_prec = np.mean(prec)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)

        # weighted avg
        weighted_prec = np.sum(prec * support) / total
        weighted_recall = np.sum(recall * support) / total
        weighted_f1 = np.sum(f1 * support) / total

        # 출력
        print(f"{'':<10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
        for i, cls in enumerate(class_names):
            print(f"{cls:<10} {prec[i]:10.4f} {recall[i]:10.4f} {f1[i]:10.4f} {support[i]:10d}")
        print(f"{'accuracy':<10} {'':>10} {'':>10} {acc:10.4f} {total:10d}")
        print(f"{'macro avg':<10} {macro_prec:10.4f} {macro_recall:10.4f} {macro_f1:10.4f} {total:10d}")
        print(f"{'weighted avg':<10} {weighted_prec:10.4f} {weighted_recall:10.4f} {weighted_f1:10.4f} {total:10d}")

        report_fnr_fpr(all_labels, all_preds, class_names)

    return acc

