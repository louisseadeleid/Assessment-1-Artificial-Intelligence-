#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 00:39:00 2025

@author: winsicheng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 00:19:22 2025

@author: winsicheng
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

df=pd.read_csv('/Users/winsicheng/Downloads/earthquake_data_tsunami.csv')

def tsunamicheck(row):
    score = 0
    if row["magnitude"] >= 7.5:
        score += 3
    elif row["magnitude"] >= 7.0:
        score += 2
    elif row["magnitude"] >= 6.5:
        score += 1

    if row["depth"] <= 50:
        score += 2
    elif row["depth"] <= 100:
        score += 1

    if row["sig"] >= 800:
        score += 3
    elif row["sig"] >= 600:
        score += 2
    elif row["sig"] >= 500:
        score += 1

    if score >= 7:
        return "Severe Risk"
    elif score >= 5:
        return "High Risk"
    elif score >= 3:
        return "Moderate Risk"
    else:
        return "Low Risk"


    
df["RiskLevel"] = df.apply(tsunamicheck, axis=1)

risk_counts = df["RiskLevel"].value_counts()
risk_counts.plot.bar(x="Risk Level", y="Number of Earthquakes")
plt.title("Tsunami Risk Levels by Earthquake Magnitude")


def classrisk(risk):
    if risk in ["High Risk"]:
        return 1
    else:
        return 0

df["Predicted_Tsunami"] = df["RiskLevel"].apply(classrisk)

y_true = df["tsunami"]

y_pred = df["Predicted_Tsunami"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")


metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(7, 5))
bars = plt.bar(metrics, values)

# Guided By AI to show a graph of Metric Scoresfor bar in bars:
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{bar.get_height():.2f}", ha='center', fontsize=10, fontweight='bold')


plt.ylim(0, 1)  
plt.title("Model Evaluation Metrics", fontsize=14, fontweight='bold')
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)