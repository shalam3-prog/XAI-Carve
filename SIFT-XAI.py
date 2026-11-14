# --------------------------------------------------------------------------------------------------------------
#
# Filename: SIFT-XAI.py
# Author: Shahid Alam (shalam3@gmail.com)
# Dated: May, 14, 2026
# Explain the results of SIFT using different XAI techniques
#
# --------------------------------------------------------------------------------------------------------------

from __future__ import print_function
import os, sys, argparse
import pandas as pd

from Classify import *

__DEBUG__  = True
CLASS_LABEL = "Class"
# File Classes
file_types = ['csv', 'dbase3', 'doc', 'eps', 'gif', 'gz', 'html', 'jpg', 'kmz', 'log', 'pdf', 'png', 'ppt', 'ps', 'swf', 'text', 'txt', 'unk', 'xls', 'xml']

if __name__=="__main__":
	print("--- SIFT-XAI START ---")

	parser = argparse.ArgumentParser()

	parser.add_argument("-f", type=str)
	args = parser.parse_args()

	if args.f == None:
		print("ERROR:SIFT-XAI: Missing arguments\nUsage:\n")
		print("   python SIFT-XAI.py -f <csv_filename>")
		sys.exit(1)
	csv_filename = args.f.strip()
	if not os.path.isfile(csv_filename):
		print("File '%s' does not exist"%csv_filename)
		print("   python SIFT-XAI.py -f <csv_filename>")
		sys.exit(1)

	CL = Classifier(CLASS_LABEL, file_types)
	dataset = pd.read_csv(csv_filename)
	if len(dataset) >= 30:   # more than 29 samples
		__DNN_ = True
		__DNN_W__ = True
		__XAI__ = True
		CL.classify(__DNN_, __DNN_W__, __XAI__, dataset, csv_filename)
	else:
		print("ERROR:Nothing to classify in file %s"%csv_filename)

	print("--- SIFT-XAI END ---")

'''
import numpy as np
import scipy.stats as stats

# Raw F1 data from 5 independent runs
NB = [50.41, 50.32, 50.76, 51.11, 51.09]
DT = [72.97, 72.55, 72.49, 73.23, 73.09]
SVC = [79.52, 79.23, 79.19, 80.02, 80.15]
RF = [80.92, 80.52, 80.33, 81.15, 81.27]
KNN = [81.52, 81.43, 81.89, 82.16, 82.28]
DNN_W = [51.89, 52.11, 52.01, 51.55, 51.65]
DNN = [86.14, 85.99, 86.19, 86.15, 86.25]  # Proposed Model

# Group the models for clean iteration
models = {
    "Naive_Bayes": NB,
    "Decision_Tree": DT,
    "SVC": SVC,
    "Random_Forest": RF,
    "KNN": KNN,
    "DNN-W": DNN_W,
    "DNN (Proposed)": DNN
}

print("--- SUMMARY STATISTICS ---")
for name, acc_list in models.items():
    mean = np.mean(acc_list)
    std = np.std(acc_list, ddof=1)  # ddof=1 ensures unbiased sample standard deviation
    print(f"{name}: Mean F1 = {mean:.2f}%, Std Dev = ± {std:.2f}%")

print("\n--- STATISTICAL SIGNIFICANCE (p-values vs Proposed) ---")
proposed_data = models["DNN (Proposed)"]

for name, acc_list in models.items():
    if name == "DNN (Proposed)":
        continue
        
    # 1. Paired t-test (Use if runs share identical seeds/data splits)
    _, p_paired = stats.ttest_rel(acc_list, proposed_data)
    
    print(f"{name} vs Proposed Model:")
    print(f"  Paired t-test p-value: {p_paired:.10f}")
'''