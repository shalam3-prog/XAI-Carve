# --------------------------------------------------------------------------------------------------------------
#
# Filename: Classify.py
# Author: Shahid Alam (shalam3@gmail.com)
# Dated: May, 14, 2026
#
# --------------------------------------------------------------------------------------------------------------

from __future__ import print_function
import os, sys, math, time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import joblib

import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
import time

__DEBUG__  = True
# Increase epochs slightly to 100 to allow the wider layers to fully converge
epochs = 100# 5 is for testing, actual value 100
batch_size = 128

def format_forensic_feature_names(raw_feature_names):
	"""
	Transforms raw vectorizer token names into standard uppercase hex strings.
	Example: '3c 3f 78' -> '0x3C3F78'
	"""
	formatted_list = []
	for name in raw_feature_names:
		# Split tokens assuming your string features are space-delimited hex characters
		# Also there is an X in the token sometimes, so we remove it
		name = name.replace("X", "").replace("x", "")
		tokens = str(name).split()
		if len(tokens) > 0:
			hex_string = "".join([t.upper() for t in tokens])
			formatted_list.append(f"0x{hex_string}")
		else:
			formatted_list.append(f"0x{str(name).upper()}")
	return formatted_list

def get_class_index(file_types, target_name):
	"""
	Finds the exact index of a class name, ignoring case differences.
	Returns the integer index, or None if the class is not found.
	"""
	# Standardize input to lowercase and strip whitespace
	clean_target = str(target_name).lower().strip()
	
	# Standardize the array values to lowercase for a reliable match
	clean_file_types = [str(f).lower().strip() for f in file_types]
	
	if clean_target in clean_file_types:
		return clean_file_types.index(clean_target)
	else:
		print(f"[!] Warning: Class '{target_name}' not found in the file_types array.")
		return None

#
#
#
class Classifier:
	def __init__(self, CLASS_LABEL, file_types):
		self.CLASS_LABEL = CLASS_LABEL
		self.n_classes = 0
		self.file_types = file_types

	#
	#
	#
	def classify(self, __DNN__, __DNN_W__, __XAI__, dataset, dataset_filename, testSize=20):
		y = dataset.Class							# Labels
		y = np.array(y)
		X = dataset.drop(self.CLASS_LABEL, axis=1)   # Features
		X = np.array(X)

		ts = testSize
		testSize = testSize / 100

		train_csv = "train_features.csv"
		test_csv = "test_features.csv"

		# Check if both pre-processed CSV files already exist
		if os.path.exists(train_csv) and os.path.exists(test_csv):
			print(f"[+] Found cached feature files ({train_csv} and {test_csv}). Loading directly...")
				
			train_df = pd.read_csv(train_csv)
			test_df = pd.read_csv(test_csv)
				
			# Separate features and labels
			train_y = np.array(train_df[self.CLASS_LABEL])
			train_X = np.array(train_df.drop(self.CLASS_LABEL, axis=1))
				
			test_y = np.array(test_df[self.CLASS_LABEL])
			test_X = np.array(test_df.drop(self.CLASS_LABEL, axis=1))
				
			print(f"[+] Loaded {len(train_X)} training samples and {len(test_X)} test samples.")

		else:
			print("[!] Cached feature files not found. Running split and TF-IDF feature selection...")
				
			# Perform train/test split FIRST
			# It's not an n-fold cross validation but a split
			train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=testSize, random_state=0, stratify=y)

			# Extract features and save to train_csv and test_csv
			train_X, test_X, train_y, test_y = self.selectFeatures(
				TFIDF=True,
				train_X=train_X,
				test_X=test_X,
				train_y=train_y,
				test_y=test_y,
				train_csv=train_csv,
				test_csv=test_csv,
				CLASS_LABEL=self.CLASS_LABEL
			)

		__BASELINE__ = True
		if __BASELINE__ == True: # With feature selection
			names = [
					"Naive Bayes",
					"Decision Tree",
					"Support Vector Machine",
					"Random Forest",
					"K-Nearest Neighbors"
					]
			classifiers = [
							GaussianNB(),
							DecisionTreeClassifier(max_depth=100),
							LinearSVC(max_iter=10000),
							RandomForestClassifier(n_estimators=500),
							KNeighborsClassifier(n_neighbors=3)
			]
			n = 0
			while n < len(names):
				clf = classifiers[n]
				name = names[n]
				cn = "Classifying with " + name + " . . . . .\n"
				print("----------------------------------------------------\n")
				print(cn)
				print("----------------------------------------------------")
				filename_result = dataset_filename + "_" + str(ts) + "_" + str(name) + ".results.txt"
				print("Classifying and saving results in " + filename_result)
				file_result = open(filename_result, "w")
				filename_roc = dataset_filename + "_" + str(ts) + "_" + name + ".png"
				self.classify_with_split(cn, train_X, test_X, train_y, test_y, clf, file_result)
				n += 1

		if __DNN__ == True: # With feature selection
			cn = "Classifying with deep learning DNN . . . . .\n"
			print("----------------------------------------------------\n")
			print(cn)
			print("----------------------------------------------------")
			filename_result = dataset_filename + "_" + str(batch_size) + "_DNN_" + str(ts) + ".results.txt"
			print("Classifying and saving results in " + filename_result)
			file_result = open(filename_result, "w")
			dataset_filename += dataset_filename + "_" + str(batch_size)
			filename_roc = dataset_filename + "_DL_" + str(ts) + ".png"
			self.classify_with_split_DL(cn, train_X, test_X, train_y, test_y, file_result, filename_roc)

		if __DNN_W__ == True: # Without feature selection
			# Perform train/test split FIRST
			# It's not an n-fold cross validation but a split
			train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=testSize, random_state=0, stratify=y)

			cn = "Classifying with deep learning DNN-W . . . . .\n"
			print("----------------------------------------------------\n")
			print(cn)
			print("----------------------------------------------------")
			filename_result = dataset_filename + "_" + str(batch_size) + "_DNN-W_" + str(ts) + ".results.txt"
			print("Classifying and saving results in " + filename_result)
			file_result = open(filename_result, "w")
			dataset_filename += dataset_filename + "_" + str(batch_size)
			filename_roc = dataset_filename + "_DL_" + str(ts) + ".png"
			self.classify_with_split_DL(cn, train_X, test_X, train_y, test_y, file_result, filename_roc)

			if __XAI__:
				for name in self.file_types:
					print(f"\n[+] Generating XAI forensic explanation for class: {name}")
					self.explain_sample_xai(train_X, test_X, test_y, class_name=name)

		file_result.close()

	#
	#
	#
	def selectFeatures(self, TFIDF, train_X, test_X, train_y, test_y, train_csv, test_csv, CLASS_LABEL):
		print("Processing and saving labelled features after train/test split...")
		
		train_X = np.array(train_X)
		test_X = np.array(test_X)
		train_y = np.array(train_y)
		test_y = np.array(test_y)

		if TFIDF:
			print("Applying TF-IDF strictly on training set to prevent data leakage")
			print("Number of Features BEFORE: " + str(len(train_X[0])))

			# 1. Convert both splits into string corpora
			train_corpus = [np.array_str(r) for r in train_X]
			test_corpus = [np.array_str(r) for r in test_X]

			# 2. Fit CountVectorizer ONLY on train_corpus, then transform both
			cv = CountVectorizer()
			train_X_counts = cv.fit_transform(train_corpus).toarray()
			test_X_counts = cv.transform(test_corpus).toarray()

			# 3. Extract and save feature names
			features_names = cv.get_feature_names_out()
			if os.path.exists('tfidf_features.joblib'):
				os.remove('tfidf_features.joblib')
			joblib.dump(features_names, 'tfidf_features.joblib')
			print("[+] Overwrote joblib with clean numeric strings list:", features_names[:5])

			# 4. Fit TfidfTransformer ONLY on train_X_counts, then transform both
			tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
			train_X = tfidf.fit_transform(train_X_counts, train_y).toarray()
			test_X = tfidf.transform(test_X_counts).toarray()

			# 5. Normalize (0 - 100) and quantize to integers
			train_X = np.round(train_X * 100).astype(int)
			test_X = np.round(test_X * 100).astype(int)

			# Helper to attach headers and class labels, then save to CSV
			def _save_split_csv(X_mat, y_vec, filename):
				header = np.array([features_names])
				matrix = np.vstack((header, X_mat))
				y_col = np.insert(y_vec.astype(str), 0, CLASS_LABEL, axis=0)
				matrix = np.insert(matrix, matrix.shape[1], y_col, axis=1)
				np.savetxt(filename, matrix, delimiter=',', fmt='%s')

			_save_split_csv(train_X, train_y, train_csv)
			_save_split_csv(test_X, test_y, test_csv)

			print("Number of Features AFTER: " + str(len(features_names)))

		print(f"Finished extracting and saving features to {train_csv} and {test_csv}")
		return train_X, test_X, train_y, test_y

	#
	#
	#
	def classify_with_split_DL(self, cn, train_X, test_X, train_y, test_y, file_result, filename_roc):
		class_labels = list(set(test_y)) # Safely list unique class metrics

		import joblib
		import os
		
		feature_file = 'tfidf_features.joblib'
		
		if os.path.exists(feature_file):
			print(f"[*] Loading precomputed feature names from {feature_file}...")
			byte_feature_list = joblib.load(feature_file)
		else:
			print(f"[!] Warning: {feature_file} not found. Generating fallback tracking markers.")
			# Fallback placeholder naming if the file is missing
			byte_feature_list = [f"Feature_{i}" for i in range(train_X.shape[-1])]
			
		# Verify the dimensions match your network's inputs perfectly
		print(f"[+] Loaded feature list length: {len(byte_feature_list)} (Matches train_X columns: {train_X.shape[-1]})")

		start = time.time()

		# === CRITICAL CORRECTION: REMOVE THE "/ 255" SCALE OPERATOR ===
		# TF-IDF vectors are already naturally bounded between 0.0 and 1.0. 
		# Compressing them by 255 starves your dense network weights.
		train_X = train_X.reshape(len(train_X), train_X.shape[-1]).astype("float32")
		test_X = test_X.reshape(len(test_X), train_X.shape[-1]).astype("float32")
		train_y = train_y.astype("float32")
		test_y = test_y.astype("float32")

		n_features = train_X.shape[-1]
		n_classes = len(np.unique(train_y))
		print(f"Features: {n_features} | Target Classes: {n_classes}")

		# === OPTIMIZED INTELLECTUAL MULTI-LAYER FUNNEL ===
		# Replaces 19 thin layers with 4 deeply scaled, stabilized blocks.
		# This prevents vanishing gradients and drastically drops execution times.

		# === HIGH-DIMENSIONAL SWISH FUNNEL ARCHITECTURE ===
		# Designed specifically to resolve overlapping high-entropy TF-IDF matrices
		__DO__ = True # For ablation study
		__SWISH__ = True # For ablation study
		__BN__ = True # For ablation study
		__ADAMW__ = True  # For ablation study
		if __DO__ == False:
			DO_1 = 0.0
			DO_2 = 0.0
			DO_3 = 0.0
		else:
			DO_1 = 0.4
			DO_2 = 0.3
			DO_3 = 0.2
		if __BN__ == False: # For ablation study
			model = Sequential([
				# Layer 1: Massive 1024-Node Feature Ingestion Space
				Dense(1024, activation='swish', kernel_initializer='he_normal', input_shape=(n_features,)),
				Dropout(0.4), # Defends against over-fitting on highly dominant words
				
				# Layer 2: Intermediate Latent Resolution Layer
				Dense(512, activation='swish', kernel_initializer='he_normal'),
				Dropout(0.3),
				
				# Layer 3: Fine-Grained Multiclass Boundary Filter
				Dense(256, activation='swish', kernel_initializer='he_normal'),
				Dropout(0.2),
				
				# Layer 4: Standardized Output Probability Mapping
				Dense(n_classes, activation='softmax')
			])
		else:
			if __SWISH__ == True:
				act = 'swish'
			else:
				act = 'linear'
			model = Sequential([
				# Layer 1: Massive 1024-Node Feature Ingestion Space
				Dense(1024, activation=act, kernel_initializer='he_normal', input_shape=(n_features,)),
				BatchNormalization(),
				Dropout(DO_1), # Defends against over-fitting on highly dominant words
					
				# Layer 2: Intermediate Latent Resolution Layer
				Dense(512, activation=act, kernel_initializer='he_normal'),
				BatchNormalization(),
				Dropout(DO_2),
					
				# Layer 3: Fine-Grained Multiclass Boundary Filter
				Dense(256, activation=act, kernel_initializer='he_normal'),
				BatchNormalization(),
				Dropout(DO_3),
					
				# Layer 4: Standardized Output Probability Mapping
				Dense(n_classes, activation='softmax')
			])
		
		print("--- Model Summary ---\n")
		model.summary()
		# Optimize learning rate for wider layers
		if __ADAMW__ == True:
			optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=1e-4)
		else:
			optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.5, weight_decay=1e-4)
		model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		
		from tensorflow.keras.callbacks import EarlyStopping
		
		early_stopping = EarlyStopping(
			monitor='val_loss',       # Watch the validation loss
			patience=3,              # Stop if val_loss doesn't improve for 3 straight epochs
			restore_best_weights=True # Roll back network weights to the best epoch at the end
		)

		print(f"Executing fit with batch_size: {batch_size} across {epochs} epochs...")
		result = f"\ntraining_size: {len(train_X)} batch_size: {batch_size}\n"
		
		history = model.fit(train_X, train_y, verbose=1, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping])
		
		print("Evaluating validation testing coordinates...")
		loss, acc = model.evaluate(test_X, test_y, verbose=1, batch_size=batch_size)
		end = time.time()
		
		print(f"The time of execution for training the DL model: {(end-start):.2f} s")
		print(f'Test Accuracy: {acc:.3f}')
		result += f"Test Accuracy: {acc}\n"
		
		# Prediction Generation
		start = time.time()
		predictions_raw = model.predict(test_X)
		predicted = tf.argmax(predictions_raw, axis=1)
		end = time.time()
		print(f"The time of execution for testing the DL model: {(end-start):.2f} s")
		
		expected = test_y
		cm = tf.math.confusion_matrix(expected, predicted).numpy()
		
		# Graph generation processing setup
		cl = [str(self.file_types[cn]) for cn in class_labels]
		predicted = predicted.numpy()

		#------------------------------------------------------------------------------
		# Confusion Matrix Plotting
		#------------------------------------------------------------------------------
		from sklearn.metrics import ConfusionMatrixDisplay
		import matplotlib.pyplot as plt

		print("[*] Generating high-resolution journal confusion matrix plot...")

		# 1. Allocate figure window
		fig, ax = plt.subplots(figsize=(14, 12), dpi=300)

		# 2. Render predictions. Disable colorbar initially so we can add a tightly-padded one manually.
		cm_fig = ConfusionMatrixDisplay.from_predictions(
			expected, 
			predicted, 
			display_labels=cl, 
			cmap='Blues',		 
			values_format='d',	
			colorbar=False,	   # Turned off default loose colorbar
			ax=ax				 
		)

		# 3. Add a custom tightly-padded Colorbar to the right of the matrix
		# pad=0.01 or 0.02 pulls the bar incredibly close to the right axis lines
		cbar = fig.colorbar(cm_fig.im_, ax=ax, fraction=0.046, pad=0.02)
		cbar.ax.tick_params(labelsize=10)

		# 4. Refine cell text properties
		for text in cm_fig.text_.flatten():
			text.set_fontsize(8)
			text.set_fontweight('normal')

		# 5. FIX: Pull Observed and Predicted titles tightly inward using small labelpad values
		cm_fig.ax_.set_xlabel("Predicted Class", fontsize=12, fontweight='bold', labelpad=4)
		cm_fig.ax_.set_ylabel("Observed Class", fontsize=12, fontweight='bold', labelpad=4)
		
		cm_fig.ax_.xaxis.set_ticks_position('top')
		cm_fig.ax_.xaxis.set_label_position('top')
		
		# 6. FIX: Pull the tick labels (file extensions) closer to the actual chart boundary lines
		# pad=2 sets a tiny distance between text and the grid boxes
		cm_fig.ax_.tick_params(axis='both', which='major', pad=2)
		
		cm_fig.ax_.set_xticklabels(cm_fig.ax_.get_xticklabels(), rotation=90, ha='center', fontsize=10)
		cm_fig.ax_.set_yticklabels(cm_fig.ax_.get_yticklabels(), fontsize=10)
		
		cm_fig.ax_.grid(False) 

		# Tight layout automatically respects our manual padding overrides
		plt.tight_layout()

		# 7. Save directly as a vector-graphic PDF
		filename = f"{file_result.name}.pdf"
		plt.savefig(filename, bbox_inches='tight', dpi=300)
		print("High-quality, compact Confusion Matrix saved in file: ", filename)
		
		result += self.classification_results(cm, class_labels, expected, predicted, cl)
		file_result.write(result)

		if __DEBUG__:
			print(result)

		self.predicted = predicted
		self.model = model
		return result

	def explain_sample_xai(self, train_X, test_X, test_y, class_name):
		import tensorflow as tf
		import numpy as np

		# 1. Resolve the integer index for your named class (e.g., 'xml' -> 19)
		class_idx = get_class_index(self.file_types, class_name)
		if class_idx is None:
			return

		# =====================================================================
		# ROUTE A: TRUE POSITIVES (Belongs to Class AND Predicted Correctly)
		# =====================================================================
		# Actually XML and Predicted as XML
		true_positive_indices = np.where((test_y == class_idx) & (self.predicted == class_idx))[0]
		print(f"[*] Available True Positive samples for {class_name}: {len(true_positive_indices)}")
		
		if len(true_positive_indices) > 0:
			first_true_sample_idx = int(true_positive_indices[0])
			print(f"[+] Generating XAI waterfall for TRUE POSITIVE index: {first_true_sample_idx}")
			self.explain_specific_sample_xai(train_X, test_X, test_y, target_class_idx=class_idx, sample_test_idx=first_true_sample_idx, state_token="tp")
		else:
			print(f"[!] Warning: No True Positives found for class {class_name}.")

		# =====================================================================
		# ROUTE B: FALSE NEGATIVES (Belongs to Class BUT Misclassified)
		# =====================================================================
		# Actually XML but Predicted as something else (The model gave a negative result)
		false_negative_indices = np.where((test_y == class_idx) & (self.predicted != class_idx))[0]
		print(f"[*] Available False Negative samples for {class_name}: {len(false_negative_indices)}")
		
		if len(false_negative_indices) > 0:
			first_false_sample_idx = int(false_negative_indices[0])
			
			# Find what wrong class the model accidentally selected instead for your paper writeup
			wrong_predicted_idx = self.predicted[first_false_sample_idx]
			wrong_class_name = self.file_types[wrong_predicted_idx]
			print(f"[+] Generating XAI waterfall for FALSE NEGATIVE index: {first_false_sample_idx}")
			print(f"	-> [Forensic Context]: This sample is ACTUALLY {class_name.upper()}, but the model misclassified it as {wrong_class_name.upper()}.")
			self.explain_specific_sample_xai(train_X, test_X, test_y, target_class_idx=class_idx, sample_test_idx=first_false_sample_idx, state_token="fn")
		else:
			# If your model gets a 100% recall score on a class (like csv), there are no errors to analyze
			print(f"[+] Perfect Recall! The model made zero negative classification errors for class {class_name}.")

		# =====================================================================
		# ROUTE C: FALSE POSITIVES (Belongs to ANOTHER Class BUT Predicted as Target)
		# =====================================================================
		# The fragment is NOT actually XML, but the model confidently guessed XML
		false_positive_indices = np.where((test_y != class_idx) & (self.predicted == class_idx))
		print(f"[*] Available False Positive samples for {class_name}: {len(false_positive_indices)}")
		
		if len(false_positive_indices[0]) > 0: # Check length of the actual array inside the tuple
			first_fp_sample_idx = int(false_positive_indices[0][0]) # Unpack the row element safely
			
			# Find what this file ACTUALLY is to present a strong case study
			actual_class_idx = test_y[first_fp_sample_idx]
			actual_class_name = self.file_types[actual_class_idx]
			
			print(f"[+] Generating XAI waterfall for FALSE POSITIVE index: {first_fp_sample_idx}")
			print(f"	-> [Forensic Context]: This sample is ACTUALLY a {actual_class_name.upper()} file, but the model was fooled into predicting it as {class_name.upper()}.")
			
			# Run the explainer on this index targeting the class_idx model output
			self.explain_specific_sample_xai(train_X, test_X, test_y, target_class_idx=class_idx, sample_test_idx=first_fp_sample_idx, state_token="fp")
		else:
			print(f"[+] Precision Shield! The model introduced zero false positive errors for class {class_name}.")

	def explain_specific_sample_xai(self, train_X, test_X, test_y, target_class_idx, sample_test_idx=0, state_token="TP"):
		"""
		Generates a local forensic explanation with explicit error-state token tagging.
		"""
		import shap
		import matplotlib.pyplot as plt
		import joblib
		import os
		import numpy as np

		class_labels = list(np.unique(test_y)) 

		# =====================================================================
		# 1. DYNAMIC LABEL EXTRACTION MAPPER FOR RE-ORDERED STRINGS
		# =====================================================================
		feature_file = 'tfidf_features.joblib'
		if os.path.exists(feature_file):
			raw_features = joblib.load(feature_file)
			print("[*] Loaded raw feature mapping array from disk:", raw_features[:3])
			
			# Formats strings to explicit hex values ("44" -> "0x2C")
			byte_feature_list = []
			for f in raw_features:
				try:
					# Clean out any accidental whitespace tags
					clean_str = str(f).strip()
					val = int(clean_str)
					byte_feature_list.append(f"0x{val:02X}")
				except ValueError:
					byte_feature_list.append(str(f).strip())
		else:
			print("[!] Warning: tfidf_features.joblib missing. Using safe positional backups.")
			byte_feature_list = [f"Feature_{i}" for i in range(train_X.shape[-1])]

		# 2. Extract dense tracking points
		np.random.seed(42)
		background_samples = train_X[np.random.choice(train_X.shape[0], 100, replace=False)]
		if hasattr(background_samples, "toarray"):
			background_samples = background_samples.toarray()
			
		target_row = test_X[sample_test_idx]
		if hasattr(target_row, "toarray"):
			target_row = target_row.toarray()
		else:
			target_row = target_row.reshape(1, -1)

		# 3. Instantiate explainer loop
		explainer = shap.KernelExplainer(self.model.predict, background_samples)
		local_shap_values = np.array(explainer.shap_values(target_row))

		# 4. Correct 3D array slicing check
		if local_shap_values.ndim == 3:
			if local_shap_values.shape[0] == len(class_labels) or local_shap_values.shape[0] > 10:
				target_class_shap = local_shap_values[target_class_idx, 0, :]
				base_value_target = explainer.expected_value[target_class_idx]
			else:
				target_class_shap = local_shap_values[0, :, target_class_idx]
				base_value_target = explainer.expected_value[target_class_idx]
		else:
			target_class_shap = local_shap_values if isinstance(local_shap_values, list) else local_shap_values[0, :]
			base_value_target = explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[target_class_idx]

		target_row_flat = target_row.flatten()

		explanation_obj = shap.Explanation(
			values=target_class_shap,
			base_values=base_value_target,
			data=target_row_flat, 
			feature_names=byte_feature_list
		)

		# 5. Render Expanded Vector Layout Graphic
		plt.figure(figsize=(10, 6)) # Expanded vertical height to support more elements cleanly
		
		# max_display=25 handles complex forensic patterns with clear visual layout parameters
		shap.plots.waterfall(explanation_obj, max_display=10, show=False)

		target_ext_name = str(self.file_types[target_class_idx]).upper()
		
		# DYNAMIC BANNERS: Updates graph presentation titles matching token state
		title_map = {
			"TP": f"Forensic Proof Vector (True Positive): Hit Analysis for Class '{target_ext_name}' (Sample #{sample_test_idx})",
			"FN": f"Forensic Proof Vector (False Negative): Omission Analysis for Class '{target_ext_name}' (Sample #{sample_test_idx})",
			"FP": f"Forensic Proof Vector (False Positive): Deception Analysis for Class '{target_ext_name}' (Sample #{sample_test_idx})"
		}
		plt.title(title_map.get(state_token.upper(), f"Analysis Matrix: {target_ext_name}"), fontsize=11, pad=15)
		plt.tight_layout()

		# DYNAMIC FILENAMES: Prefixes state token cleanly for quick directory scanning
		# Example output: "XAI_TP_Sample_181_XML.pdf"
		out_name = f"XAI_{state_token.upper()}_Sample_{sample_test_idx}_{target_ext_name.lower()}.pdf"
		plt.savefig(out_name, bbox_inches='tight', dpi=300)
		print(f"[+] Target state visual graphic saved successfully: {out_name}")
		plt.close()

	#
	# It's not an n-fold cross validation but a %age split,
	# e.g., 80% training and 20% testing
	# clf is the classifier passed
	# e.g., NaiveBayes, RandomForest etc
	#
	def classify_with_split(self, cn, train_X, test_X, train_y, test_y, clf, file_result):
		try:
			warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

			clf.fit(train_X, train_y)
			predicted = clf.predict(test_X)
			expected = test_y
			class_labels = set(expected)		# Extract the class labels from the true_y
			class_labels = list(class_labels) 	# Confusion matrix requires class labels in a list
			cl = [str(self.file_types[cn]) for cn in class_labels]
			cm = confusion_matrix(expected, predicted, labels=class_labels)
			# Confusion matrix of 2 classes
			#   |_A__|_B__
			# A |_TP_|_FN_
			# B |_FP_|_TN_
			print("--- Confusion matrix ---\n", cm)

			result = self.classification_results(cm, class_labels, expected, predicted, class_names=cl)

			file_result.write(result)
			if __DEBUG__:
				print(result)
		except ValueError:
			print("--- Data ERROR ---")
			print("Error:Classify::classify_with_split: Value Error!")
			pass

	#
	#
	#
	def classification_results(self, cm, class_labels, expected, predicted, class_names):
		result = ""

		report = classification_report(expected, predicted, target_names=class_names)
		print("--- Report ---\n", report)
		result += "--- Report ---\n" + str(report) + "\n"
		# =====================================================================
		# DERIVE WEIGHTED PRECISION, RECALL, AND F1 DIRECTLY FROM MATRIX
		# =====================================================================
		# 1. Calculate base sum vectors from matrix axes
		diagonal = np.diag(cm)
		row_sums = np.sum(cm, axis=1)	# Support per class (True counts)
		col_sums = np.sum(cm, axis=0)	# Total times class was predicted
		total_samples = np.sum(cm)	   # Total dataset size (N)

		# 2. Compute individual class metrics handling zero-denominator bounds safely
		# prevents runtime divide-by-zero warnings if a class has 0 predictions
		precision_per_class = np.where(col_sums > 0, diagonal / col_sums, 0.0)
		recall_per_class = np.where(row_sums > 0, diagonal / row_sums, 0.0)
		
		# Calculate F1-Score per class
		f1_denominator = precision_per_class + recall_per_class
		f1_per_class = np.where(f1_denominator > 0, 2 * (precision_per_class * recall_per_class) / f1_denominator, 0.0)

		# 3. Compute the overall Weighted Averages (Score multiplied by True Row Support / Total N)
		weighted_avg_precision = np.sum(precision_per_class * row_sums) / total_samples
		weighted_avg_recall = np.sum(recall_per_class * row_sums) / total_samples
		weighted_avg_f1 = np.sum(f1_per_class * row_sums) / total_samples

		print(f"[+] Matrix Derived Weighted Precision : {weighted_avg_precision:.4f}")
		print(f"[+] Matrix Derived Weighted Recall	: {weighted_avg_recall:.4f} (Identical to Accuracy)")
		print(f"[+] Matrix Derived Weighted F1-Score  : {weighted_avg_f1:.4f}")

		result += f"\n--- Derived Metrics Matrix Report ---\n"
		result += f"Weighted Precision (from CM): {weighted_avg_precision:.4f}\n"
		result += f"Weighted Recall (from CM):	{weighted_avg_recall:.4f}\n"
		result += f"Weighted F1-Score (from CM):  {weighted_avg_f1:.4f}\n"
		# =====================================================================
		# CALCULATE WEIGHTED AVERAGE ACCURACY DIRECTLY FROM CONFUSION MATRIX
		# =====================================================================
		# 1. Sum the main diagonal to get total true positive counts
		total_correct_predictions = np.trace(cm) 
		
		# 2. Sum every cell in the matrix to get the absolute total population size
		total_sample_population = np.sum(cm) 
		
		# 3. Compute the weighted accuracy ratio
		weighted_avg_accuracy = total_correct_predictions / total_sample_population
		
		print(f"[+] Total Correctly Carved Fragments : {total_correct_predictions}")
		print(f"[+] Total Dataset Support Population : {total_sample_population}")
		print(f"[+] Calculated Weighted Average Accuracy : {weighted_avg_accuracy:.4f}")
		
		result += f"Weighted Average Accuracy (from CM): {weighted_avg_accuracy:.4f}\n"
		return result

'''
		result += "\n--- Per-Class Metrics ---\n"
		tp = np.diag(cm)
		fn = cm.sum(axis=1) - np.diag(cm)
		fp = cm.sum(axis=0) - np.diag(cm)
		tn = cm.sum() - (tp + fn + fp)
		classes = len(cm[0])
		tpr = np.empty(classes, dtype=float)
		fpr = np.empty(classes, dtype=float)
		accuracy = np.empty(classes, dtype=float)
		f1_score = np.empty(classes, dtype=float)
		precision = np.empty(classes, dtype=float)
		mean_tpr = mean_fpr = mean_accuracy = mean_f1_score = mean_precision = 0.0
		for i in range(classes):
			tpr[i] = 0
			if (tp[i] + fn[i]) > 0:
				tpr[i] = tp[i] / (tp[i] + fn[i])   #tp / (tp + fn)
			mean_tpr = mean_tpr + tpr[i]
			fpr[i] = 0
			if (fp[i] + tn[i]) > 0:
				fpr[i] = fp[i] / (fp[i] + tn[i])   #fp / (fp + tn)
			mean_fpr = mean_fpr + fpr[i]
			accuracy[i] = 0
			if (tp[i] + tn[i] + fp[i] + fn[i]) > 0:
				accuracy[i] = (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i])   #(tp + tn) / (tp + tn + fp + fn)
			mean_accuracy = mean_accuracy + accuracy[i]
			f1_score[i] = 0
			if ((2 * tp[i]) + fp[i] + fn[i]) > 0:
				f1_score[i] = (2 * tp[i]) / ((2 * tp[i]) + fp[i] + fn[i])   #(2 * tp) / ((2 * tp) + fp + fn)
			mean_f1_score = mean_f1_score + f1_score[i]
			precision[i] = 0
			if (tp[i] + fp[i]) > 0:
				precision[i] = tp[i] / (tp[i] + fp[i])   #tp / tp + fp
			mean_precision = mean_precision + precision[i]
			n = class_labels[i]# Get the specific class label (int) from the confusion matrix class labels
			c = self.file_types[n]  # Get the type (string label) of the file from the stored file types
			result += "Reporting results for Class " + c + "\n"
			result += "   TPR = " + str(tpr[i]) + "\n"
			result += "   FPR = " + str(fpr[i]) + "\n"
			result += "   Accuracy = " + str(accuracy[i]) + "\n"
			result += "   F1 Score = " + str(f1_score[i]) + "\n"
			result += "   Precision = " + str(precision[i]) + "\n"
		mean_tpr = mean_tpr / classes
		mean_fpr = mean_fpr / classes
		mean_accuracy = mean_accuracy / classes
		mean_f1_score = mean_f1_score / classes
		mean_precision = mean_precision / classes
		result += "Average TPR = " + str(mean_tpr) + "\n"
		result += "Average FPR = " + str(mean_fpr) + "\n"
		result += "Average Accuracy = " + str(mean_accuracy) + "\n"
		result += "Average F1 Score = " + str(mean_f1_score) + "\n"
		result += "Average Precision = " + str(mean_precision) + "\n\n"

		return result
'''