
!pip install pyspark

!git clone https://github.com/paavansaireddy/MultiDiseasePrediction

import numpy as npy
import pandas as pnd
import seaborn as sbn
import matplotlib.pyplot as mtplt

from pyspark import RDD
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

spark = SparkSession.builder \
    .appName("CSV Read and Split") \
    .getOrCreate()

df = pnd.read_csv("/content/MultiDiseasePrediction/MultDiseaseDB/Blood_samples_dataset_balanced_2(f).csv")

print(df.shape)

dfc = df.copy()
print(dfc.shape)

df_anlys = df.copy()

print(df_anlys.head())

print(df_anlys.columns)

missing_count = df_anlys.isnull().sum()

# check the type of all features
df_anlys.dtypes

print(missing_count)
# Null values are already handled

# calculate the class imbalance
print(df_anlys['Disease'].value_counts())
df_anlys['Disease'].value_counts().plot(kind='bar')

# plotting the distribution of all features
df_num_columns= df_anlys.select_dtypes(include=['float64'])
df_num_columns.hist(figsize=(12,12), bins=50, xlabelsize=10, ylabelsize=10, color = 'orange')

label_encoder = LabelEncoder()
df_anlys['Disease_Encoded'] = label_encoder.fit_transform(df_anlys['Disease'])

print(df_anlys['Disease_Encoded'].value_counts())

for reading in ['LDL Cholesterol', 'HDL Cholesterol', 'BMI', 'Insulin']:
  mtplt.figure(figsize=(10, 5))
  sbn.boxplot(x='Disease_Encoded', y=reading, data=df_anlys, palette='coolwarm')
  mtplt.show()

n_col = df_anlys.select_dtypes(include=['int64', 'float64']).columns.tolist()
corr = df_anlys[n_col].corr()
mtplt.figure(figsize=(15, 15))
sbn.heatmap(corr, annot=True, cmap='coolwarm')

# get the top 5 correlated features with the Key column except the Key column itself
top_5_corr_features = corr['Disease_Encoded'].sort_values(ascending=False)[1:6]
print("Top 5 correlated features with the Key column are: ")

feat_resampl = dfc.drop('Disease', axis=1).values
pred_resampl = dfc['Disease'].values

x_train, x_test, y_train, y_test = train_test_split(feat_resampl, pred_resampl, random_state=1, test_size=0.2)

sc = spark.sparkContext



class RDDSimpleDecisionTreeRegressor:
    def __init__(self, dpt=1):
        self.dpt = dpt
        self.tree_ = None

    def fit(self, data):
        def grow_tree(data, depth):
            if depth >= self.dpt or len(data) <= 1:
                prediction = sum([x.label for x in data]) / len(data)
                return {'prediction': prediction}

            bst_gain = -1
            best_split = None
            num_features = len(data[0].features)

            for feature in range(num_features):
                points = sorted(set([z.features[feature] for z in data]))
                for i in range(1, len(points)):
                    split_point = (points[i-1] + points[i]) / 2
                    left_split = [z for z in data if z.features[feature] < split_point]
                    right_split = [z for z in data if z.features[feature] >= split_point]

                    if not left_split or not right_split:
                        continue

                    size_left = len(left_split)
                    size_right = len(right_split)
                    sum_left = sum(x.label for x in left_split)
                    sum_right = sum(x.label for x in right_split)

                    average_left = sum_left / size_left
                    average_right = sum_right / size_right
                    average_total = sum([x.label for x in data]) / len(data)

                    variance_left = sum((x.label - average_left)**2 for x in left_split)
                    variance_right = sum((x.label - average_right)**2 for x in right_split)
                    total_variance = sum((x.label - average_total)**2 for x in data)

                    tot_gain = total_variance - (variance_left + variance_right)

                    if tot_gain > bst_gain:
                        bst_gain = tot_gain
                        best_split = (feature, split_point, left_split, right_split)

            if bst_gain > 0:
                feature, split_point, left_split, right_split = best_split
                left_tree = grow_tree(left_split, depth + 1)
                right_tree = grow_tree(right_split, depth + 1)
                return {'feature': feature, 'value': split_point, 'left': left_tree, 'right': right_tree}

            prediction = sum([x.label for x in data]) / len(data)
            return {'prediction': prediction}

        if isinstance(data, RDD):
            data = data.collect()
        self.tree_ = grow_tree(data, 0)

    def predict(self, features):
        def predict_single(tree, features):
            nd_tree = tree
            while 'left' in nd_tree or 'right' in nd_tree:
                if features[nd_tree['feature']] < nd_tree['value']:
                    nd_tree = nd_tree['left']
                else:
                    nd_tree = nd_tree['right']
            return nd_tree['prediction']

        return predict_single(self.tree_, features)

class RDDGradientBoostingClassifier:
    def __init__(self, iter, l_r, dpt):
        self.iter = iter
        self.l_r = l_r
        self.dpt = dpt
        self.boost = []

    def fit(self, rdd):
        self.cls = rdd.map(lambda x: x.label).distinct().count()
        self.boost = []

        for m in range(self.iter):
            trees = []
            for k in range(self.cls):
                rdd_prepared = self._prepare_data(rdd, k)
                tree_regressor = RDDSimpleDecisionTreeRegressor(dpt=self.dpt)
                tree_regressor.fit(rdd_prepared)
                trees.append(tree_regressor)
            self.boost.append(trees)

    def _prepare_data(self, rdd, class_index):
        return rdd.map(lambda x: LabeledPoint(
            int(x.label == class_index),  # Binary classification per class
            x.features))

    def predict(self, rdd):
        boost = self.boost
        l_r = self.l_r
        cls = self.cls

        def predict_partition(iterator):
            predictions = []
            for item in iterator:
                features = item.features
                votes = npy.zeros(cls)
                for tr in boost:
                    for idx, tree in enumerate(tr):
                        votes[idx] += l_r * tree.predict(features)
                predictions.append((item, npy.argmax(votes)))
            return predictions

        return rdd.mapPartitions(predict_partition)

def numpy_to_labeled_point(data, labels):
    return sc.parallelize([
        LabeledPoint(labels[i], Vectors.dense(data[i]))
        for i in range(len(data))
    ])

from sklearn.preprocessing import LabelEncoder

lbl_encoder = LabelEncoder()
y_train_en = lbl_encoder.fit_transform(y_train)
y_test_en = lbl_encoder.transform(y_test)



rdd_train = numpy_to_labeled_point(x_train, y_train_en)
rdd_test = numpy_to_labeled_point(x_test, y_test_en)



gbc_rdd = RDDGradientBoostingClassifier(iter=10, l_r=0.2, dpt=3)

# Fit the model
gbc_rdd.fit(rdd_train)

predictions = gbc_rdd.predict(rdd_test)
pr_lb = []
for pred in predictions.collect():
  pr_lb.append(pred[1])
tr_lb = rdd_test.map(lambda x: x.label).collect()
print(pr_lb)

predictions = gbc_rdd.predict(rdd_train)
pred_lbs_train = []
for pred in predictions.collect():
  pred_lbs_train.append(pred[1])
tr_lb_train= rdd_train.map(lambda x: x.label).collect()
print(pred_lbs_train)

from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

training_acc=accuracy_score(tr_lb_train, pred_lbs_train)
print("Training- Accuracy:", training_acc)

testing_acc= accuracy_score(tr_lb, pr_lb)
print("Testing- Accuracy:", testing_acc)

precision_gbm = precision_score(tr_lb, pr_lb, average='weighted')
print("Precision:", precision_gbm)
# Calculating recall
recall_gbm = recall_score(tr_lb, pr_lb, average='weighted')
print("Recall:", recall_gbm)
# Calculating F1 score
f1_gbm = f1_score(tr_lb, pr_lb, average='weighted')
print("F1-score:", f1_gbm)

from sklearn.metrics import classification_report

report = classification_report(tr_lb, pr_lb, output_dict=True)

report['accuracy'] = {'precision': None, 'recall': None, 'f1-score': None, 'accuracy': training_acc}

df_report = pnd.DataFrame(report).transpose()

print(df_report)

report['accuracy'] = {'precision': None, 'recall': None, 'f1-score': None, 'accuracy': testing_acc}

df_report = pnd.DataFrame(report).transpose()

print(df_report)

mtplt.figure(figsize=(8, 6))
mtplt.scatter(range(len(tr_lb)), tr_lb, color='Green', label='Actual')
mtplt.scatter(range(len(pr_lb)), pr_lb, color='Yellow', marker='x', label='Predicted')

mtplt.xlabel('Sample Index')
mtplt.ylabel('Label')
mtplt.title('Actual_labls vs Predicted_labls')
mtplt.legend()
mtplt.show()

mtplt.figure(figsize=(8, 6))
mtplt.scatter(range(len(tr_lb_train)), tr_lb_train, color='Green', label='Actual')
mtplt.scatter(range(len(pred_lbs_train)), pred_lbs_train, color='Yellow', marker='x', label='Predicted')

mtplt.xlabel('Sample Index')
mtplt.ylabel('Label')
mtplt.title('Actual_labls vs Predicted_Labls')
mtplt.legend()
mtplt.show()

y_train_en = label_encoder.fit_transform(y_train)
y_test_en = label_encoder.transform(y_test)

spark = SparkSession.builder.appName("CustomGBM").getOrCreate()

# Convert to RDDs,
x_train_rdd = spark.sparkContext.parallelize(x_train.tolist())
y_train_rdd = spark.sparkContext.parallelize(y_train_en.tolist())
x_test_rdd = spark.sparkContext.parallelize(x_test.tolist())
y_test_rdd = spark.sparkContext.parallelize(y_test_en.tolist())

# Display the shapes of the data
print(x_train.shape, y_train, x_test.shape, y_test)

from pyspark.ml.linalg import Vectors

# Combine features and labels for training data
train_rdd_svm = x_train_rdd.zip(y_train_rdd).map(lambda Z: (Vectors.dense(Z[0]), Z[1]))

# Combine features and labels for testing data
test_rdd_svm = x_test_rdd.zip(y_test_rdd).map(lambda Z: (Vectors.dense(Z[0]), Z[1]))

import numpy as np
from collections import defaultdict

class MultiClassSVM:
    def __init__(self, max_iter=100, lr=0.01, reg_param=0.01):
        self.max_iter = max_iter
        self.lr = lr
        self.reg_param = reg_param
        self.classifiers = {}

    def fit(self, rdd):
        # Get all distinct labels
        temp = rdd.map(lambda x: x[1])
        labels= temp.sample(False,0.01).distinct().collect()
        print(len(labels))

        for label in labels:
            print(label)
            binary_rdd = rdd.map(lambda Z: (Z[0], 1 if Z[1] == label else -1))
            classifier = LinearSVM(self.max_iter, self.lr, self.reg_param)
            classifier.fit(binary_rdd)
            self.classifiers[label] = classifier

    def predict(self, features):
        scores = {label: classifier.predict(features) for label, classifier in self.classifiers.items()}
        return max(scores, key=scores.get)


train_rdd_svm = x_train_rdd.zip(y_train_rdd).map(lambda Z: (np.array(Z[0]), Z[1]))


test_rdd_svm = x_test_rdd.zip(y_test_rdd).map(lambda Z: (np.array(Z[0]), Z[1]))

# Define the Linear SVM class for binary classification
class LinearSVM:
    def __init__(self, max_iter=100, lr=0.01, reg_param=0.01):
        self.max_iter = max_iter
        self.lr = lr
        self.reg_param = reg_param
        self.weights = None
        self.bias = None

    def fit(self, rdd):
        num_features = len(rdd.first()[0])
        self.weights = np.zeros(num_features)
        self.bias = 0

        for AB in range(self.max_iter):
            print("iteration-",AB)
            for features, label in rdd.collect():
                features = np.array(features)
                label = 1 if label == 1 else -1

                prediction = np.dot(features, self.weights) + self.bias
                condition = label * prediction >= 1

                if condition:
                    grad_w = self.reg_param * self.weights
                    grad_b = 0
                else:
                    grad_w = self.reg_param * self.weights - label * features
                    grad_b = -label

                self.weights -= self.lr * grad_w
                self.bias -= self.lr * grad_b

    def predict(self, features):
        return np.dot(features, self.weights) + self.bias

# Instantiate and fit the multi-class SVM model
multi_class_svm = MultiClassSVM(max_iter=100, lr=0.01, reg_param=0.01)
multi_class_svm.fit(train_rdd_svm)

# Evaluate the Model
# Evaluate training accuracy
train_preds = train_rdd_svm.map(lambda Z: (Z[1], multi_class_svm.predict(Z[0])))
train_accuracy = train_preds.filter(lambda Z: Z[0] == Z[1]).count() / float(train_preds.count())
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Evaluate testing accuracy
test_preds = test_rdd_svm.map(lambda Z: (Z[1], multi_class_svm.predict(Z[0])))
test_accuracy = test_preds.filter(lambda Z: Z[0] == Z[1]).count() / float(test_preds.count())
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# Helper function to compute overall metrics
def compute_overall_metrics(predictions):
    tp_total, fp_total, fn_total = 0, 0, 0
    metrics = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    for actual, predicted in predictions:
        if actual == predicted:
            metrics[actual]["TP"] += 1
        else:
            metrics[actual]["FN"] += 1
            metrics[predicted]["FP"] += 1

    # Summing up all the TP, FP, FN
    for stat in metrics.values():
        tp_total += stat["TP"]
        fp_total += stat["FP"]
        fn_total += stat["FN"]

    precision_svm = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall_svm = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1_svm = 2 * precision_svm * recall_svm / (precision_svm + recall_svm) if (precision_svm + recall_svm) > 0 else 0

    print(f"Precision: {precision_svm:.2f}")
    print(f"Recall: {recall_svm:.6f}")
    print(f"F1-Score: {f1_svm:.6f}")

# Evaluate training predictions
train_predictions = train_rdd_svm.map(lambda x: (x[1], multi_class_svm.predict(x[0]))).collect()
print("Training Metrics:")
compute_overall_metrics(train_predictions)

# Evaluate testing predictions
test_predictions = test_rdd_svm.map(lambda x: (x[1], multi_class_svm.predict(x[0]))).collect()
print("\nTesting Metrics:")
compute_overall_metrics(test_predictions)
