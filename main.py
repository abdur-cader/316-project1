import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import math

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, year, datediff, lit, floor,
    count, avg, max, stddev
)
from pyspark.sql.types import IntegerType

# ---------------------------------------------------------------------------
# PySpark compatibility aliases
# ---------------------------------------------------------------------------
# Some function names in this file use older/non-standard identifiers.
# Keep the rest of the script unchanged by mapping them to the canonical
# PySpark SQL function names.
from pyspark.sql.functions import (  # noqa: E402
    current_date as current_date,
    to_date as to_date,
    months_between as months_between,
    monotonically_increasing_id as monotonically_increasing_id,
)

from pyspark.sql.functions import udf, sum as spark_sum
from pyspark.sql import functions as F

# ML imports
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier, RandomForestClassifier, LogisticRegression,
    DecisionTreeClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "MrheLandGrants.csv"
OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create/initialize Spark session
print("=" * 70)
print("INITIALIZING SPARK SESSION")
print("=" * 70)

spark = SparkSession.builder \
    .master("local") \
    .appName("CSCI316_Project1") \
    .config("spark.ui.port", 4050) \
    .getOrCreate()

print(f"Spark Version: {spark.version}")
print(f"Spark Context: {spark.sparkContext}")

# ============================================================================
# PART 0: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
print(f"Initial row count: {df.count()}")
df.show(30)
df.printSchema()

# ============================================================================
# DATA CLEANING
# ============================================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# Drop rows with missing values in key columns
columns_to_check = ["AGE", "APPLICANT_FAMILY_COUNT", "MARITAL_STATUS", "GENDER_DESC"]
df = df.dropna(subset=columns_to_check)
print(f"Rows after dropping nulls: {df.count()}")

# Convert timestamp columns to date type
df = df.withColumn("APPLICATION_DATE", to_date(col("APPLICATION_DATE")))
df = df.withColumn("APPROVED_DATE", to_date(col("APPROVED_DATE")))
df = df.withColumn("APPLICANT_BIRTH_DATE", to_date(col("APPLICANT_BIRTH_DATE")))

# Cast AGE to IntegerType
df = df.withColumn("AGE", col("AGE").cast(IntegerType()))

# Drop AGE column and show updated schema
df = df.drop("AGE")
df.printSchema()
df.show(50)
print(f"Final cleaned row count: {df.count()}")

# # =============================================================================
# # PART 1: FEATURE ENGINEERING
# # =============================================================================
# # Research Question: To what extent do applicant characteristics predict
# # land-grant application approval, and can applicants be clustered into
# # meaningful groups based on those attributes?


# ============================================================================
# PART 1: FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

# 1. Calculate AGE at time of application (more accurate than current age)
df = df.withColumn(
    'AGE_AT_APPLICATION',
    floor(months_between(col('APPLICATION_DATE'), col('APPLICANT_BIRTH_DATE')) / 12)
)

# 2. Extract APPLICATION_YEAR as a feature
df = df.withColumn('APPLICATION_YEAR', year(col('APPLICATION_DATE')))

# 3. Calculate PROCESSING_DAYS (difference between approval and application)
df = df.withColumn(
    'PROCESSING_DAYS',
    datediff(col('APPROVED_DATE'), col('APPLICATION_DATE'))
)

# 4. Create binary TARGET variable (1 = Approved/موافقة, 0 = Other statuses)
# Note: موافقة = Approved, منتفـع = Beneficiary (already received benefit)
df = df.withColumn(
    'APPROVAL_LABEL',
    when(col('APPLICATION_STATUS_NAME') == 'موافقة', 1).otherwise(0)
)

# 5. Create AGE GROUP categories for additional analysis
df = df.withColumn(
    'AGE_GROUP',
    when(col('AGE_AT_APPLICATION') < 25, 'Young (18-24)')
    .when(col('AGE_AT_APPLICATION') < 35, 'Young Adult (25-34)')
    .when(col('AGE_AT_APPLICATION') < 45, 'Middle Age (35-44)')
    .when(col('AGE_AT_APPLICATION') < 55, 'Mature (45-54)')
    .when(col('AGE_AT_APPLICATION') < 65, 'Senior (55-64)')
    .otherwise('Elderly (65+)')
)

# 6. Create FAMILY_SIZE_CATEGORY
df = df.withColumn(
    'FAMILY_SIZE_CATEGORY',
    when(col('APPLICANT_FAMILY_COUNT') == 0, 'No Dependents')
    .when(col('APPLICANT_FAMILY_COUNT') <= 2, 'Small (1-2)')
    .when(col('APPLICANT_FAMILY_COUNT') <= 4, 'Medium (3-4)')
    .when(col('APPLICANT_FAMILY_COUNT') <= 6, 'Large (5-6)')
    .otherwise('Very Large (7+)')
)

# Filter out invalid ages (negative or too old)
df = df.filter((col('AGE_AT_APPLICATION') >= 18) & (col('AGE_AT_APPLICATION') <= 100))

print("Dataset after feature engineering:")
df.printSchema()
df.select('AGE_AT_APPLICATION', 'APPLICATION_YEAR', 'PROCESSING_DAYS',
          'APPROVAL_LABEL', 'AGE_GROUP', 'FAMILY_SIZE_CATEGORY').show(10)
print(f"Total records after filtering: {df.count()}")

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Check target variable distribution
print("Target Variable Distribution (APPROVAL_LABEL):")
df.groupBy('APPROVAL_LABEL').count().show()

# Check categorical variable distributions
print("\nMarital Status Distribution:")
df.groupBy('MARITAL_STATUS').count().orderBy('count', ascending=False).show()

print("\nGender Distribution:")
df.groupBy('GENDER_DESC').count().orderBy('count', ascending=False).show()

print("\nAge Group Distribution:")
df.groupBy('AGE_GROUP').count().orderBy('count', ascending=False).show()

print("\nFamily Size Category Distribution:")
df.groupBy('FAMILY_SIZE_CATEGORY').count().orderBy('count', ascending=False).show()

# Numerical features summary
print("\nNumerical Features Summary:")
df.select('AGE_AT_APPLICATION', 'APPLICANT_FAMILY_COUNT', 'APPLICATION_YEAR').describe().show()

# ============================================================================
# FEATURE TRANSFORMATION PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("FEATURE TRANSFORMATION PIPELINE")
print("=" * 70)

# Index categorical columns
marital_indexer = StringIndexer(inputCol='MARITAL_STATUS', outputCol='MARITAL_STATUS_IDX', handleInvalid='keep')
gender_indexer = StringIndexer(inputCol='GENDER_DESC', outputCol='GENDER_DESC_IDX', handleInvalid='keep')

# One-Hot Encode indexed columns
marital_encoder = OneHotEncoder(inputCol='MARITAL_STATUS_IDX', outputCol='MARITAL_STATUS_VEC')
gender_encoder = OneHotEncoder(inputCol='GENDER_DESC_IDX', outputCol='GENDER_DESC_VEC')

# Assemble all features into a single vector
feature_cols = [
    'AGE_AT_APPLICATION',
    'APPLICANT_FAMILY_COUNT',
    'APPLICATION_YEAR',
    'MARITAL_STATUS_VEC',
    'GENDER_DESC_VEC'
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_unscaled', handleInvalid='skip')

# Scale features
scaler = StandardScaler(inputCol='features_unscaled', outputCol='features', withStd=True, withMean=True)

# Build the preprocessing pipeline
preprocessing_pipeline = Pipeline(stages=[
    marital_indexer,
    gender_indexer,
    marital_encoder,
    gender_encoder,
    assembler,
    scaler
])

# Fit and transform the data
pipeline_model = preprocessing_pipeline.fit(df)
df_transformed = pipeline_model.transform(df)

print("Transformed Dataset Schema:")
df_transformed.printSchema()

# Show sample of transformed data
df_transformed.select('features', 'APPROVAL_LABEL').show(5, truncate=False)
print(f"Total samples for ML: {df_transformed.count()}")

# =============================================================================
# ============================================================================
# VISUALIZATION SETUP & CORRELATION HEATMAP
# ============================================================================

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Create correlation heatmap from numeric features
print("="*70)
print("CORRELATION HEATMAP")
print("="*70)

# Get numeric columns for correlation
numeric_cols = ['AGE_AT_APPLICATION', 'APPLICANT_FAMILY_COUNT', 'APPLICATION_YEAR', 'APPROVAL_LABEL']
corr_df = df.select(numeric_cols).toPandas()

# Compute correlation matrix
corr_matrix = corr_df.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
            fmt='.3f', square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap - Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nCorrelation heatmap saved as '{OUTPUT_DIR}/correlation_heatmap.png'")

# # =============================================================================
# # PART 2: MODEL TRAINING WITH MANUAL 10-FOLD CROSS-VALIDATION
# # =============================================================================


# ============================================================================
# PART 2: MANUAL 10-FOLD CROSS-VALIDATION IMPLEMENTATION
# ============================================================================

print("\n" + "=" * 70)
print("MANUAL 10-FOLD CROSS-VALIDATION IMPLEMENTATION")
print("=" * 70)

def manual_kfold_cross_validation(df, model, k=10, label_col='APPROVAL_LABEL', features_col='features'):
    """
    Manually implement k-fold cross-validation for SparkML models.

    Parameters:
    - df: Transformed DataFrame with features
    - model: SparkML classifier (untrained)
    - k: Number of folds (default 10)
    - label_col: Name of the label column
    - features_col: Name of the features column

    Returns:
    - Dictionary with average metrics across all folds
    """
    # Add a fold assignment column using row numbers
    df_with_id = df.withColumn('row_id', monotonically_increasing_id())
    total_count = df_with_id.count()
    fold_size = total_count // k

    # Assign fold numbers (0 to k-1)
    df_with_fold = df_with_id.withColumn(
        'fold',
        (col('row_id') % k).cast('int')
    )

    # Initialize metrics storage
    auc_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Evaluators
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol='rawPrediction',
        metricName='areaUnderROC'
    )

    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol='prediction',
        metricName='accuracy'
    )

    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol='prediction',
        metricName='weightedPrecision'
    )

    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol='prediction',
        metricName='weightedRecall'
    )

    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol='prediction',
        metricName='f1'
    )

    print(f"Starting {k}-Fold Cross-Validation...")
    print("=" * 60)

    for fold_num in range(k):
        # Split data: current fold is test, rest is train
        test_df = df_with_fold.filter(col('fold') == fold_num)
        train_df = df_with_fold.filter(col('fold') != fold_num)

        # Train model
        trained_model = model.fit(train_df)

        # Make predictions
        predictions = trained_model.transform(test_df)

        # Calculate metrics
        auc = auc_evaluator.evaluate(predictions)
        accuracy = accuracy_evaluator.evaluate(predictions)
        precision = precision_evaluator.evaluate(predictions)
        recall = recall_evaluator.evaluate(predictions)
        f1 = f1_evaluator.evaluate(predictions)

        # Store metrics
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        print(f"Fold {fold_num + 1}/{k} - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    print("=" * 60)

    # Calculate averages and standard deviations
    results = {
        'AUC': {'mean': np.mean(auc_scores), 'std': np.std(auc_scores), 'scores': auc_scores},
        'Accuracy': {'mean': np.mean(accuracy_scores), 'std': np.std(accuracy_scores), 'scores': accuracy_scores},
        'Precision': {'mean': np.mean(precision_scores), 'std': np.std(precision_scores), 'scores': precision_scores},
        'Recall': {'mean': np.mean(recall_scores), 'std': np.std(recall_scores), 'scores': recall_scores},
        'F1': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores), 'scores': f1_scores}
    }

    print("\nCross-Validation Summary:")
    for metric, values in results.items():
        print(f"  {metric}: {values['mean']:.4f} (+/- {values['std']:.4f})")

    return results

print("Manual 10-Fold Cross-Validation function defined successfully!")

# =============================================================================
# ============================================================================
# MODEL 1: CUSTOM BAGGING ENSEMBLE (FROM SCRATCH) ✅
# ============================================================================
# - Base learner: DecisionTreeClassifier (allowed from library)
# - Ensemble logic (bootstrapping + aggregation/voting): implemented by YOU

print("\n" + "=" * 70)
print("MODEL 1: CUSTOM BAGGING ENSEMBLE (FROM SCRATCH)")
print("=" * 70)

# UDF to create Spark vector probability [P(0), P(1)] from p1
to_prob_vec = udf(lambda p1: Vectors.dense([float(1.0 - p1), float(p1)]), VectorUDT())


def bagging_predict(train_df, test_df,
                    label_col="APPROVAL_LABEL",
                    features_col="features",
                    num_models=25,
                    sample_fraction=1.0,
                    base_max_depth=5,
                    seed=42):
    """
    Train a bagging ensemble (bootstrap aggregating) from scratch.
    Returns: test_df with columns: prediction, probability (Vector)
    """

    # We assume test_df has 'row_id' already (your CV function adds it).
    # Ensure it's there:
    if "row_id" not in test_df.columns:
        raise ValueError("test_df must contain 'row_id' for joining predictions.")

    # Collect per-model predictions (as 0/1)
    vote_df = None

    for i in range(num_models):
        boot = train_df.sample(withReplacement=True,
                               fraction=sample_fraction,
                               seed=seed + i)

        base = DecisionTreeClassifier(
            labelCol=label_col,
            featuresCol=features_col,
            maxDepth=base_max_depth,
            seed=seed + i
        )

        m = base.fit(boot)

        preds_i = (
            m.transform(test_df)
             .select("row_id", col("prediction").cast("int").alias(f"pred_{i}"))
        )

        vote_df = preds_i if vote_df is None else vote_df.join(preds_i, on="row_id", how="inner")

    # Sum votes across all base models
    pred_cols = [col(f"pred_{i}") for i in range(num_models)]
    vote_sum_expr = pred_cols[0]
    for pc in pred_cols[1:]:
        vote_sum_expr = vote_sum_expr + pc

    vote_df = vote_df.withColumn("vote_sum", vote_sum_expr)

    # Probability estimate = fraction of models voting 1
    vote_df = vote_df.withColumn("p1", col("vote_sum") / lit(float(num_models)))

    # Final prediction = majority vote
    vote_df = vote_df.withColumn("prediction", (col("p1") >= lit(0.5)).cast("double"))

    # Create probability vector for evaluators (BinaryClassificationEvaluator can use a vector column)
    vote_df = vote_df.withColumn("probability", to_prob_vec(col("p1")))

    # Join back onto test_df to keep label column for evaluation
    out = (
        test_df.select("row_id", label_col)
               .join(vote_df.select("row_id", "prediction", "probability"), on="row_id", how="inner")
    )
    return out


def manual_kfold_cv_custom_ensemble(df, k=10, label_col="APPROVAL_LABEL", features_col="features",
                                   num_models=25, sample_fraction=1.0, base_max_depth=5, seed=42):
    """
    Manual K-fold CV where the ensemble training logic is custom (from scratch).
    """
    from pyspark.sql.functions import monotonically_increasing_id
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    import numpy as np

    df_with_id = df.withColumn("row_id", monotonically_increasing_id())
    df_with_fold = df_with_id.withColumn("fold", (col("row_id") % k).cast("int"))

    auc_evaluator = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="probability",   # we provide probability vector
        metricName="areaUnderROC"
    )

    acc_eval = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="f1"
    )

    aucs, accs, f1s = [], [], []

    print(f"Starting {k}-Fold CV for Custom Bagging...")
    print("=" * 60)

    for fold_num in range(k):
        test_df = df_with_fold.filter(col("fold") == fold_num)
        train_df = df_with_fold.filter(col("fold") != fold_num)

        preds = bagging_predict(
            train_df=train_df,
            test_df=test_df,
            label_col=label_col,
            features_col=features_col,
            num_models=num_models,
            sample_fraction=sample_fraction,
            base_max_depth=base_max_depth,
            seed=seed
        )

        auc = auc_evaluator.evaluate(preds)
        acc = acc_eval.evaluate(preds)
        f1 = f1_eval.evaluate(preds)

        aucs.append(auc); accs.append(acc); f1s.append(f1)

        print(f"Fold {fold_num+1}/{k}  AUC={auc:.4f}  ACC={acc:.4f}  F1={f1:.4f}")

    print("=" * 60)
    results = {
        "AUC": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs)), "scores": aucs},
        "Accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "scores": accs},
        "F1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s)), "scores": f1s},
    }

    print("CV Summary:")
    for m, v in results.items():
        print(f"  {m}: {v['mean']:.4f} (+/- {v['std']:.4f})")

    return results


# Run Custom Bagging CV
bagging_results = manual_kfold_cv_custom_ensemble(
    df=df_transformed,
    k=10,
    label_col="APPROVAL_LABEL",
    features_col="features",
    num_models=25,
    sample_fraction=1.0,   # bootstrap size ~ train size
    base_max_depth=5,
    seed=42
)


# =============================================================================
# MODEL 1 VISUALIZATIONS: CUSTOM BAGGING ENSEMBLE
# =============================================================================
print("="*70)
print("MODEL 1 VISUALIZATIONS: CUSTOM BAGGING ENSEMBLE")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Cross-Validation Scores per Fold
ax1 = axes[0]
folds = list(range(1, 11))
auc_scores = bagging_results['AUC']['scores']
acc_scores = bagging_results['Accuracy']['scores']

ax1.plot(folds, auc_scores, 'o-', label='AUC', color='#2ecc71', linewidth=2, markersize=8)
ax1.plot(folds, acc_scores, 's-', label='Accuracy', color='#3498db', linewidth=2, markersize=8)
ax1.axhline(y=bagging_results['AUC']['mean'], color='#2ecc71', linestyle='--', alpha=0.5, label=f"AUC Mean: {bagging_results['AUC']['mean']:.4f}")
ax1.axhline(y=bagging_results['Accuracy']['mean'], color='#3498db', linestyle='--', alpha=0.5, label=f"Acc Mean: {bagging_results['Accuracy']['mean']:.4f}")
ax1.set_xlabel('Fold', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Bagging: 10-Fold CV Scores', fontsize=13, fontweight='bold')
ax1.set_xticks(folds)
ax1.legend(loc='lower right', fontsize=9)
ax1.set_ylim(0.5, 1.0)
ax1.grid(True, alpha=0.3)

# 2. Performance Metrics Bar Chart
ax2 = axes[1]
metrics = ['AUC', 'Accuracy', 'F1']
means = [bagging_results[m]['mean'] for m in metrics]
stds = [bagging_results[m]['std'] for m in metrics]
colors = ['#2ecc71', '#3498db', '#e74c3c']

bars = ax2.bar(metrics, means, yerr=stds, capsize=8, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Bagging: Performance Metrics', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1.0)
for bar, mean, std in zip(bars, means, stds):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02, 
             f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 3. Score Distribution Boxplot
ax3 = axes[2]
box_data = [bagging_results['AUC']['scores'], bagging_results['Accuracy']['scores'], bagging_results['F1']['scores']]
bp = ax3.boxplot(box_data, labels=metrics, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('Bagging: Score Distribution', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model1_bagging_viz.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nModel 1 visualizations saved as '{OUTPUT_DIR}/model1_bagging_viz.png'")

# ============================================================================
# MODEL 2: CUSTOM ADABOOST (FROM SCRATCH) ✅ (FIXED)
# ============================================================================

print("\n" + "=" * 70)
print("MODEL 2: CUSTOM ADABOOST (FROM SCRATCH) - FIXED")
print("=" * 70)

# -----------------------------------------------------------------------------
# Helper: Convert p1 to Spark probability vector [p0, p1]
# -----------------------------------------------------------------------------
to_prob_vec = udf(lambda p1: Vectors.dense([float(1.0 - p1), float(p1)]), VectorUDT())


# -----------------------------------------------------------------------------
# AdaBoost Training + Prediction (Fixed: keep features; use weightCol)
# -----------------------------------------------------------------------------
def adaboost_predict(train_df, test_df,
                     label_col="APPROVAL_LABEL",
                     features_col="features",
                     rounds=10,
                     seed=42):

    # Ensure row_id exists
    if "row_id" not in train_df.columns:
        train_df = train_df.withColumn("row_id", monotonically_increasing_id())
    if "row_id" not in test_df.columns:
        test_df = test_df.withColumn("row_id", monotonically_increasing_id())

    # Make sure label is double (Spark likes it that way)
    train_df = train_df.withColumn(label_col, col(label_col).cast("double"))
    test_df  = test_df.withColumn(label_col, col(label_col).cast("double"))

    # Initialize uniform weights
    n = train_df.count()
    dfw = train_df.withColumn("w", lit(1.0 / float(n)))

    models = []
    alphas = []

    for t in range(rounds):
        # Weak learner (LR). Use weights via weightCol.
        base = LogisticRegression(
            labelCol=label_col,
            featuresCol=features_col,
            weightCol="w",
            maxIter=50,
            regParam=0.0
        )

        model = base.fit(dfw)
        models.append(model)

        # Predict on training data to compute weighted error
        pred_train = (
            model.transform(dfw)
                 .select("row_id", features_col, label_col, col("prediction").cast("int").alias("pred"), "w")
        )

        pred_train = pred_train.withColumn("mis", (col("pred") != col(label_col)).cast("double"))

        weighted_error = (
            pred_train.withColumn("weighted_mis", col("mis") * col("w"))
                      .agg(spark_sum("weighted_mis").alias("err"))
                      .collect()[0]["err"]
        )
        weighted_error = float(weighted_error)

        # Avoid division by zero / infinity
        eps = 1e-10
        if weighted_error < eps:
            weighted_error = eps
        if weighted_error > 1.0 - eps:
            weighted_error = 1.0 - eps

        alpha = 0.5 * math.log((1.0 - weighted_error) / weighted_error)
        alphas.append(alpha)

        print(f"Round {t+1}/{rounds} - Error: {weighted_error:.4f}, Alpha: {alpha:.4f}")

        # Update weights: w_new = w * exp(alpha * (2*mis - 1))
        pred_train = pred_train.withColumn(
            "w_new",
            col("w") * F.exp(lit(alpha) * (2 * col("mis") - 1))
        )

        # Normalize weights
        total_weight = pred_train.agg(spark_sum("w_new").alias("tw")).collect()[0]["tw"]
        total_weight = float(total_weight)

        # IMPORTANT FIX: keep features_col in dfw for next round
        dfw = (
            pred_train.withColumn("w", col("w_new") / lit(total_weight))
                      .select("row_id", features_col, label_col, "w")
        )

    # -----------------------------------------------------------------------------
    # Prediction Phase (Weighted voting)
    # -----------------------------------------------------------------------------
    score_df = test_df.select("row_id", label_col)

    # Accumulate score = Σ alpha_t * h_t(x), where h_t ∈ {-1, +1}
    score_df = score_df.withColumn("score", lit(0.0))

    for i, (model, alpha) in enumerate(zip(models, alphas)):
        pred = model.transform(test_df).select("row_id", col("prediction").cast("int").alias("pred_i"))
        # convert 0/1 -> -1/+1
        pred = pred.withColumn("h", (col("pred_i") * 2 - 1).cast("double")).drop("pred_i")

        score_df = score_df.join(pred, on="row_id", how="inner")
        score_df = score_df.withColumn("score", col("score") + lit(float(alpha)) * col("h")).drop("h")

    # Sigmoid(score) -> p1
    score_df = score_df.withColumn("p1", 1.0 / (1.0 + F.exp(-col("score"))))
    score_df = score_df.withColumn("prediction", (col("p1") >= lit(0.5)).cast("double"))
    score_df = score_df.withColumn("probability", to_prob_vec(col("p1")))

    return score_df.select("row_id", label_col, "prediction", "probability")


# -----------------------------------------------------------------------------
# Manual K-Fold CV for Custom AdaBoost
# -----------------------------------------------------------------------------
def manual_kfold_cv_custom_adaboost(df,
                                    k=10,
                                    label_col="APPROVAL_LABEL",
                                    features_col="features",
                                    rounds=10,
                                    seed=42):

    df_with_id = df.withColumn("row_id", monotonically_increasing_id())
    df_with_fold = df_with_id.withColumn("fold", (col("row_id") % k).cast("int"))

    auc_eval = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="probability",
        metricName="areaUnderROC"
    )
    acc_eval = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="f1"
    )

    aucs, accs, f1s = [], [], []

    print(f"\nStarting {k}-Fold CV for Custom AdaBoost...")
    print("=" * 60)

    for fold in range(k):
        train_df = df_with_fold.filter(col("fold") != fold)
        test_df  = df_with_fold.filter(col("fold") == fold)

        preds = adaboost_predict(
            train_df=train_df,
            test_df=test_df,
            label_col=label_col,
            features_col=features_col,
            rounds=rounds,
            seed=seed + fold
        )

        auc = auc_eval.evaluate(preds)
        acc = acc_eval.evaluate(preds)
        f1  = f1_eval.evaluate(preds)

        aucs.append(auc); accs.append(acc); f1s.append(f1)

        print(f"Fold {fold+1}/{k} - AUC: {auc:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}")

    print("=" * 60)

    results = {
        "AUC": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs)), "scores": aucs},
        "Accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "scores": accs},
        "F1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s)), "scores": f1s},
    }

    print("\nCV Summary:")
    for metric, vals in results.items():
        print(f"{metric}: {vals['mean']:.4f} (+/- {vals['std']:.4f})")

    return results


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
adaboost_results = manual_kfold_cv_custom_adaboost(
    df=df_transformed,
    k=10,
    label_col="APPROVAL_LABEL",
    features_col="features",
    rounds=10,
    seed=42
)


# =============================================================================
# MODEL 2 VISUALIZATIONS: CUSTOM ADABOOST
# =============================================================================
print("\n" + "="*70)
print("MODEL 2 VISUALIZATIONS: CUSTOM ADABOOST")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Cross-Validation Scores per Fold
ax1 = axes[0]
folds = list(range(1, 11))
auc_scores = adaboost_results['AUC']['scores']
acc_scores = adaboost_results['Accuracy']['scores']

ax1.plot(folds, auc_scores, 'o-', label='AUC', color='#9b59b6', linewidth=2, markersize=8)
ax1.plot(folds, acc_scores, 's-', label='Accuracy', color='#e67e22', linewidth=2, markersize=8)
ax1.axhline(y=adaboost_results['AUC']['mean'], color='#9b59b6', linestyle='--', alpha=0.5, label=f"AUC Mean: {adaboost_results['AUC']['mean']:.4f}")
ax1.axhline(y=adaboost_results['Accuracy']['mean'], color='#e67e22', linestyle='--', alpha=0.5, label=f"Acc Mean: {adaboost_results['Accuracy']['mean']:.4f}")
ax1.set_xlabel('Fold', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('AdaBoost: 10-Fold CV Scores', fontsize=13, fontweight='bold')
ax1.set_xticks(folds)
ax1.legend(loc='lower right', fontsize=9)
ax1.set_ylim(0.5, 1.0)
ax1.grid(True, alpha=0.3)

# 2. Performance Metrics Bar Chart
ax2 = axes[1]
metrics = ['AUC', 'Accuracy', 'F1']
means = [adaboost_results[m]['mean'] for m in metrics]
stds = [adaboost_results[m]['std'] for m in metrics]
colors = ['#9b59b6', '#e67e22', '#1abc9c']

bars = ax2.bar(metrics, means, yerr=stds, capsize=8, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('AdaBoost: Performance Metrics', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1.0)
for bar, mean, std in zip(bars, means, stds):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02, 
             f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 3. Score Distribution Boxplot
ax3 = axes[2]
box_data = [adaboost_results['AUC']['scores'], adaboost_results['Accuracy']['scores'], adaboost_results['F1']['scores']]
bp = ax3.boxplot(box_data, labels=metrics, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('AdaBoost: Score Distribution', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model2_adaboost_viz.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nModel 2 visualizations saved as '{OUTPUT_DIR}/model2_adaboost_viz.png'")

# ============================================================================
# MODEL 3: LOGISTIC REGRESSION (BASELINE MODEL)
# ============================================================================

print("\n" + "=" * 70)
print("MODEL 3: LOGISTIC REGRESSION (BASELINE)")
print("=" * 70)

# Define Logistic Regression
lr_classifier = LogisticRegression(
    labelCol='APPROVAL_LABEL',
    featuresCol='features',
    maxIter=100,
    regParam=0.01,        # Regularization parameter
    elasticNetParam=0.5   # Mix of L1 and L2 regularization
)

# Run manual 10-fold cross-validation for Logistic Regression
lr_results = manual_kfold_cross_validation(
    df=df_transformed,
    model=lr_classifier,
    k=10,
    label_col='APPROVAL_LABEL',
    features_col='features'
)

# =============================================================================
# MODEL 3 VISUALIZATIONS: LOGISTIC REGRESSION
# =============================================================================
print("\n" + "="*70)
print("MODEL 3 VISUALIZATIONS: LOGISTIC REGRESSION")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Cross-Validation Scores per Fold
ax1 = axes[0]
folds = list(range(1, 11))
auc_scores = lr_results['AUC']['scores']
acc_scores = lr_results['Accuracy']['scores']

ax1.plot(folds, auc_scores, 'o-', label='AUC', color='#e74c3c', linewidth=2, markersize=8)
ax1.plot(folds, acc_scores, 's-', label='Accuracy', color='#34495e', linewidth=2, markersize=8)
ax1.axhline(y=lr_results['AUC']['mean'], color='#e74c3c', linestyle='--', alpha=0.5, label=f"AUC Mean: {lr_results['AUC']['mean']:.4f}")
ax1.axhline(y=lr_results['Accuracy']['mean'], color='#34495e', linestyle='--', alpha=0.5, label=f"Acc Mean: {lr_results['Accuracy']['mean']:.4f}")
ax1.set_xlabel('Fold', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Logistic Regression: 10-Fold CV Scores', fontsize=13, fontweight='bold')
ax1.set_xticks(folds)
ax1.legend(loc='lower right', fontsize=9)
ax1.set_ylim(0.5, 1.0)
ax1.grid(True, alpha=0.3)

# 2. Performance Metrics Bar Chart
ax2 = axes[1]
metrics = ['AUC', 'Accuracy', 'F1', 'Precision', 'Recall']
means = [lr_results[m]['mean'] for m in metrics]
stds = [lr_results[m]['std'] for m in metrics]
colors = ['#e74c3c', '#34495e', '#f39c12', '#27ae60', '#8e44ad']

bars = ax2.bar(metrics, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Logistic Regression: All Metrics', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1.0)
ax2.tick_params(axis='x', rotation=15)
for bar, mean in zip(bars, means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
             f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 3. Score Distribution Boxplot
ax3 = axes[2]
box_metrics = ['AUC', 'Accuracy', 'F1']
box_data = [lr_results[m]['scores'] for m in box_metrics]
bp = ax3.boxplot(box_data, labels=box_metrics, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#e74c3c', '#34495e', '#f39c12']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('Logistic Regression: Score Distribution', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model3_lr_viz.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nModel 3 visualizations saved as '{OUTPUT_DIR}/model3_lr_viz.png'")

# ============================================================================
# MODEL COMPARISON SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY (10-FOLD CROSS-VALIDATION)")
print("=" * 70)

print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY (10-FOLD CROSS-VALIDATION)")
print("="*70)

# Create comparison DataFrame with actual model names
comparison_data = {
    'Model': ['Custom Bagging', 'Custom AdaBoost', 'Logistic Regression'],
    'AUC Mean': [bagging_results['AUC']['mean'], adaboost_results['AUC']['mean'], lr_results['AUC']['mean']],
    'AUC Std': [bagging_results['AUC']['std'], adaboost_results['AUC']['std'], lr_results['AUC']['std']],
    'Accuracy Mean': [bagging_results['Accuracy']['mean'], adaboost_results['Accuracy']['mean'], lr_results['Accuracy']['mean']],
    'Accuracy Std': [bagging_results['Accuracy']['std'], adaboost_results['Accuracy']['std'], lr_results['Accuracy']['std']],
    'F1 Mean': [bagging_results['F1']['mean'], adaboost_results['F1']['mean'], lr_results['F1']['mean']],
    'F1 Std': [bagging_results['F1']['std'], adaboost_results['F1']['std'], lr_results['F1']['std']],
}

comparison_df = pd.DataFrame(comparison_data)
print("\nDetailed Model Comparison:")
print(comparison_df.to_string(index=False))

# Find best model
best_model_idx = comparison_df['AUC Mean'].idxmax()
print(f"\n*** Best Model by AUC: {comparison_df.loc[best_model_idx, 'Model']} ***")
print(f"    AUC: {comparison_df.loc[best_model_idx, 'AUC Mean']:.4f} (+/- {comparison_df.loc[best_model_idx, 'AUC Std']:.4f})")

# ============================================================================
# MODEL COMPARISON VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("MODEL COMPARISON VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Model Comparison Bar Chart (AUC)
ax1 = axes[0, 0]
models = ['Custom\nBagging', 'Custom\nAdaBoost', 'Logistic\nRegression']
auc_means = [bagging_results['AUC']['mean'], adaboost_results['AUC']['mean'], lr_results['AUC']['mean']]
auc_stds = [bagging_results['AUC']['std'], adaboost_results['AUC']['std'], lr_results['AUC']['std']]
colors = ['#2ecc71', '#9b59b6', '#e74c3c']

bars = ax1.bar(models, auc_means, yerr=auc_stds, capsize=8, color=colors, edgecolor='black', linewidth=2)
ax1.set_ylabel('AUC Score', fontsize=12)
ax1.set_title('Model Comparison - AUC (10-Fold CV)', fontsize=13, fontweight='bold')
ax1.set_ylim(0.5, 1.0)
for bar, mean in zip(bars, auc_means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f'{mean:.4f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
ax1.legend()

# 2. Multiple Metrics Comparison
ax2 = axes[0, 1]
x = np.arange(3)
width = 0.25

auc_vals = [bagging_results['AUC']['mean'], adaboost_results['AUC']['mean'], lr_results['AUC']['mean']]
acc_vals = [bagging_results['Accuracy']['mean'], adaboost_results['Accuracy']['mean'], lr_results['Accuracy']['mean']]
f1_vals = [bagging_results['F1']['mean'], adaboost_results['F1']['mean'], lr_results['F1']['mean']]

ax2.bar(x - width, auc_vals, width, label='AUC', color='#3498db', edgecolor='black')
ax2.bar(x, acc_vals, width, label='Accuracy', color='#2ecc71', edgecolor='black')
ax2.bar(x + width, f1_vals, width, label='F1 Score', color='#e74c3c', edgecolor='black')

ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('All Models - Multiple Metrics', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['Bagging', 'AdaBoost', 'LR'])
ax2.legend(loc='lower right')
ax2.set_ylim(0.5, 1.0)
ax2.grid(True, alpha=0.3, axis='y')

# 3. AUC Scores Distribution (Boxplot)
ax3 = axes[1, 0]
box_data = [bagging_results['AUC']['scores'], adaboost_results['AUC']['scores'], lr_results['AUC']['scores']]
bp = ax3.boxplot(box_data, labels=['Bagging', 'AdaBoost', 'LR'], patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_ylabel('AUC Score', fontsize=12)
ax3.set_title('AUC Score Distribution Across Folds', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Radar/Spider Chart for all metrics
ax4 = axes[1, 1]
metrics_names = ['AUC', 'Accuracy', 'F1']

# Prepare data for grouped bar comparison
bagging_vals = [bagging_results[m]['mean'] for m in metrics_names]
adaboost_vals = [adaboost_results[m]['mean'] for m in metrics_names]
lr_vals_all = [lr_results[m]['mean'] for m in metrics_names]

x_pos = np.arange(len(metrics_names))
width = 0.25

ax4.barh(x_pos - width, bagging_vals, width, label='Bagging', color='#2ecc71', edgecolor='black')
ax4.barh(x_pos, adaboost_vals, width, label='AdaBoost', color='#9b59b6', edgecolor='black')
ax4.barh(x_pos + width, lr_vals_all, width, label='LR', color='#e74c3c', edgecolor='black')

ax4.set_xlabel('Score', fontsize=12)
ax4.set_title('Horizontal Comparison by Metric', fontsize=13, fontweight='bold')
ax4.set_yticks(x_pos)
ax4.set_yticklabels(metrics_names)
ax4.legend(loc='lower right')
ax4.set_xlim(0.5, 1.0)
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/model_comparison_viz.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nModel comparison visualizations saved as '{OUTPUT_DIR}/model_comparison_viz.png'")

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS (USING GBT)
# ============================================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

# Train final GBT model on full dataset
final_gbt = GBTClassifier(
    labelCol='APPROVAL_LABEL',
    featuresCol='features',
    maxIter=50,
    maxDepth=5,
    stepSize=0.1,
    seed=42
)

final_gbt_model = final_gbt.fit(df_transformed)

# Get feature importances
feature_names = [
    'AGE_AT_APPLICATION',
    'APPLICANT_FAMILY_COUNT',
    'APPLICATION_YEAR',
    'MARITAL_STATUS_1', 'MARITAL_STATUS_2', 'MARITAL_STATUS_3', 'MARITAL_STATUS_4',
    'GENDER_1', 'GENDER_2'
]

importances = final_gbt_model.featureImportances.toArray()

# Handle case where feature names don't match
if len(importances) > len(feature_names):
    feature_names = [f'Feature_{i}' for i in range(len(importances))]
    feature_names[:3] = ['AGE_AT_APPLICATION', 'APPLICANT_FAMILY_COUNT', 'APPLICATION_YEAR']

importance_df = pd.DataFrame({
    'Feature': feature_names[:len(importances)],
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Ranking (GBT):")
print(importance_df.to_string(index=False))

# Identify most predictive features
print(f"\nTop 3 Most Important Features:")
for idx, row in importance_df.head(3).iterrows():
    print(f"  - {row['Feature']}: {row['Importance']:.4f}")

# ============================================================================
# FEATURE IMPORTANCE VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("FEATURE IMPORTANCE VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Horizontal Bar Chart - Feature Importance
ax1 = axes[0]
top_n = min(10, len(importance_df))
top_features = importance_df.head(top_n)

colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
bars = ax1.barh(range(top_n), top_features['Importance'].values, color=colors, edgecolor='black')
ax1.set_yticks(range(top_n))
ax1.set_yticklabels(top_features['Feature'].values)
ax1.invert_yaxis()
ax1.set_xlabel('Importance Score', fontsize=12)
ax1.set_title('Feature Importance Ranking (GBT Model)', fontsize=13, fontweight='bold')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_features['Importance'].values)):
    ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
             va='center', fontsize=10)

# 2. Pie Chart - Feature Importance Distribution
ax2 = axes[1]
# Group smaller features if there are many
if len(importance_df) > 5:
    top_5 = importance_df.head(5)
    others_sum = importance_df.iloc[5:]['Importance'].sum()
    pie_labels = list(top_5['Feature'].values) + ['Others']
    pie_values = list(top_5['Importance'].values) + [others_sum]
else:
    pie_labels = importance_df['Feature'].tolist()
    pie_values = importance_df['Importance'].tolist()

colors_pie = plt.cm.Set3(np.linspace(0, 1, len(pie_labels)))
wedges, texts, autotexts = ax2.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', 
                                    colors=colors_pie, startangle=90)
ax2.set_title('Feature Importance Distribution', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance_viz.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFeature importance visualizations saved as '{OUTPUT_DIR}/feature_importance_viz.png'")

# # =============================================================================
# # PART 3: CLUSTERING ANALYSIS
# # =============================================================================
# # Research Question Part 2: Can applicants be clustered into meaningful groups?


# ============================================================================
# PART 3: K-MEANS CLUSTERING ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING ANALYSIS")
print("=" * 70)

# Find optimal number of clusters using Silhouette Score
silhouette_scores = []
k_values = range(2, 11)

print("\nFinding optimal number of clusters (k)...")
print("-" * 40)

for k in k_values:
    kmeans = KMeans(
        featuresCol='features',
        predictionCol='cluster',
        k=k,
        seed=42,
        maxIter=50
    )

    model = kmeans.fit(df_transformed)
    predictions = model.transform(df_transformed)

    evaluator = ClusteringEvaluator(
        featuresCol='features',
        predictionCol='cluster',
        metricName='silhouette'
    )

    silhouette = evaluator.evaluate(predictions)
    silhouette_scores.append(silhouette)
    print(f"k = {k}: Silhouette Score = {silhouette:.4f}")

# Find optimal k
optimal_k = k_values[np.argmax(silhouette_scores)]
print(f"\n*** Optimal number of clusters: k = {optimal_k} ***")
print(f"    Best Silhouette Score: {np.max(silhouette_scores):.4f}")

# ============================================================================
# FINAL CLUSTERING WITH OPTIMAL K & CLUSTER PROFILING
# ============================================================================

print("\n" + "=" * 70)
print(f"CLUSTER PROFILING (k = {optimal_k})")
print("=" * 70)

# Train final K-Means model with optimal k
final_kmeans = KMeans(
    featuresCol='features',
    predictionCol='cluster',
    k=optimal_k,
    seed=42,
    maxIter=100
)

final_kmeans_model = final_kmeans.fit(df_transformed)
clustered_df = final_kmeans_model.transform(df_transformed)

# Analyze cluster characteristics
print("\nCluster Distribution:")
clustered_df.groupBy('cluster').count().orderBy('cluster').show()

# Profile each cluster
print("\nCluster Profiles:")
print("-" * 60)

for cluster_id in range(optimal_k):
    cluster_data = clustered_df.filter(col('cluster') == cluster_id)
    cluster_count = cluster_data.count()

    print(f"\n*** CLUSTER {cluster_id} ({cluster_count} applicants) ***")

    # Average age
    avg_age = cluster_data.agg(avg('AGE_AT_APPLICATION')).collect()[0][0]
    print(f"  Average Age at Application: {avg_age:.1f} years")

    # Average family count
    avg_family = cluster_data.agg(avg('APPLICANT_FAMILY_COUNT')).collect()[0][0]
    print(f"  Average Family Count: {avg_family:.1f}")

    # Dominant marital status
    print(f"  Marital Status Distribution:")
    cluster_data.groupBy('MARITAL_STATUS').count().orderBy('count', ascending=False).show(5, truncate=False)

    # Dominant gender
    print(f"  Gender Distribution:")
    cluster_data.groupBy('GENDER_DESC').count().orderBy('count', ascending=False).show(5, truncate=False)

    # Approval rate
    approval_rate = cluster_data.filter(col('APPROVAL_LABEL') == 1).count() / cluster_count * 100
    print(f"  Approval Rate: {approval_rate:.2f}%")

# ============================================================================

print("\n" + "="*70)
print("FINAL COMPREHENSIVE VISUALIZATION DASHBOARD")
print("="*70)

fig = plt.figure(figsize=(18, 14))

# 1. Model Comparison Bar Chart (Top Left)
ax1 = fig.add_subplot(2, 3, 1)
models = ['Bagging', 'AdaBoost', 'LR']
auc_means = [bagging_results['AUC']['mean'], adaboost_results['AUC']['mean'], lr_results['AUC']['mean']]
auc_stds = [bagging_results['AUC']['std'], adaboost_results['AUC']['std'], lr_results['AUC']['std']]
colors = ['#2ecc71', '#9b59b6', '#e74c3c']

bars = ax1.bar(models, auc_means, yerr=auc_stds, capsize=5, color=colors, edgecolor='black')
ax1.set_ylabel('AUC Score')
ax1.set_title('Model Comparison (10-Fold CV)', fontweight='bold')
ax1.set_ylim(0.5, 1.0)
for bar, mean in zip(bars, auc_means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{mean:.3f}',
             ha='center', va='bottom', fontweight='bold')

# 2. Silhouette Scores (Top Middle)
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(list(k_values), silhouette_scores, 'bo-', markersize=8, linewidth=2)
ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Optimal Cluster Selection', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Feature Importance (Top Right)
ax3 = fig.add_subplot(2, 3, 3)
top_features = importance_df.head(8)
ax3.barh(top_features['Feature'], top_features['Importance'], color='#9b59b6', edgecolor='black')
ax3.set_xlabel('Importance')
ax3.set_title('Feature Importance (GBT)', fontweight='bold')
ax3.invert_yaxis()

# 4. Cluster Distribution (Bottom Left)
ax4 = fig.add_subplot(2, 3, 4)
cluster_counts = clustered_df.groupBy('cluster').count().orderBy('cluster').toPandas()
ax4.pie(cluster_counts['count'], labels=[f'Cluster {i}' for i in cluster_counts['cluster']],
        autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors[:optimal_k])
ax4.set_title(f'Cluster Distribution (k={optimal_k})', fontweight='bold')

# 5. Cluster Approval Rates (Bottom Middle)
ax5 = fig.add_subplot(2, 3, 5)
cluster_approval = clustered_df.groupBy('cluster').agg(
    {'APPROVAL_LABEL': 'avg'}
).withColumnRenamed('avg(APPROVAL_LABEL)', 'approval_rate').orderBy('cluster').toPandas()

bars = ax5.bar(cluster_approval['cluster'].astype(str), cluster_approval['approval_rate'] * 100,
               color=plt.cm.Set2.colors[:len(cluster_approval)], edgecolor='black')
ax5.set_xlabel('Cluster')
ax5.set_ylabel('Approval Rate (%)')
ax5.set_title('Approval Rate by Cluster', fontweight='bold')
for bar, rate in zip(bars, cluster_approval['approval_rate']):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate*100:.1f}%',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 6. All Models - All Metrics (Bottom Right)
ax6 = fig.add_subplot(2, 3, 6)
x = np.arange(3)
width = 0.25

auc_all = [bagging_results['AUC']['mean'], adaboost_results['AUC']['mean'], lr_results['AUC']['mean']]
acc_all = [bagging_results['Accuracy']['mean'], adaboost_results['Accuracy']['mean'], lr_results['Accuracy']['mean']]
f1_all = [bagging_results['F1']['mean'], adaboost_results['F1']['mean'], lr_results['F1']['mean']]

ax6.bar(x - width, auc_all, width, label='AUC', color='#3498db', edgecolor='black')
ax6.bar(x, acc_all, width, label='Accuracy', color='#2ecc71', edgecolor='black')
ax6.bar(x + width, f1_all, width, label='F1', color='#e74c3c', edgecolor='black')
ax6.set_ylabel('Score')
ax6.set_title('All Metrics Comparison', fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(['Bagging', 'AdaBoost', 'LR'])
ax6.legend(loc='lower right', fontsize=8)
ax6.set_ylim(0.5, 1.0)

plt.suptitle('Machine Learning Analysis Dashboard - Land Grant Applications', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ml_analysis_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE - All results and visualizations saved to output directory")
print("=" * 70)

spark.stop()
print("\nSpark session stopped.")