import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

import joblib

# ---------------------------------------
# Streamlit Page Configuration
# ---------------------------------------
st.set_page_config(page_title="Semiconductor Yield Prediction", layout="wide")
st.title("🔬 Semiconductor Yield Prediction Dashboard")

# ---------------------------------------
# 1️⃣ Data Upload & Exploration
# ---------------------------------------
st.header("1️⃣ Data Upload & Exploration")
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Data Loaded Successfully!")

    # Display preview
    st.write("### Preview of the Dataset", data.head())

    # Data Info
    with st.expander("🔍 Data Info & Description"):
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.write("### Data Description")
        st.write(data.describe())

    # Check for missing values
    missing_values = data.isnull().sum().sum()
    if missing_values > 0:
        st.warning(f"⚠️ Missing values detected: {missing_values}. Filling them appropriately...")

        # Separate numeric and non-numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns

        # Fill numeric columns with median
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

        # Fill non-numeric columns with mode
        for col in non_numeric_cols:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].mode()[0])

        st.success("✅ Missing values filled: numeric (median), non-numeric (mode).")
    else:
        st.success("✅ No missing values detected!")

    # Target column
    target_col = data.columns[-1]
    st.write(f"🎯 **Target Column:** `{target_col}`")
    st.write(data[target_col].value_counts())

    # Target distribution
    fig, ax = plt.subplots()
    sns.countplot(x=target_col, data=data, ax=ax, palette="pastel")
    ax.set_title("Target Distribution")
    st.pyplot(fig)

    # Correlation Heatmap (numeric columns only)
    with st.expander("📊 Correlation Heatmap (numeric features)"):
        numeric_data = data.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(numeric_data.corr(), cmap="coolwarm", center=0)
        st.pyplot(fig)

# ---------------------------------------
# 2️⃣ Data Preprocessing
# ---------------------------------------
    st.header("2️⃣ Data Preprocessing")

    # Features and Target segregation
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Convert non-numeric features (like dates) to numeric if needed
    for col in X.select_dtypes(include=["object", "datetime64"]).columns:
        try:
            X[col] = pd.to_datetime(X[col]).astype(int) / 10**9  # Convert to UNIX timestamp
        except Exception:
            X[col] = pd.factorize(X[col])[0]  # Label encoding for categorical data

    # Class balancing
    st.write("✅ Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    st.write("✅ Class balancing done! Balanced class distribution:")
    st.write(y.value_counts())

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    # Scaling
    st.write("✅ Scaling the features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for future use
    joblib.dump(scaler, "scaler.pkl")

# ---------------------------------------
# 3️⃣ Model Training & Evaluation
# ---------------------------------------
    st.header("3️⃣ Model Training & Evaluation")

    selected_model = st.selectbox(
        "Select Model", ["RandomForest", "SVM", "NaiveBayes"]
    )

    if st.button("🚀 Train & Evaluate Model"):
        if selected_model == "RandomForest":
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            }
            grid = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=3,
                n_jobs=-1,
                scoring="accuracy",
            )
            grid.fit(X_train_scaled, y_train)
            best_model = grid.best_estimator_
            st.write("✅ Best Hyperparameters:", grid.best_params_)

        elif selected_model == "SVM":
            param_grid = {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
            }
            grid = GridSearchCV(
                SVC(random_state=42),
                param_grid,
                cv=3,
                n_jobs=-1,
                scoring="accuracy",
            )
            grid.fit(X_train_scaled, y_train)
            best_model = grid.best_estimator_
            st.write("✅ Best Hyperparameters:", grid.best_params_)

        else:  # NaiveBayes
            best_model = GaussianNB()
            best_model.fit(X_train_scaled, y_train)

        # Evaluation
        y_train_pred = best_model.predict(X_train_scaled)
        y_test_pred = best_model.predict(X_test_scaled)

        st.subheader("📈 Classification Report (Train)")
        st.text(classification_report(y_train, y_train_pred))

        st.subheader("📈 Classification Report (Test)")
        st.text(classification_report(y_test, y_test_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # Save the trained model
        model_filename = f"{selected_model}_best_model.pkl"
        joblib.dump(best_model, model_filename)
        st.success(f"✅ {selected_model} model saved as `{model_filename}`")

# ---------------------------------------
# 4️⃣ Model Comparison
# ---------------------------------------
    st.header("4️⃣ Model Comparison")

    models = ["RandomForest", "SVM", "NaiveBayes"]
    results = {}

    for model_name in models:
        try:
            model = joblib.load(f"{model_name}_best_model.pkl")
            train_acc = model.score(X_train_scaled, y_train)
            test_acc = model.score(X_test_scaled, y_test)
            results[model_name] = {
                "Train Accuracy": round(train_acc, 3),
                "Test Accuracy": round(test_acc, 3),
            }
        except FileNotFoundError:
            continue

    if results:
        df_results = pd.DataFrame(results).T
        st.dataframe(df_results.style.highlight_max(axis=0))

        fig, ax = plt.subplots()
        df_results.plot(kind="bar", ax=ax)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Train vs Test Accuracy Comparison")
        st.pyplot(fig)
    else:
        st.info("No models trained yet. Please train at least one model.")

# ---------------------------------------
# 5️⃣ Conclusion
# ---------------------------------------
    st.header("5️⃣ Conclusion & Next Steps")
    st.write("""
✅ We have:
- Cleaned & balanced the dataset (including mixed data types like dates).
- Trained **RandomForest**, **SVM**, and **NaiveBayes** models.
- Tuned hyperparameters (where applicable).
- Evaluated & compared models visually.

🎉 Ready for deployment or further optimization!
""")
