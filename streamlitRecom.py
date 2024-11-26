import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('EdX.csv')
    data.columns = map(str.lower, data.columns)
    return data

data = load_data()

# Preprocessing - Create User-Course Interaction Matrix
num_users = 100
num_courses = data.shape[0]
np.random.seed(0)
user_course_matrix = np.random.randint(2, size=(num_users, num_courses))

# Matrix Factorization with SVD
svd = TruncatedSVD(n_components=20, random_state=42)
user_factors = svd.fit_transform(user_course_matrix)
course_factors = svd.components_.T
predicted_matrix = np.dot(user_factors, course_factors.T)

# Collaborative Filtering Predictions
user_similarity = cosine_similarity(user_course_matrix)
np.fill_diagonal(user_similarity, 0)
predicted_ratings_cf = np.dot(user_similarity, user_course_matrix) / np.array([np.abs(user_similarity).sum(axis=1)]).T

# Recommendation Functions
def recommend_courses_cf(user_id, num_recommendations):
    user_ratings_cf = predicted_ratings_cf[user_id]
    recommended_courses_indices = np.argsort(user_ratings_cf)[::-1][:num_recommendations]
    return data.iloc[recommended_courses_indices][['name', 'university', 'difficulty level', 'link']]

def recommend_courses_mf(user_id, num_recommendations):
    user_ratings_mf = predicted_matrix[user_id]
    recommended_courses_indices = np.argsort(user_ratings_mf)[::-1][:num_recommendations]
    return data.iloc[recommended_courses_indices][['name', 'university', 'difficulty level', 'link']]

# Streamlit App UI
st.title("Course Recommendation System")
st.sidebar.header("User Inputs")

# Sidebar Inputs
user_id = st.sidebar.number_input("Enter User ID", min_value=0, max_value=num_users-1, value=0, step=1)
num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, value=5)
method = st.sidebar.selectbox("Recommendation Method", ["Collaborative Filtering", "Matrix Factorization"])

# Display the Data
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Course Dataset")
    st.dataframe(data)

# Generate Recommendations
if st.sidebar.button("Get Recommendations"):
    if method == "Collaborative Filtering":
        recommendations = recommend_courses_cf(user_id, num_recommendations)
        st.subheader(f"Recommendations for User {user_id} (Collaborative Filtering):")
    else:
        recommendations = recommend_courses_mf(user_id, num_recommendations)
        st.subheader(f"Recommendations for User {user_id} (Matrix Factorization):")
    st.dataframe(recommendations)

# Evaluation Metrics Section
st.sidebar.subheader("Evaluation Metrics")
threshold = st.sidebar.slider("Classification Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

# Evaluate Matrix Factorization
binary_predictions_mf = (predicted_matrix >= threshold).astype(int)
y_pred_mf = binary_predictions_mf.flatten()
y_true_mf = user_course_matrix.flatten()

accuracy_mf = accuracy_score(y_true_mf, y_pred_mf)
precision_mf = precision_score(y_true_mf, y_pred_mf, zero_division=0)
recall_mf = recall_score(y_true_mf, y_pred_mf, zero_division=0)
f1_mf = f1_score(y_true_mf, y_pred_mf, zero_division=0)

# Evaluate Collaborative Filtering
binary_predictions_cf = (predicted_ratings_cf >= threshold).astype(int)
y_pred_cf = binary_predictions_cf.flatten()

accuracy_cf = accuracy_score(y_true_mf, y_pred_cf)
precision_cf = precision_score(y_true_mf, y_pred_cf, zero_division=0)
recall_cf = recall_score(y_true_mf, y_pred_cf, zero_division=0)
f1_cf = f1_score(y_true_mf, y_pred_cf, zero_division=0)

# Display Evaluation Metrics
# st.sidebar.markdown("### Matrix Factorization Metrics")
# st.sidebar.write(f"Accuracy: {accuracy_mf:.2f}")
# st.sidebar.write(f"Precision: {precision_mf:.2f}")
# st.sidebar.write(f"Recall: {recall_mf:.2f}")
# st.sidebar.write(f"F1 Score: {f1_mf:.2f}")

# st.sidebar.markdown("### Collaborative Filtering Metrics")
# st.sidebar.write(f"Accuracy: {accuracy_cf:.2f}")
# st.sidebar.write(f"Precision: {precision_cf:.2f}")
# st.sidebar.write(f"Recall: {recall_cf:.2f}")
# st.sidebar.write(f"F1 Score: {f1_cf:.2f}")

# Plot ROC Curve for Matrix Factorization
# fpr_mf, tpr_mf, _ = roc_curve(y_true_mf, y_pred_mf)
# roc_auc_mf = auc(fpr_mf, tpr_mf)

# Plot ROC Curve for Collaborative Filtering
# fpr_cf, tpr_cf, _ = roc_curve(y_true_mf, y_pred_cf)
# roc_auc_cf = auc(fpr_cf, tpr_cf)

# st.sidebar.markdown("### ROC Curve")
# if st.sidebar.checkbox("Show ROC Curve"):
 #   st.markdown("#### ROC Curve for Matrix Factorization")
 #   st.line_chart({"FPR (MF)": fpr_mf, "TPR (MF)": tpr_mf})
 #   st.markdown("#### ROC Curve for Collaborative Filtering")
 #   st.line_chart({"FPR (CF)": fpr_cf, "TPR (CF)": tpr_cf})
