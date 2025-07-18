import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.pipeline.predict_pipeline import PredictPipeline, Customdata

# Caching the dataset loading for performance
@st.cache_data
def load_data():
    # <-- Replace below path with your actual data file path -->
    df = pd.read_csv("/home/apurva/mlproject/notebook/data/stud.csv")
    return df

def eda_page(df):
    st.title("Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Columns")
    st.write(df.columns.tolist())

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Basic Info")
    st.markdown(f"- **Shape:** {df.shape}")
    st.markdown(f"- **Number of features:** {df.shape[1]}")
    st.markdown(f"- **Number of samples:** {df.shape[0]}")

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Summary Statistics")
    st.write(df.describe(include='all'))  # include all data types

    st.markdown("---")
    st.subheader("Categorical Feature Distributions")

    categorical_cols = [
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course"
    ]

    for col in categorical_cols:
        if col in df.columns:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"{col} Value Counts")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.warning(f"Column '{col}' not found in dataset, skipping plot.")

    st.markdown("---")
    st.subheader("Numerical Feature Distributions")

    numeric_cols = ["reading score", "writing score"]

    for col in numeric_cols:
        if col in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, bins=30, ax=ax)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)
        else:
            st.warning(f"Column '{col}' not found in dataset, skipping plot.")

    st.markdown("---")
    st.subheader("Box Plots for Outlier Detection")

    for col in numeric_cols:
        if col in df.columns:
            fig, ax = plt.subplots()
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(f"Box Plot of {col}")
            st.pyplot(fig)
        else:
            st.warning(f"Column '{col}' not found in dataset, skipping plot.")

    st.markdown("---")
    st.subheader("Correlation Heatmap (Numerical Features Only)")

    # Only keep numeric_cols that exist in the df
    valid_numeric_cols = [col for col in numeric_cols if col in df.columns]

    if len(valid_numeric_cols) > 1:
        fig, ax = plt.subplots()
        corr = df[valid_numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numerical columns available for correlation heatmap.")

def predict_page():
    st.title("Predict Student Performance")

    st.write("Fill in the details below to get a prediction:")

    gender = st.selectbox("Gender", ["male", "female"])
    race_ethnicity = st.selectbox("Race / Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
    parental_level_of_education = st.selectbox(
        "Parental Level of Education", [
            "some high school", "high school", "some college",
            "associate's degree", "bachelor's degree", "master's degree"
        ])
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
    reading_score = st.number_input("Reading Score", min_value=0.0, max_value=100.0, step=1.0, value=50.0)
    writing_score = st.number_input("Writing Score", min_value=0.0, max_value=100.0, step=1.0, value=50.0)

    if st.button("Predict"):
        # Prepare the data
        data = Customdata(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        input_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(input_df)

        st.success(f"Predicted Math Score: {prediction[0]:.2f}")


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["EDA", "Prediction"])

    df = load_data()

    if page == "EDA":
        eda_page(df)
    elif page == "Prediction":
        predict_page()


if __name__ == "__main__":
    main()
