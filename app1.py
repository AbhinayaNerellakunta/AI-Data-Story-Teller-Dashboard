# ==============================
# AI Data Story Teller Dashboard
# Redesigned Layout (Sidebar + Tabs)
# ==============================

# --- Imports ---
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# --- Page Config ---
st.set_page_config(
    page_title="AI Data Story Teller",
    page_icon="📊",
    layout="wide"
)

# --- Title ---
st.markdown("<h1 style='color:#4CAF50;'> AI Data Story Teller</h1>", unsafe_allow_html=True)
st.write("Upload a CSV file and explore it with automated EDA, visualizations, and insights.")

# --- Sidebar (File Upload) ---
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])

# --- If file is uploaded ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Tabs for sections ---
    tab1, tab2, tab3 = st.tabs([" Overview", " Visualizations", " Insights"])

    # ======================
    # Tab 1: Dataset Overview
    # ======================
    with tab1:
        st.subheader(" Dataset Preview")
        st.write(df.head().astype(str))


        st.subheader(" Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.subheader(" Column Data Types")
        st.write(df.dtypes)

        st.subheader(" Missing Values")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])

        st.subheader(" Summary Statistics")
        st.write(df.describe())

    # ======================
    # Tab 2: Visualizations
    # ======================
    with tab2:
        st.subheader(" Visualizations")

        numeric_cols = df.select_dtypes(include=['int64','float64']).columns
        cat_cols = df.select_dtypes(include=['object']).columns

        # --- Histogram (left) + Top Categories (right) ---
        col1, col2 = st.columns(2)
        with col1:
            if len(numeric_cols) > 0:
                st.markdown("#### Histogram")
                num_col = st.selectbox("Select numeric column", numeric_cols, key="hist")
                fig, ax = plt.subplots(figsize=(5,3))
                sns.histplot(df[num_col], bins=30, kde=True, ax=ax, color="skyblue")
                st.pyplot(fig)
        with col2:
            if len(cat_cols) > 0:
                st.markdown("#### Top Categories")
                cat_col = st.selectbox("Select categorical column", cat_cols, key="topcat")
                top_cats = df[cat_col].value_counts().head(10)
                st.bar_chart(top_cats)

        # --- Correlation Heatmap ---
        if len(numeric_cols) > 1:
            st.markdown("#### Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(6,4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # --- Boxplot ---
        if len(numeric_cols) > 0 and len(cat_cols) > 0:
            st.markdown("#### Boxplot (Numeric vs Category)")
            num_col = st.selectbox("Select numeric column", numeric_cols, key="box_num")
            cat_col = st.selectbox("Select categorical column", cat_cols, key="box_cat")
            fig, ax = plt.subplots(figsize=(4,3))
            sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
            ax.set_xlabel(cat_col, fontsize=8)      # smaller x-axis label
            ax.set_ylabel(num_col, fontsize=8)      
            plt.xticks(rotation=30, fontsize=7)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

    # ======================
    # Tab 3: Insights
    # ======================
    with tab3:
        st.subheader(" Automated Insights")

        insights = []

        # Dataset size
        insights.append(f"The dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

        # Missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            insights.append(f"Missing values found in: {', '.join(missing_cols)}.")
        else:
            insights.append("No missing values detected.")

        # Numeric insights
        if len(numeric_cols) > 0:
            desc = df[numeric_cols].describe().T
            for col in numeric_cols:
                mean_val = round(desc.loc[col, "mean"], 2)
                min_val = round(desc.loc[col, "min"], 2)
                max_val = round(desc.loc[col, "max"], 2)
                insights.append(
                    f"Column **{col}** → mean = {mean_val}, range = [{min_val}, {max_val}]."
                )

        # Correlation
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
            strongest_corr = [
                (i, j, v) for i, j, v in zip(
                    corr_pairs.index.get_level_values(0),
                    corr_pairs.index.get_level_values(1),
                    corr_pairs.values
                ) if i != j
                ]
            if strongest_corr:
                col1, col2, val = strongest_corr[0]
                insights.append(
                    f"Strongest correlation is between **{col1}** and **{col2}** (corr={val:.2f})."
                )

        # Categorical insights
        if len(cat_cols) > 0:
            for col in cat_cols:
                most_common = df[col].mode()[0]
                insights.append(
                    f"In column **{col}**, the most common category is **{most_common}**."
                )

                top5 = df[col].value_counts().head(5)
                top5_list = ", ".join([f"{i} ({v})" for i, v in top5.items()])
                insights.append(f"Top 5 categories in **{col}**: {top5_list}")

        # Show insights in dashboard
        for i in insights:
            st.markdown(f"- {i}")
            # ======================
    # Export Report
    # ======================
    st.subheader(" Export Report")

    # Ensure /report folder exists
    os.makedirs("report", exist_ok=True)

    if st.button("Generate PDF Report"):
        pdf_path = "report/data_report.pdf"
        doc = SimpleDocTemplate(pdf_path)
        styles = getSampleStyleSheet()
        story = []

        # --- Title ---
        story.append(Paragraph("AI Data Story Teller - Executive Summary Report", styles['Title']))
        story.append(Spacer(1, 12))

        # --- Dataset Info ---
        story.append(Paragraph(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}", styles['Normal']))
        story.append(Spacer(1, 12))

        # --- Automated Insights ---
        story.append(Paragraph("Automated Insights:", styles['Heading2']))
        for i in insights:
            story.append(Paragraph(i, styles['Normal']))
            story.append(Spacer(1, 6))

        # --- Add Visualizations ---
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Histogram (first numeric column)
        if len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=(5,3))
            sns.histplot(df[numeric_cols[0]], bins=30, kde=True, ax=ax, color="skyblue")
            plot_path = "report/hist_plot.png"
            fig.savefig(plot_path)
            plt.close(fig)
            story.append(Paragraph("Histogram:", styles['Heading2']))
            story.append(Image(plot_path, width=400, height=250))
            story.append(Spacer(1, 12))

        # Correlation Heatmap
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            plot_path = "report/heatmap.png"
            fig.savefig(plot_path)
            plt.close(fig)
            story.append(Paragraph("Correlation Heatmap:", styles['Heading2']))
            story.append(Image(plot_path, width=400, height=250))
            story.append(Spacer(1, 12))

        # Boxplot (first numeric vs first category)
        if len(numeric_cols) > 0 and len(cat_cols) > 0:
            fig, ax = plt.subplots(figsize=(5,3))
            sns.boxplot(x=df[cat_cols[0]], y=df[numeric_cols[0]], ax=ax)
            plt.xticks(rotation=30, fontsize=7)
            plot_path = "report/boxplot.png"
            fig.savefig(plot_path)
            plt.close(fig)
            story.append(Paragraph("Boxplot:", styles['Heading2']))
            story.append(Image(plot_path, width=400, height=250))
            story.append(Spacer(1, 12))

        # --- Build PDF ---
        doc.build(story)

        # --- Download Button ---
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download Report",
                data=f,
                file_name="data_report.pdf",
                mime="application/pdf"
            )

        st.success("Report generated successfully!")
