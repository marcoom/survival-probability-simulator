"""
Survival Probability Simulator - Streamlit Application

A data science application that leverages machine learning to predict
passenger survival probability based on the Titanic dataset with comprehensive
exploratory data analysis visualizations.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import the FeaturePassthrough class from the training script
sys.path.append('scripts')
from train_model import FeaturePassthrough


@st.cache_resource(ttl=3600)
def load_model():
    """
    Load the trained model pipeline with caching.
    
    Returns:
        sklearn.Pipeline: Trained model pipeline
    """
    try:
        model_path = Path("model/titanic_model_pipeline.joblib")
        if not model_path.exists():
            error_msg = f"Model file not found: {model_path}"
            try:
                st.error(error_msg)
            except:
                print(error_msg)
            return None
        
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        error_msg = f"Error loading model: {e}"
        try:
            st.error(error_msg)
        except:
            print(error_msg)
        return None


@st.cache_data(ttl=3600)
def load_data():
    """
    Load the processed dataset with caching.
    
    Returns:
        pd.DataFrame: Processed Titanic dataset
    """
    try:
        data_path = Path("data/titanic_processed.parquet")
        df = pd.read_parquet(data_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data(ttl=3600)
def load_raw_data():
    """
    Load the raw dataset for some visualizations.
    
    Returns:
        pd.DataFrame: Raw Titanic dataset
    """
    try:
        data_path = Path("data/titanic_raw.csv")
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Error loading raw data: {e}")
        return None


def prepare_user_input(age, sex, pclass, embarked, sibsp, parch, fare):
    """
    Transform user inputs to match model expectations.
    
    Args:
        age (float): Age in years
        sex (str): 'Male' or 'Female'
        pclass (int): Passenger class (1, 2, or 3)
        embarked (str): Embarkation port
        sibsp (int): Number of siblings/spouses
        parch (int): Number of parents/children
        fare (float): Ticket fare
        
    Returns:
        pd.DataFrame: Processed input ready for prediction
    """
    # Create child flag
    child = 1 if age <= 15 else 0
    
    # Gender encoding
    male = 1 if sex == 'Male' else 0
    
    # One-hot encode passenger class
    class_first = 1 if pclass == 1 else 0
    class_second = 1 if pclass == 2 else 0
    class_third = 1 if pclass == 3 else 0
    
    # One-hot encode embarkation port
    embark_cherbourg = 1 if embarked == 'Cherbourg' else 0
    embark_queenstown = 1 if embarked == 'Queenstown' else 0
    embark_southampton = 1 if embarked == 'Southampton' else 0
    
    # Scale features based on training data ranges
    # Age: normalize to 0-122 range
    age_scaled = age / 122.0
    
    # Fare: approximate StandardScaler (mean ~32, std~50 from training data)
    fare_scaled = (fare - 32) / 50
    
    # Sibsp and Parch: MinMaxScaler based on training data max values
    sibsp_scaled = sibsp / 8.0  # max sibsp in training was 8
    parch_scaled = parch / 6.0  # max parch in training was 6
    
    # Create input DataFrame with exact column order from training
    input_data = pd.DataFrame({
        'child': [child],
        'male': [male],
        'embark_town_Cherbourg': [embark_cherbourg],
        'embark_town_Queenstown': [embark_queenstown],
        'embark_town_Southampton': [embark_southampton],
        'class_First': [class_first],
        'class_Second': [class_second],
        'class_Third': [class_third],
        'age_scaled': [age_scaled],
        'sibsp_scaled': [sibsp_scaled],
        'parch_scaled': [parch_scaled],
        'fare_scaled': [fare_scaled]
    })
    
    return input_data


def create_survival_prediction_tab():
    """Create the interactive survival prediction interface."""
    st.write("Adjust the parameters below to predict survival probability:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 1, 80, 30, 1)
        sex = st.radio("Gender", ["Male", "Female"])
        pclass = st.selectbox("Passenger Class", [1, 2, 3], 
                             format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class")
        embarked = st.selectbox("Embarkation Port", 
                               ["Southampton", "Cherbourg", "Queenstown"])
    
    with col2:
        sibsp = st.number_input("Number of Siblings/Spouses", 0, 8, 0)
        parch = st.number_input("Number of Parents/Children", 0, 6, 0)
        fare = st.number_input("Ticket Fare (£)", 0.0, 512.3, 30.0, 5.0)
    
    if st.button("Predict Survival", type="primary"):
        pipeline = load_model()
        if pipeline is not None:
            try:
                # Prepare input data
                input_data = prepare_user_input(age, sex, pclass, embarked, 
                                              sibsp, parch, fare)
                
                # Make prediction
                survival_prob = pipeline.predict_proba(input_data)[0][1]
                survival_percentage = survival_prob * 100
                
                # Display results
                st.subheader("Prediction Results")
                
                if survival_percentage > 50:
                    st.success(f"🎉 **Survival Probability: {survival_percentage:.1f}%**")
                    st.balloons()
                else:
                    st.error(f"💔 **Survival Probability: {survival_percentage:.1f}%**")
                
                # Additional context
                if survival_percentage > 70:
                    st.info("High survival probability - favorable conditions!")
                elif survival_percentage > 50:
                    st.info("Moderate survival probability.")
                elif survival_percentage > 30:
                    st.warning("Low survival probability - unfavorable conditions.")
                else:
                    st.warning("Very low survival probability - high risk scenario.")
                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")


def create_age_histogram(df_raw):
    """Create age distribution histogram."""
    fig = px.histogram(df_raw, x='Age', nbins=50, opacity=1,
                      title="Age Distribution of Passengers",
                      labels={'Age': 'Age (years)', 'count': 'Number of Passengers'})
    fig.update_layout(showlegend=False, bargap=0.02)
    return fig


def create_class_pie_chart(df_raw):
    """Create passenger class distribution pie chart."""
    class_counts = df_raw['Pclass'].value_counts().sort_index()
    fig = px.pie(values=class_counts.values, 
                names=[f"{i}{'st' if i==1 else 'nd' if i==2 else 'rd'} Class" 
                      for i in class_counts.index],
                title="Passenger Class Distribution")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False)
    return fig

def create_gender_pie_chart(df_raw):
    """Create gender distribution pie chart."""
    gender_counts = df_raw['Sex'].value_counts()
    fig = px.pie(values=gender_counts.values, names=gender_counts.index.str.title(),
                title="Gender Distribution")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False)
    return fig

def create_embarkation_bar_chart(df_raw):
    """Create passengers by embarkation port bar chart."""
    embark_names = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}

    ser = df_raw['Embarked'].fillna('Unknown')
    counts = ser.value_counts().rename_axis('code').reset_index(name='count')
    counts['port'] = counts['code'].map(embark_names).fillna(counts['code'])
    counts = counts.sort_values('count', ascending=False)

    total = counts['count'].sum()
    counts['pct'] = counts['count'] / total * 100
    counts['text'] = counts.apply(lambda r: f"{r['count']} ({r['pct']:.1f}%)", axis=1)

    fig = px.bar(
        counts,
        x='port',
        y='count',
        title=f"Passengers by Embarkation Port",
        labels={'port': 'Embarkation Port', 'count': 'Number of Passengers'}
    )
    fig.update_layout(
        yaxis=dict(rangemode='tozero', tickformat=',d'),
        bargap=0.25
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Passengers: %{y:,}<br>Share: %{customdata:.1f}%<extra></extra>",
        customdata=counts['pct']
    )

    return fig

def create_fare_histogram(df_raw):
    """Create fare distribution histogram."""
    fig = px.histogram(df_raw, x='Fare', nbins=50,
                      title="Ticket Fare Distribution",
                      labels={'Fare': 'Fare (£)', 'count': 'Number of Passengers'})
    fig.update_layout(showlegend=False, bargap=0.02)
    return fig


def create_outcome_pie_chart(df):
    """Create overall survival outcome pie chart."""
    survival_counts = df['survived'].value_counts()
    fig = px.pie(values=survival_counts.values, 
                names=['Did Not Survive', 'Survived'],
                title="Overall Survival Outcome")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False)
    return fig


def create_feature_importance_chart():
    """Create feature importance bar chart from model."""
    pipeline = load_model()
    df = load_data()
    
    if pipeline is not None and df is not None:
        rf_model = pipeline.named_steps['classifier']
        feature_names = df.drop(['survived'], axis=1).columns
        importance_scores = rf_model.feature_importances_
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', 
                    orientation='h',
                    title="Feature Importance in Survival Prediction")
        return fig
    return None


def create_class_gender_survival_chart(df_raw):
    """Create grouped bar chart by class and gender."""
    # Calculate survival rates by class and gender
    survival_by_class_gender = df_raw.groupby(['Pclass', 'Sex'])['Survived'].agg(['count', 'sum']).reset_index()
    survival_by_class_gender['survival_rate'] = survival_by_class_gender['sum'] / survival_by_class_gender['count'] * 100
    
    fig = px.bar(survival_by_class_gender, x='Pclass', y='survival_rate', 
                color='Sex', barmode='group',
                title="Survival Rate by Class and Gender",
                labels={'Pclass': 'Passenger Class', 'survival_rate': 'Survival Rate (%)', 'Sex': 'Gender'})
    return fig


def create_embarkation_survival_chart(df_raw):
    """Create survival rate by embarkation port bar chart."""
    embark_survival = df_raw.groupby('Embarked')['Survived'].agg(['count', 'sum']).reset_index()
    embark_survival['survival_rate'] = embark_survival['sum'] / embark_survival['count'] * 100
    
    embark_names = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
    embark_survival['Port'] = embark_survival['Embarked'].map(embark_names)
    
    fig = px.bar(embark_survival, x='Port', y='survival_rate',
                title="Survival Rate by Embarkation Port",
                labels={'survival_rate': 'Survival Rate (%)'})
    return fig


def create_age_survival_histograms(df_raw):
    """Create overlapping age histograms by survival."""
    fig = go.Figure()
    
    # Survivors
    survivors = df_raw[df_raw['Survived'] == 1]['Age']
    fig.add_trace(go.Histogram(x=survivors, name='Survived', opacity=0.7, nbinsx=10))
    
    # Non-survivors
    non_survivors = df_raw[df_raw['Survived'] == 0]['Age']
    fig.add_trace(go.Histogram(x=non_survivors, name='Did Not Survive', opacity=0.5, nbinsx=10))
    
    fig.update_layout(
        title="Age Distribution by Survival Outcome",
        xaxis_title="Age (years)",
        yaxis_title="Number of Passengers",
        barmode='overlay',
        bargap=0.02
    )
    return fig


def create_family_size_survival_chart(df_raw):
    """Create survival by family size bar chart."""
    df_raw = df_raw.copy()
    df_raw['family_size'] = df_raw['SibSp'] + df_raw['Parch']
    family_survival = df_raw.groupby('family_size')['Survived'].agg(['count', 'sum']).reset_index()
    family_survival['survival_rate'] = family_survival['sum'] / family_survival['count'] * 100
    
    fig = px.bar(family_survival, x='family_size', y='survival_rate',
                title="Survival Rate by Family Size",
                labels={'family_size': 'Total Family Members Aboard', 'survival_rate': 'Survival Rate (%)'})
    return fig


def create_fare_survival_boxplot(df_raw):
    """Create boxplot of fare vs survival."""
    df_raw = df_raw.copy()
    df_raw['Survival_Status'] = df_raw['Survived'].map({0: 'Did Not Survive', 1: 'Survived'})
    
    fig = px.box(df_raw, x='Survival_Status', y='Fare',
                title="Fare Distribution by Survival Outcome",
                labels={'Survival_Status': 'Survival Status', 'Fare': 'Fare (£)'})
    
    fig.update_layout(yaxis_range=[0, 150])
    return fig





def create_age_fare_filled_contour(df_raw):
    """Alternative: Filled contour plot with color regions."""
    df_raw = df_raw.copy()
    df_clean = df_raw.dropna(subset=['Age', 'Fare'])
    
    # Same grid setup as main version
    age_min, age_max = df_clean['Age'].min(), df_clean['Age'].max()
    fare_min, fare_max = 0, df_clean['Fare'].quantile(0.95)
    age_grid = np.linspace(age_min, age_max, 40)
    fare_grid = np.linspace(fare_min, fare_max, 40)
    survival_prob_grid = np.zeros((len(fare_grid), len(age_grid)))
    
    # Calculate probabilities
    for i in range(len(fare_grid)):
        for j in range(len(age_grid)):
            age_diff = df_clean['Age'] - age_grid[j]
            fare_diff = df_clean['Fare'] - fare_grid[i]
            age_weight = np.exp(-(age_diff**2) / (2 * 8**2))
            fare_weight = np.exp(-(fare_diff**2) / (2 * 15**2))
            combined_weight = age_weight * fare_weight
            significant_mask = combined_weight > 0.01
            if significant_mask.sum() >= 10:
                weights = combined_weight[significant_mask]
                survival_values = df_clean.loc[significant_mask, 'Survived']
                survival_prob_grid[i, j] = np.average(survival_values, weights=weights)
            else:
                survival_prob_grid[i, j] = np.nan
    
    # Create filled contour
    fig = go.Figure(data=go.Contour(
        x=age_grid, y=fare_grid, z=survival_prob_grid,
        colorscale='blues_r',
        contours=dict(
            coloring='fill',  # Fill areas between contours
            showlines=True,
            start=0, end=1, size=0.1
        ),
        colorbar=dict(title="Survival<br>Probability"),
        hovertemplate="Survival Probability: %{z:.0%}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Filled Contour: Survival Probability Zones",
        xaxis_title="Age (years)",
        yaxis_title="Fare (£)"
    )
    return fig




def create_correlation_heatmap(df):
    """Create correlation heatmap of all variables."""
    # Select numerical columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(correlation_matrix, 
                   title="Correlation Heatmap of All Variables",
                   aspect='auto',
                   color_continuous_scale='blues_r')
    return fig


def create_age_class_area_chart(df_raw):
    """Create area chart by age and class."""
    df_raw = df_raw.copy()
    # Create age bins
    df_raw['age_bin'] = pd.cut(df_raw['Age'], bins=8, precision=0)
    
    # Calculate survival rates by age bin and class
    age_class_survival = df_raw.groupby(['age_bin', 'Pclass'], observed=True)['Survived'].agg(['count', 'sum']).reset_index()
    age_class_survival = age_class_survival[age_class_survival['count'] > 0]  # Remove empty bins
    age_class_survival['survival_rate'] = age_class_survival['sum'] / age_class_survival['count'] * 100
    age_class_survival['age_mid'] = age_class_survival['age_bin'].apply(lambda x: x.mid)
    
    fig = px.area(age_class_survival, x='age_mid', y='survival_rate', 
                 color='Pclass',
                 title="Survival Rate by Age and Class",
                 labels={'age_mid': 'Age', 'survival_rate': 'Survival Rate (%)', 'Pclass': 'Class'})
    return fig


def create_eda_gallery_tab():
    """Create the EDA gallery with all 15 visualizations."""
    st.write("Interactive visualizations exploring survival patterns in the Titanic dataset:")
    
    df = load_data()
    df_raw = load_raw_data()
    
    if df is None or df_raw is None:
        st.error("Could not load dataset for visualizations.")
        return
    
    # Clean raw data for visualizations
    df_raw_clean = df_raw.dropna(subset=['Age', 'Embarked'])
    
    # Row 1: Basic demographics
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_age_histogram(df_raw_clean), use_container_width=True)
        st.caption("*Peak between ages 20-30 reveals most passengers were young adults*")
    
    with col2:
        st.plotly_chart(create_class_pie_chart(df_raw), use_container_width=True)
        st.caption("*Over half (55%) traveled in 3rd class, highlighting socioeconomic divide*")
    
    # Row 2: Gender and embarkation
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_gender_pie_chart(df_raw), use_container_width=True)
        st.caption("*65% male vs 35% female, significant given 'women and children first' protocol*")
    
    with col2:
        st.plotly_chart(create_embarkation_bar_chart(df_raw_clean), use_container_width=True)
        st.caption("*Southampton dominated with 72% of passengers*")
    
    # Row 3: Fare and outcomes
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_fare_histogram(df_raw_clean), use_container_width=True)
        st.caption("*Strongly skewed toward low fares under £50*")
    
    with col2:
        st.plotly_chart(create_outcome_pie_chart(df), use_container_width=True)
        st.caption("*Only 38% survived, emphasizing the disaster's severity*")
    
    # Row 4: Feature importance and survival patterns
    col1, col2 = st.columns(2)
    with col1:
        feature_chart = create_feature_importance_chart()
        if feature_chart:
            st.plotly_chart(feature_chart, use_container_width=True)
            st.caption("*Fare, gender, and age emerge as strongest survival predictors*")
        else:
            st.error("Could not load feature importance chart")
    
    with col2:
        st.plotly_chart(create_class_gender_survival_chart(df_raw_clean), use_container_width=True)
        st.caption("*Women had higher survival rates across all classes*")
    
    # Row 5: Port survival and age patterns
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_embarkation_survival_chart(df_raw_clean), use_container_width=True)
        st.caption("*Cherbourg's 55% survival rate suggests wealthier passengers*")
    
    with col2:
        st.plotly_chart(create_age_survival_histograms(df_raw_clean), use_container_width=True)
        st.caption("*Young children under 10 show distinct survival advantage*")
    
    # Row 6: Family and fare analysis
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_family_size_survival_chart(df_raw_clean), use_container_width=True)
        st.caption("*Optimal survival for families of 3 members*")
    
    with col2:
        st.plotly_chart(create_fare_survival_boxplot(df_raw_clean), use_container_width=True)
        st.caption("*Survivors' median fare significantly higher*")
    
    # Row 7: Advanced analysis
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_age_fare_filled_contour(df_raw_clean))
        st.caption("*Two clear survival escape routes: being young or being wealthy*")
    
    with col2:
        st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)
        st.caption("*Strong negative correlation between male gender and survival*")
    
    # Row 8: Final analysis
    st.plotly_chart(create_age_class_area_chart(df_raw_clean), use_container_width=True)
    st.caption("*Young passengers survived better across all classes, but 1st class maintained advantage at all ages*")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Survival Probability Simulator",
        page_icon="🚢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚢 Survival Probability Simulator")
    st.markdown("""
    This application uses machine learning to predict passenger survival probability 
    and provides comprehensive exploratory data analysis of the Titanic dataset.
    """)
    
    # Create tabs
    tab1, tab2 = st.tabs(["🎯 Survival Predictor", "📊 Data Exploration"])
    
    with tab1:
        create_survival_prediction_tab()
    
    with tab2:
        create_eda_gallery_tab()
    
    # Sidebar information
    st.sidebar.markdown("## About This App")
    st.sidebar.info("""
    **Survival Probability Simulator**
    
    Uses a Random Forest model trained on the Titanic dataset to predict 
    survival probability based on passenger characteristics.
    
    **Model Performance:**
    - ~86% accuracy on test set
    - 1000 decision trees
    - Optimized hyperparameters via GridSearch
    
    **Features:**
    - Interactive prediction interface
    - Comprehensive visualizations
    - Real-time probability calculation
    """)


if __name__ == "__main__":
    main()