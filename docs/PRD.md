#### 4.3 Python Scripts

##### 4.3.1 preprocess_data.py
**Purpose**: Clean and preprocess the raw Titanic dataset for model training and visualization

**Functionality**:
- Load raw data from `data/titanic_raw.csv` (Kaggle Titanic competition dataset, already included in repository)
- Apply all preprocessing steps from the notebook:
  - Drop unnecessary columns (name, ticket, cabin)
  - Handle missing values with group-based imputation
  - Remove outliers (3 standard deviations)
  - Create engineered features (child, one-hot encoding)
  - Scale features appropriately
- Save processed data to `data/titanic_processed.parquet`
- Should be runnable independently: `python scripts/preprocess_data.py`

**Implementation notes**:
- Extract preprocessing logic from notebook but refactor into clean functions
- Add proper error handling and logging
- Use functions with clear single responsibilities
- Include docstrings following PEP 257

##### 4.3.2 train_model.py
**Purpose**: Train the Random Forest model using the processed dataset and save it with preprocessing pipeline

**Functionality**:
- Load processed data from `data/titanic_processed.parquet`
- Create preprocessing pipeline including:
  - All fitted scalers (MinMaxScaler for sibsp/parch, StandardScaler for fare, custom for age)
  - Feature engineering transformations
  - One-hot encoding configuration
- Train Random Forest with specified hyperparameters:
  - n_estimators=1000
  - max_depth=9
  - min_samples_leaf=1
  - min_samples_split=2
  - max_features=0.2
  - max_samples=0.2
  - bootstrap=True
  - random_state=42
- Create sklearn Pipeline combining preprocessing and Random Forest
- Evaluate model performance and print metrics
- Save complete pipeline to `model/titanic_model_pipeline.joblib`
- Should be runnable independently: `python scripts/train_model.py`

**Implementation notes**:
- Use sklearn.pipeline.Pipeline to combine preprocessing and model
- Include all scalers and transformers in the pipeline
- Must store feature importances, as it is going to be used later in app.py. It may be accesed as follows: 
   ```# In app.py
   rf_model = pipeline.named_steps['classifier']
   feature_importance = rf_model.feature_importances_```
- This ensures the model file contains everything needed for prediction# Product Requirements Document
## Survival Probability Simulator

### 1. Executive Summary

The Survival Probability Simulator is a data science application that leverages machine learning to predict passenger survival probability based on the Titanic dataset. The project combines exploratory data analysis with predictive modeling, delivering insights through an interactive web application built with Streamlit.

### 2. Product Overview

#### 2.1 Purpose
Create an educational and analytical tool that demonstrates machine learning capabilities through historical data analysis, allowing users to:
- Explore survival patterns from the Titanic disaster through comprehensive visualizations
- Predict survival probability based on passenger characteristics using a trained Random Forest model
- Understand the key factors that influenced survival outcomes

#### 2.2 Target Audience
- Data science students and educators
- Machine learning practitioners
- History enthusiasts interested in data-driven insights
- General users curious about predictive modeling

### 3. Technical Architecture

#### 3.1 Technology Stack
- **Programming Language**: Python 3.8+
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization library**: Use Plotly exclusively for all charts (interactive by default)
- **Web Framework**: Streamlit
- **Model Serialization**: Joblib
- **Data Storage**: Parquet format for processed data

#### 3.2 Repository Structure
```
Survival-Probability-Simulator/
├── app.py
├── CONTRIBUTING.rst
├── data/
│   ├── titanic_raw.csv
│   └── titanic_processed.parquet
├── Dockerfile
├── docs/
│   └── PRD.md
├── LICENSE
├── MANIFEST.in
├── media/
├── model/
│   └── titanic_model_pipeline.joblib
├── notebooks/
│   └── titanic_eda_and_modeling.ipynb
├── README.md
├── requirements.txt
├── scripts/
│   ├── preprocess_data.py
│   └── train_model.py
└── SECURITY.md
```

### 4. Functional Requirements

#### 4.1 Jupyter Notebook Component

**Note**: The existing notebook (titanic_eda_and_modeling.ipynb) should be kept as-is and not modified. It serves as the reference implementation for the preprocessing and model training scripts.

##### 4.1.1 Data Preprocessing Module
- **Input**: Raw Titanic dataset (titanic_raw.csv)
- **Processing Requirements**:
  - Drop unnecessary columns (name, ticket, cabin)
  - Handle missing values:
    - Cabin: Drop column (77% missing)
    - Embarked: Drop rows with missing values (only 2 rows)
    - Age: Group-based median imputation using pclass, sibsp_bin (0 vs >0), and parch_bin (0, 1_or_2, 3_plus)
    - Fare: Median imputation
  - Remove outliers outside 3 standard deviations for age, fare, sibsp, and parch
  - Feature engineering:
    - Create 'child' feature (age ≤ 15)
    - One-hot encode sex, embarked, and pclass
    - Drop redundant sex_female column, keep only male
  - Feature scaling:
    - Age: Custom scaling (0-122 range)
    - Sibsp and Parch: MinMaxScaler
    - Fare: StandardScaler
- **Output**: Cleaned dataset saved as titanic_processed.parquet

##### 4.1.2 Exploratory Data Analysis (EDA)
Generate exactly 15 visualizations for the Streamlit application's second tab:

1. **Age Histogram**: Age distribution with bins showing passenger count
   - *Insight: Peak between ages 20-30 reveals most passengers were young adults, with relatively few children and elderly*

2. **Pie Chart of Classes**: Proportions of 1st, 2nd, and 3rd class passengers
   - *Insight: Over half (55%) traveled in 3rd class, highlighting the socioeconomic divide on board*

3. **Pie Chart of Gender Distribution**: Male vs. Female passenger proportions
   - *Insight: 65% male passengers vs 35% female, significant given "women and children first" protocol*

4. **Bar Chart of Passengers by Embarkation Port**: Passenger counts from Southampton, Cherbourg, and Queenstown
   - *Insight: Southampton dominated with 72% of passengers, while Cherbourg's wealthier passengers may explain higher survival rates*

5. **Fare Histogram**: Ticket price distribution with fare ranges
   - *Insight: Strongly skewed toward low fares under £50, with few paying luxury prices up to £512*

6. **Pie Chart of Outcome**: Overall survival vs. non-survival rates
   - *Insight: Only 38% survived, emphasizing the disaster's severity with nearly two-thirds perishing*

7. **Feature Importance Bar Chart**: Random Forest derived importance scores showing feature contribution to survival prediction.For this plot, the information should be extracted from pipeline in app.py
    ```# In app.py
    rf_model = pipeline.named_steps['classifier']
    feature_importance = rf_model.feature_importances_```
   - *Insight: Fare, sex, and age emerge as strongest survival predictors, confirming wealth and gender as key factors*

8. **Grouped Bar Chart by Class and Gender**: Survival rates by class and gender combination
   - *Insight: Women had higher survival rates across all classes, with 3rd class men facing the worst odds*

9. **Bar Chart of Survival Rate by Embarkation Port**: Survival percentage by port of embarkation
   - *Insight: Cherbourg's 55% survival rate suggests wealthier 1st-class passengers boarded there*

10. **Age Histograms by Survival**: Two overlapping histograms in one plot comparing age distribution of survivors vs. non-survivors
    - *Insight: Young children under 10 show distinct survival advantage, while adult age distributions are similar*

11. **Bar Chart of Survival by Family Size**: Survival percentage by total number of family members aboard (sibsp + parch)
    - *Insight: Optimal survival for families of 3-4 members; solo travelers and large families fared worse*

12. **Boxplot of Fare vs. Survival**: Fare distribution comparison between survivors and non-survivors showing median, IQR, and outliers
    - *Insight: Survivors' median fare significantly higher, confirming wealth as survival factor*

13. **Filled Contour Plot of Age vs. Fare Survival Probability**: Survival probability zones across age-fare space with filled color regions
    - *Insight: Two survival paths emerge - being wealthy (high fare) or being very young (low age)*

14. **Correlation Heatmap**: Pearson correlations between all numerical and encoded categorical variables
    - *Insight: Strong negative correlation between male gender and survival (-0.54) confirms gender bias in rescue*

15. **Area Chart by Age and Class**: Age-based survival rates split by passenger class with stacked or semi-transparent areas
    - *Insight: Young passengers survived better across all classes, but 1st class maintained advantage at all ages*

##### 4.1.3 Model Training and Selection
- **Algorithm**: Random Forest Classifier only
- **Hyperparameter optimization using GridSearchCV**:
  - Base parameters: n_estimators=1000, random_state=42, bootstrap=True, max_features=0.2, max_samples=0.2
  - Grid search parameters: max_depth=[7, 8, 9], min_samples_split=[2, 3], min_samples_leaf=[1, 2]
  - Optimal parameters found: max_depth=9, min_samples_leaf=1, min_samples_split=2
  - Cross-validation: 3-fold CV optimizing for accuracy
- **Expected performance**: ~83% accuracy on cross-validation
- **Feature importance**: Extract and visualize feature importance scores
- **Output**: Serialized model saved as random_forest_model.joblib

#### 4.2 Streamlit Application (app.py)

##### 4.2.1 Application Structure
- Two-tab interface using Streamlit's tab component
- Load trained model and processed dataset on initialization
- Responsive design compatible with desktop and mobile browsers

##### 4.2.2 Tab 1: Interactive Survival Prediction

**Input Components**:
- **Age**: Slider widget (0.42-80 years, based on dataset min/max)
- **Sex**: Radio button or selectbox (Male/Female)
- **Passenger Class**: Selectbox (1st, 2nd, 3rd)
- **Embarkation Port**: Selectbox (Southampton, Cherbourg, Queenstown)
- **Number of Siblings/Spouses**: Number input (0-8, based on dataset max)
- **Number of Parents/Children**: Number input (0-6, based on dataset max)
- **Fare**: Number input or slider (0-512.33, based on dataset max)

**Processing Pipeline**:
- User inputs are in intuitive formats (actual age, actual fare, etc.)
- App internally transforms these inputs to match model expectations:
  - Age: scales to 0-1 range using 0-122 bounds
  - Fare: applies StandardScaler transformation
  - Sibsp/Parch: applies MinMaxScaler transformation
  - Creates derived features (child flag for age ≤ 15)
  - One-hot encodes categorical variables
- Transformation logic uses the pipeline saved in `model/titanic_model_pipeline.joblib`

**Functionality**:
- "Predict Survival" button to trigger inference
- Display survival probability as percentage (0-100%)
- Visual feedback:
  - If probability > 50%: Trigger st.balloons() celebration
  - If probability ≤ 50%: No animation
- Real-time updates when input values change
- Clear result display with appropriate formatting

##### 4.2.3 Tab 2: EDA Gallery

**Visualization Requirements**:
- Display all 15 charts from the specifications
- **Mandatory interactivity**: 
  - Hover tooltips showing exact values
  - Interactive legends where applicable
  - Zoom and pan capabilities for detailed exploration
- **Performance optimization**:
  - Use st.cache_data() decorator for data loading functions
  - Cache computed visualizations and aggregations
  - Load dataset once and reuse for all charts
  - Set TTL (time-to-live) of 1 hour for cached data
  - Cache the model pipeline loading with st.cache_resource()
- **Layout**: 
  - Organized grid or sequential display
  - Consistent color scheme across all charts
  - Responsive sizing for different screen dimensions

**Optional Enhancement**:
- Brief insight text for each visualization using one of:
  - Expandable info buttons
  - Subtle captions below charts
  - Hover-triggered insight tooltips

### 5. Non-Functional Requirements

#### 5.1 Performance
- Model inference response time < 500ms
- Chart rendering time < 2 seconds per visualization
- Application startup time < 10 seconds

#### 5.2 Usability
- Intuitive interface requiring no technical expertise
- Clear labeling of all input fields
- Informative error messages for invalid inputs
- Mobile-responsive design

#### 5.3 Reliability
- Graceful handling of edge cases (e.g., extreme input values)
- Model prediction stability across all valid input combinations
- Consistent visualization rendering across browsers

#### 5.4 Maintainability
- Modular code structure with clear separation of concerns
- Comprehensive code comments and docstrings
- Version control using Git
- Clear README with setup instructions

### 6. Data Specifications

#### 6.1 Input Data Schema
Raw Titanic dataset containing:
- PassengerId
- Survived (target variable)
- Pclass (passenger class)
- Name
- Sex
- Age
- SibSp (siblings/spouses)
- Parch (parents/children)
- Ticket
- Fare
- Cabin
- Embarked

#### 6.2 Processed Data Requirements
- No missing values in critical features
- Normalized numerical features
- Encoded categorical variables
- Engineered features (e.g., Title extraction, Family size, IsChild)

### 7. Model Specifications

#### 7.1 Random Forest Configuration
- n_estimators: 1000
- max_depth: 9
- min_samples_leaf: 1
- min_samples_split: 2
- max_features: 0.2
- max_samples: 0.2
- bootstrap: True
- random_state: 42
- Feature importance extraction capability
- Probability prediction support (not just classification)

#### 7.2 Model Performance Targets
- Minimum accuracy: 80%
- Cross-validation score variation < 5%
- Consistent predictions across similar input combinations

### 8. Dependencies and Environment

#### 8.1 Required Libraries (requirements.txt)
```
streamlit>=1.25.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
plotly>=5.15.0
joblib>=1.3.0
```

#### 8.2 Python Version
- Python 3.12 required
- Virtual environment recommended for dependency management

#### 8.3 Docker Configuration
**Dockerfile Requirements**:
- Base image: python:3.12-slim
- Working directory: /app
- Copy all project files to container
- Install dependencies from requirements.txt
- Expose port 8501 (Streamlit default)
- Set command to run Streamlit app: `streamlit run app.py`

**Example Dockerfile structure**:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Docker build and run commands**:
- Build: `docker build -t titanic-survival-simulator .`
- Run: `docker run -p 8501:8501 titanic-survival-simulator`

### 9. Documentation and Coding Standards

#### 9.1 Code Style Requirements
- **PEP 8 Compliance** (Python Style Guide):
  - Maximum line length: 79 characters
  - Use 4 spaces per indentation level (no tabs)
  - Two blank lines between top-level function and class definitions
  - One blank line between method definitions inside a class
  - Imports at the top of files, grouped in order: standard library, third-party, local
  - Use lowercase with underscores for function and variable names (snake_case)
  - Use CapWords for class names
  - Constants in UPPER_CASE_WITH_UNDERSCORES
  
- **PEP 257 Compliance** (Docstring Conventions):
  - One-line docstrings for simple functions: """Return a foobang."""
  - Multi-line docstrings for complex functions with:
    - Summary line
    - Blank line
    - Detailed description if needed
    - Parameters and return types
  - Docstrings immediately after function/class/module definition
  - Triple double quotes for all docstrings

#### 9.2 Implementation Guidelines
- **Simplicity**: Write minimum code required for functionality
- **Readability**: Code should be self-documenting; prefer clear variable names over comments
- **Comments**: Use sparingly, only when code intent is not obvious
  - Inline comments sparingly and separated by at least two spaces from code
  - Block comments for complex algorithms only
  - No redundant comments (avoid: i += 1  # increment i)
- **No additional files**: Strictly follow the repository structure; do not create extra files
- **No testing files**: Testing is not required for this project

#### 9.3 README.md
**Required sections in order**:
1. **Project Overview**: Brief description of the Survival Probability Simulator
2. **Features**: Two-tab interface description (prediction and EDA gallery)
3. **Installation**: Step-by-step setup instructions for local development
4. **Docker Guide**: 
   - Building the Docker image
   - Running the container
   - Accessing the application
5. **Usage**: How to use both tabs with screenshots
6. **Scripts**: 
   - `preprocess_data.py`: Cleans raw Titanic dataset and saves processed version
   - `train_model.py`: Trains Random Forest model with preprocessing pipeline
   - Note: "These scripts are provided for reference. The processed data and trained model are already included in the repository."
   - Note: "The raw dataset (titanic_raw.csv) is the Kaggle Titanic competition dataset, already included in the repository."
   - Execution order if needed: First run preprocess_data.py, then train_model.py
7. **Project Structure**: Complete directory tree with descriptions
8. **Data Dictionary**: Table defining each feature in the processed dataset
9. **Model Performance**: Accuracy metrics and feature importance summary
10. **Contributing**: Guidelines referencing CONTRIBUTING.rst
11. **License**: "This project is licensed under the **MIT License** — you are free to use, modify, and distribute it, with attribution. See the LICENSE file for details."

#### 9.4 Code Documentation
- Docstrings required for:
  - All modules (at file top)
  - All functions and methods
  - All classes
- Comments only when necessary for complex logic
- Self-documenting code through clear naming conventions
- Type hints optional but recommended for function signatures

#### 9.5 Notebook Documentation
- Markdown cells explaining each analysis step
- Rationale for preprocessing decisions
- Model selection justification
- Feature importance interpretation

### 10. Success Criteria

The project will be considered successful when:
1. All 15 specified visualizations are implemented and interactive
2. Random Forest model achieves >80% accuracy
3. Streamlit application runs without errors
4. Prediction interface provides real-time results
5. Visual feedback (balloons) triggers correctly based on threshold
6. All charts in EDA gallery have hover interactivity
7. Code is modular, documented, and maintainable
8. Repository follows specified structure exactly

### 11. Constraints and Assumptions

#### 11.1 Constraints
- Single developer implementation
- Limited to Titanic dataset
- Must use Random Forest as final model with specified hyperparameters
- Exactly 15 visualizations required
- 50% threshold for balloon animation

#### 11.2 Assumptions
- User has basic understanding of web interfaces
- Modern browser with JavaScript enabled
- Sufficient local computational resources
- Access to required Python libraries

### 12. Risk Assessment

| Risk | Impact | Mitigation |
|### 13. Data Dictionary

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| survived | float | 0, 1 | Target variable (0=died, 1=survived) |
| age_scaled | float | 0-1 | Age normalized to 0-122 range |
| male | float | 0, 1 | Binary gender indicator (1=male, 0=female) |
| child | float | 0, 1 | Binary indicator for age ≤ 15 |
| fare_scaled | float | ~-2 to ~10 | Standardized ticket fare |
| class_First | float | 0, 1 | One-hot encoded 1st class |
| class_Second | float | 0, 1 | One-hot encoded 2nd class |
| class_Third | float | 0, 1 | One-hot encoded 3rd class |
| sibsp_scaled | float | 0-1 | MinMax scaled siblings/spouses count |
| parch_scaled | float | 0-1 | MinMax scaled parents/children count |
| embark_town_Cherbourg | float | 0, 1 | One-hot encoded Cherbourg embarkation |
| embark_town_Queenstown | float | 0, 1 | One-hot encoded Queenstown embarkation |
| embark_town_Southampton | float | 0, 1 | One-hot encoded Southampton embarkation |

------|--------|------------|
| Model overfitting | High | Cross-validation and regularization |
| Large dataset processing time | Medium | Data caching and efficient algorithms |
| Browser compatibility issues | Low | Testing on multiple browsers |
| Dependency conflicts | Medium | Virtual environment and version pinning |

---

**Document Version**: 1.0  
**Last Updated**: [Current Date]  
**Status**: Ready for Implementation
