**Overview**

This Streamlit application performs pairwise agreement analysis to measure consistency between pairs of AI models (ChatGPT-Gemini, ChatGPT-Mistral AI, Gemini-Mistral AI) on plant classification tasks. The application provides detailed model-to-model comparison and reveals patterns of systematic agreement or disagreement.

Features
Pairwise Analysis: Calculates agreement rates between all model pairs

Multiple Visualization Types

1. Pairwise agreement heatmap

2. Model performance comparison chart

3. Plant-level agreement patterns

**Detailed comparison matrix**

- Flexible Data Input
- Pre-loaded research data
- User-provided classification data

**Advanced Analytics**

1. Best/worst performing model pairs

2. Overall agreement metrics

3. Statistical interpretation

***Dependencies**
# Core dependencies
pip install streamlit
pip install pandas
pip install numpy

# Visualization dependencies
pip install matplotlib
pip install seaborn

# Statistical analysis
pip install scipy

# Enhanced colormaps
pip install matplotlib-colors

**How to run the app local**
1. step 1- install dependencies
pip install -r requirements_pairwise.txt

2. step 2- Run the application
streamlit run pairwise_agreement_app.py


3. Access the application
Open browser to http://localhost:8501
Application loads with study data analysis by default


**Usage instructions**

1. Study Data Analysis
Default view displays research data with 20 indigenous plants

Click "Run Pairwise Agreement Analysis on Study Data"

View comprehensive results
- Pairwise agreement heatmap (color-coded matrix)
- Detailed comparison table with counts and percentages
- Model performance bar chart
- Plant-level agreement patterns

2. Custom Data Analysis
- Select "Insert Your Own Data" in navigation
- Input requirements
- Minimum 3 plants (fewer required than Fleiss Kappa)
- Unique plant names only
- Complete classifications for all three models
- Run analysis to get personalized pairwise results

Output Interpretation
Agreement Rate Scale
≥ 0.80: Excellent Agreement (Models are very consistent)

0.60 - 0.79: Good Agreement (Models are generally consistent)

0.40 - 0.59: Moderate Agreement (Some inconsistency present)

0.20 - 0.39: Fair Agreement (Significant differences)

< 0.20: Poor Agreement (Models frequently disagree)

Key Outputs
Pairwise Matrix: Agreement rates between each model pair
Comparison Table: Detailed counts and interpretations
Performance Chart: Average agreement per model
Plant Heatmap: Agreement patterns at individual plant level

Formula
Agreement Rate = Number of Matching Classifications / Number of Valid Comparisons
- Only counts instances where both models provided valid responses (excluding "No Results")
- Provides conservative estimate of true agreement


Overall Agreement
- Measures consensus across all models
- Only considers plants with at least 2 valid responses
- Counts instances where all valid responses are identical


   
