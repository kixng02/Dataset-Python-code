**Overview**
This Streamlit application performs Fleiss Kappa analysis to measure inter-rater agreement among multiple AI models (ChatGPT, Gemini, Mistral AI) on indigenous plant classification tasks. The application provides statistical analysis and visualizations to assess the consistency and reliability of AI-generated content concerning Indigenous Knowledge Systems.

**Features**
- Statistical Analysis: Calculates Fleiss' Kappa coefficient with p-values and confidence intervals

Multiple Data Sources

- Pre-loaded research study data (20 indigenous plants)

- User-provided custom classification data

Visualizations

- Agreement patterns heatmap

- Detailed classification heatmap

- Results bar charts

Data Management

- Input validation and duplicate prevention

- Session state management

- CSV export functionality


**Dependencies**
# Core dependencies
pip install streamlit
pip install pandas
pip install numpy

# Visualization and statistical dependencies
pip install matplotlib
pip install seaborn
pip install scipy

# Optional: For enhanced data handling
pip install openpyxl  # For Excel file support


Installation & Execution
1. Installation & Execution
pip install -r requirements.txt

2. Run the application
streamlit run fleiss_kappa_app.py

3. Access the application
- Open your web browser to http://localhost:8501
- The application will automatically load with study data analysis


**Usage Instructions**

1. Study Data Analysis
i.  Default view shows original research data (20 indigenous plants)
ii. Click "Run Fleiss Kappa Analysis on Study Data" to generate results
iii. View comprehensive statistics including
  - Fleiss' Kappa coefficient
  - P-value and statistical significance
  - Observed vs. expected agreement rates
  - Interpretation based on Landis & Koch scale

2. User Data Input
i. Switch to "Insert Your Own Data" in the navigation sidebar
ii. Add plant classifications
  - Enter unique plant name (required)
  - Select classification from each AI model (Medicinal, Edible, Poisonous, No Results)
  - Click "Add Plant" to submit (form auto-clears)
    
iii. Minimum requirement: 5 plants for valid Fleiss Kappa analysis
iv. Run analysis when sufficient data is collected



**Output Interpretation**
Fleiss' Kappa Interpretation

< 0.00: Poor agreement (less than chance)

0.00-0.20: Slight agreement

0.21-0.40: Fair agreement

0.41-0.60: Moderate agreement

0.61-0.80: Substantial agreement

0.81-1.00: Almost perfect agreement


**Troubleshooting**
Visualization errors: Ensure matplotlib and seaborn are properly installed

Session issues: Clear browser cache or restart the application

Data persistence: User data is maintained during session only







































