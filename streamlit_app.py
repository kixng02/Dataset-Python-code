import streamlit as st
import pandas as pd
import numpy as np
import io

# Try to import visualization libraries with error handling
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from matplotlib.colors import ListedColormap
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    st.error(f"Visualization libraries not available: {e}")
    VISUALIZATION_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Fleiss Kappa Analysis - AI Bias Study",
    page_icon="🫧",
    layout="wide"
)

class FleissKappaAnalyzer:
    """
    A comprehensive Fleiss' Kappa calculator for assessing inter-rater agreement
    among multiple AI models on indigenous plant classification.
    """
    
    def __init__(self):
        self.kappa = None
        self.p_value = None
        self.interpretation = None
        
    def validate_input(self, ratings):
        """Validate input data format and values"""
        if not isinstance(ratings, (np.ndarray, pd.DataFrame, list)):
            raise ValueError("Input must be numpy array, pandas DataFrame, or list")
            
        ratings = np.array(ratings)
        
        if ratings.ndim != 2:
            raise ValueError("Input must be 2-dimensional matrix")
            
        if ratings.size == 0:
            raise ValueError("Input matrix cannot be empty")
            
        # Check for non-integer values
        if not np.all(np.equal(np.mod(ratings, 1), 0)):
            raise ValueError("All ratings must be integer values")
            
        return ratings.astype(int)
    
    def calculate_fleiss_kappa(self, ratings, categories=None):
        """
        Calculate Fleiss' Kappa for multiple raters.
        """
        try:
            # Input validation and preprocessing
            ratings = self.validate_input(ratings)
            n, k = ratings.shape  # n subjects, k raters
            
            if n < 2 or k < 2:
                raise ValueError("Need at least 2 subjects and 2 raters")
            
            # Determine categories automatically if not provided
            if categories is None:
                categories = list(range(int(np.min(ratings)), int(np.max(ratings)) + 1))
            m = len(categories)
            
            # Build frequency matrix
            freq_matrix = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    freq_matrix[i, j] = np.sum(ratings[i] == categories[j])
            
            # Calculate observed agreement (P₀)
            p0_numerator = 0
            for i in range(n):
                p0_numerator += np.sum(freq_matrix[i] * (freq_matrix[i] - 1))
            P0 = p0_numerator / (n * k * (k - 1))
            
            # Calculate expected agreement (Pₑ)
            p_j = np.sum(freq_matrix, axis=0) / (n * k)
            Pe = np.sum(p_j ** 2)
            
            # Handle edge cases
            if Pe == 1:
                self.kappa = 1.0  # Perfect agreement
            else:
                self.kappa = (P0 - Pe) / (1 - Pe)
            
            # Calculate statistical significance
            self.p_value = self._calculate_significance(n, k, P0, Pe)
            
            # Interpret results
            self.interpretation = self._interpret_kappa(self.kappa)
            
            return {
                'kappa': self.kappa,
                'p_value': self.p_value,
                'interpretation': self.interpretation,
                'observed_agreement': P0,
                'expected_agreement': Pe,
                'n_subjects': n,
                'n_raters': k,
                'categories': categories
            }
            
        except Exception as e:
            st.error(f"Error calculating Fleiss' Kappa: {str(e)}")
            return None
    
    def _calculate_significance(self, n, k, P0, Pe):
        """Calculate approximate p-value for Fleiss' Kappa"""
        if not VISUALIZATION_AVAILABLE:
            # Simple p-value calculation without scipy
            if Pe == 1:
                return 0.0
            se = np.sqrt((2 * (1 - Pe)) / (n * k * (k - 1)))
            z_score = self.kappa / se if se > 0 else 0
            # Basic normal approximation
            p_value = 2 * (1 - 0.5 * (1 + np.math.erf(abs(z_score) / np.sqrt(2))))
            return p_value
            
        if Pe == 1:
            return 0.0
            
        # Standard error approximation
        se = np.sqrt((2 * (1 - Pe)) / (n * k * (k - 1)))
        z_score = self.kappa / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return p_value
    
    def _interpret_kappa(self, kappa):
        """Interpret Kappa value using Landis & Koch scale"""
        if kappa < 0:
            return "Poor agreement (less than chance)"
        elif kappa <= 0.2:
            return "Slight agreement"
        elif kappa <= 0.4:
            return "Fair agreement"
        elif kappa <= 0.6:
            return "Moderate agreement"
        elif kappa <= 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

def prepare_actual_classification_data():
    """
    Convert actual Table 1 data into numerical format for Fleiss' Kappa
    Encoding: 0 = Medicinal, 1 = Edible, 2 = Poisonous, -1 = No results/Not accurate
    """
    
    plant_names = [
        "Aloe ferox", "African ginger", "Wild rosemary", "Devil's claw", 
        "African wormwood", "Pepperbark tree", "Pineapple flower", "Spekboom",
        "False horsewood", "Sand raisin", "Mountain nettle", "Acacia",
        "River karee", "Kudu lily", "Waterberg raisin", "Sweet wild garlic",
        "Cyrtanthus sanguineus", "Ruttya fruticosa", "Sesamum trilobum", "Aloe hahnii"
    ]
    
    actual_classifications = [
        [0, 0, 1],       # Aloe ferox: Medicinal, Medicinal, Edible
        [0, 0, 0],       # African ginger: All Medicinal
        [-1, -1, -1],    # Wild rosemary: All No results
        [0, 0, 0],       # Devil's claw: All Medicinal
        [0, 0, 0],       # African wormwood: All Medicinal
        [-1, -1, -1],    # Pepperbark tree: All Not accurate
        [0, 0, 0],       # Pineapple flower: All Medicinal
        [-1, -1, -1],    # Spekboom: All Not accurate
        [2, -1, -1],     # False horsewood: Poisonous, No results, Not accurate
        [-1, -1, 1],     # Sand raisin: No results, Not accurate, Edible
        [-1, -1, -1],    # Mountain nettle: All Not accurate
        [0, 2, 2],       # Acacia: Medicinal, Poisonous, Poisonous
        [0, -1, -1],     # River karee: Medicinal, No results, Not accurate
        [0, -1, 0],      # Kudu lily: Medicinal, Not accurate, Medicinal
        [-1, -1, -1],    # Waterberg raisin: All Not accurate
        [-1, -1, -1],    # Sweet wild garlic: All No results
        [-1, -1, 1],     # Cyrtanthus sanguineus: Not accurate, Not accurate, Edible
        [-1, 0, -1],     # Ruttya fruticosa: No results, Medicinal, Not accurate
        [-1, -1, 1],     # Sesamum trilobum: No results, No results, Edible
        [-1, 0, -1]      # Aloe hahnii: No results, Medicinal, Not accurate
    ]
    
    return actual_classifications, plant_names

def text_to_code(text_response):
    """Convert text responses to numerical codes"""
    response_map = {
        'Medicinal': 0,
        'Edible': 1,
        'Poisonous': 2,
        'No Results': -1
    }
    return response_map.get(text_response, -1)

def code_to_text(code):
    """Convert numerical codes back to text"""
    code_map = {
        0: 'Medicinal',
        1: 'Edible',
        2: 'Poisonous',
        -1: 'No Results'
    }
    return code_map.get(code, 'No Results')

def create_agreement_heatmap(ratings, plant_names, model_names):
    """Create a heatmap visualization of agreement patterns"""
    if not VISUALIZATION_AVAILABLE:
        return None
        
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert to agreement matrix (1 = all agree, 0 = disagree)
    agreement_matrix = np.zeros(len(ratings))
    
    for i in range(len(ratings)):
        valid_ratings = [r for r in ratings[i] if r != -1]
        if len(valid_ratings) > 0:
            agreement_matrix[i] = len(set(valid_ratings)) == 1
        else:
            agreement_matrix[i] = 0
    
    agreement_display = np.tile(agreement_matrix, (3, 1)).T
    
    sns.heatmap(agreement_display, 
               xticklabels=model_names,
               yticklabels=plant_names,
               cmap=['red', 'green'],
               cbar_kws={'label': 'Key (Red=Disagree, Green=Agree)'},
               ax=ax)
    
    ax.set_title('Inter-Model Agreement Patterns on Indigenous Plant Classification')
    ax.set_xlabel('AI Models')
    ax.set_ylabel('Indigenous Plants')
    
    return fig

def create_detailed_classification_heatmap(ratings, plant_names, model_names):
    """Create a detailed heatmap showing actual classifications"""
    if not VISUALIZATION_AVAILABLE:
        return None, None
        
    # Convert numerical ratings to descriptive labels for visualization
    rating_labels = []
    for plant_ratings in ratings:
        labels = [code_to_text(r) for r in plant_ratings]
        rating_labels.append(labels)
    
    rating_df = pd.DataFrame(rating_labels, 
                           index=plant_names, 
                           columns=model_names)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create custom colormap
    cmap = ListedColormap(['#2E8B57', '#FFD700', '#DC143C', '#696969'])
    
    heatmap_data = rating_df.apply(lambda x: pd.Categorical(x).codes)
    sns.heatmap(heatmap_data, 
               cmap=cmap,
               xticklabels=model_names,
               yticklabels=plant_names,
               cbar_kws={'ticks': [0, 1, 2, 3], 
                       'label': 'Classification'},
               ax=ax)
    
    # Customize colorbar labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['Medicinal', 'Edible', 'Poisonous', 'No Results'])
    
    ax.set_title('Detailed AI Model Classifications\n(User-Provided Data)')
    ax.set_xlabel('AI Models')
    ax.set_ylabel('Indigenous Plants')
    
    return fig, rating_df

def create_results_chart(results):
    """Create a bar chart of key results"""
    if not VISUALIZATION_AVAILABLE:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Kappa', 'Observed Agreement', 'Expected Agreement']
    values = [results['kappa'], results['observed_agreement'], results['expected_agreement']]
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7)
    ax.set_ylabel('Score')
    ax.set_title('Fleiss Kappa Analysis Results')
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom')
    
    return fig

def create_simple_visualization(results, ratings, plant_names, model_names):
    """Create simple text-based visualizations when matplotlib is not available"""
    
    st.subheader("Analysis Results (Text-based)")
    
    # Create a simple table for results
    results_data = {
        'Metric': ['Fleiss Kappa', 'P-value', 'Observed Agreement', 'Expected Agreement', 'Interpretation'],
        'Value': [
            f"{results['kappa']:.4f}",
            f"{results['p_value']:.6f}",
            f"{results['observed_agreement']:.4f}",
            f"{results['expected_agreement']:.4f}",
            results['interpretation']
        ]
    }
    st.table(pd.DataFrame(results_data))
    
    # Agreement analysis
    st.subheader("Agreement Analysis")
    total_agreements = 0
    agreement_details = []
    
    for i, plant_ratings in enumerate(ratings):
        valid_ratings = [r for r in plant_ratings if r != -1]
        if len(valid_ratings) > 0 and len(set(valid_ratings)) == 1:
            total_agreements += 1
            agreement_details.append(f"✅ {plant_names[i]}: All models agree")
        else:
            agreement_details.append(f"❌ {plant_names[i]}: Models disagree")
    
    agreement_rate = total_agreements / len(ratings)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Agreement Rate", f"{agreement_rate:.1%}")
    with col2:
        st.metric("Plants with Consensus", f"{total_agreements}/{len(ratings)}")
    
    # Show detailed agreement status
    with st.expander("View Detailed Agreement Status"):
        for detail in agreement_details:
            st.write(detail)

def create_visualizations(results, ratings, plant_names, model_names):
    """Create all visualizations including detailed heatmaps"""
    
    if not VISUALIZATION_AVAILABLE:
        create_simple_visualization(results, ratings, plant_names, model_names)
        return
    
    try:
        # Results bar chart
        st.subheader(" Results Visualization")
        results_chart = create_results_chart(results)
        if results_chart:
            st.pyplot(results_chart)
        
        # Agreement heatmap
        st.subheader("🔄 Agreement Patterns")
        agreement_fig = create_agreement_heatmap(ratings, plant_names, model_names)
        if agreement_fig:
            st.pyplot(agreement_fig)
            st.caption("Green indicates all models agree, red indicates disagreement or inconsistent classifications")
        
        # Detailed classification heatmap
        st.subheader("Detailed Classification Heatmap")
        detailed_fig, detailed_df = create_detailed_classification_heatmap(ratings, plant_names, model_names)
        if detailed_fig:
            st.pyplot(detailed_fig)
            st.caption("Complete classification data showing each AI model's response for every plant")
            
            # Show the data table
            with st.expander("View Raw Classification Data"):
                st.dataframe(detailed_df)
        
    except Exception as e:
        st.warning(f"Could not create visualizations: {e}")
        create_simple_visualization(results, ratings, plant_names, model_names)

def get_existing_plant_names():
    """Get list of existing plant names to prevent duplicates"""
    if 'user_plants' not in st.session_state:
        return []
    return [plant['plant_name'].lower().strip() for plant in st.session_state.user_plants]

def user_data_input():
    """Allow users to input their own plant classification data"""
    st.header("🌿 Insert Your Own Data")
    
    st.markdown("""
    **Instructions:**
    1. Enter at least **5 plant names** (each plant name must be unique)
    2. Select the classification response from each AI model for each plant
    3. Available responses: **Medicinal, Edible, Poisonous, No Results**
    4. Click 'Add Plant' after entering each plant's data
    5. Click 'Run Analysis with User Data' when you have at least 5 plants
    """)
    
    # Initialize session state for user data
    if 'user_plants' not in st.session_state:
        st.session_state.user_plants = []
    
    # Initialize session state for form clearing
    if 'form_cleared' not in st.session_state:
        st.session_state.form_cleared = False
    
    # Get existing plant names for duplicate checking
    existing_plants = get_existing_plant_names()
    
    # Input form for new plant
    with st.form("plant_input_form", clear_on_submit=True):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            plant_name = st.text_input("Plant Name *", placeholder="e.g., Aloe vera", key="plant_name_input")
        with col2:
            chatgpt_response = st.selectbox("ChatGPT *", ["No Results", "Medicinal", "Edible", "Poisonous"], key="chatgpt_input")
        with col3:
            gemini_response = st.selectbox("Gemini *", ["No Results", "Medicinal", "Edible", "Poisonous"], key="gemini_input")
        with col4:
            mistral_response = st.selectbox("Mistral AI *", ["No Results", "Medicinal", "Edible", "Poisonous"], key="mistral_input")
        
        st.markdown("**All fields are required**")
        
        submitted = st.form_submit_button("Add Plant")
        
        if submitted:
            # Validate all fields are filled
            if not plant_name.strip():
                st.error("❌ Please enter a plant name.")
            elif plant_name.lower().strip() in existing_plants:
                st.error(f"❌ Plant '{plant_name}' already exists in your dataset. Please use a different name.")
            else:
                # All validations passed, add the plant
                plant_data = {
                    'plant_name': plant_name.strip(),
                    'chatgpt': text_to_code(chatgpt_response),
                    'gemini': text_to_code(gemini_response),
                    'mistral': text_to_code(mistral_response)
                }
                st.session_state.user_plants.append(plant_data)
                st.success(f"✅ Added '{plant_name}' to the dataset!")
                
                # Set flag to show the form was cleared
                st.session_state.form_cleared = True
    
    # Display current user data
    if st.session_state.user_plants:
        st.subheader("📋 Your Current Plant Data")
        display_data = []
        for plant in st.session_state.user_plants:
            display_data.append({
                'Plant Name': plant['plant_name'],
                'ChatGPT': code_to_text(plant['chatgpt']),
                'Gemini': code_to_text(plant['gemini']),
                'Mistral AI': code_to_text(plant['mistral'])
            })
        
        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True)
        
        # Show data statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Plants", len(st.session_state.user_plants))
        with col2:
            unique_plants = len(set([p['plant_name'].lower() for p in st.session_state.user_plants]))
            st.metric("Unique Plants", unique_plants)
        with col3:
            if len(st.session_state.user_plants) >= 5:
                st.metric("Status", "Ready for Analysis ✅")
            else:
                st.metric("Status", "Need More Data ⚠️")
        
        # Clear data button
        if st.button("🗑️ Clear All Data"):
            st.session_state.user_plants = []
            st.session_state.form_cleared = False
            st.rerun()
        
        # Check if we have enough data for analysis
        if len(st.session_state.user_plants) >= 5:
            st.success(f"✅ You have {len(st.session_state.user_plants)} plants. Ready for analysis!")
            return True
        else:
            st.warning(f"⚠️ You have {len(st.session_state.user_plants)} plants. Need at least 5 for analysis.")
            return False
    else:
        st.info("Start by adding your first plant using the form above.")
        return False

def prepare_user_data():
    """Convert user data to the format needed for Fleiss Kappa analysis"""
    if 'user_plants' not in st.session_state:
        return [], []
    
    plant_names = []
    classifications = []
    
    for plant in st.session_state.user_plants:
        plant_names.append(plant['plant_name'])
        classifications.append([
            plant['chatgpt'],
            plant['gemini'], 
            plant['mistral']
        ])
    
    return classifications, plant_names

def run_study_data_analysis(analyzer):
    """Run and display the study data analysis"""
    st.header("📋 Research Study Data Analysis")
    st.markdown("""
    This analysis uses the original research data from **Table 1** of the study:
    **"A model for addressing AI algorithms biasness through indigenous South African knowledge systems."**
    
    The data shows how three AI models (ChatGPT, Gemini, Mistral AI) classified 20 indigenous South African plants.
    """)
    
    # Load actual data
    ratings, plant_names = prepare_actual_classification_data()
    model_names = ["ChatGPT", "Gemini", "Mistral AI"]
    
    # Display data table
    st.subheader("Research Data (Table 1)")
    display_data = []
    for i, plant in enumerate(plant_names):
        display_data.append({
            'Plant Name': plant,
            'ChatGPT': code_to_text(ratings[i][0]),
            'Gemini': code_to_text(ratings[i][1]),
            'Mistral AI': code_to_text(ratings[i][2])
        })
    
    df_display = pd.DataFrame(display_data)
    st.dataframe(df_display, use_container_width=True)
    
    # Run analysis automatically on first load
    if st.button("Run Fleiss Kappa Analysis on Study Data", type="primary"):
        with st.spinner("Calculating Fleiss' Kappa for research data..."):
            results = analyzer.calculate_fleiss_kappa(ratings, categories=[-1, 0, 1, 2])
        
        if results:
            # Display results
            st.header(" Research Study Results")
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Fleiss' Kappa", f"{results['kappa']:.3f}")
            
            with col2:
                st.metric("P-value", f"{results['p_value']:.4f}")
            
            with col3:
                st.metric("Observed Agreement", f"{results['observed_agreement']:.3f}")
            
            with col4:
                st.metric("Expected Agreement", f"{results['expected_agreement']:.3f}")
            
            # Interpretation
            st.info(f"**Interpretation**: {results['interpretation']}")
            
            # Create visualizations
            create_visualizations(results, ratings, plant_names, model_names)
            
            # Additional statistics
            st.subheader("Additional Statistics")
            total_agreements = 0
            for plant_ratings in ratings:
                valid_ratings = [r for r in plant_ratings if r != -1]
                if len(valid_ratings) > 0 and len(set(valid_ratings)) == 1:
                    total_agreements += 1
            
            agreement_rate = total_agreements / len(ratings)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Agreement Rate", f"{agreement_rate:.1%}")
            with col2:
                st.metric("Plants with Consensus", f"{total_agreements}/{len(ratings)}")
            with col3:
                st.metric("Statistical Significance", 
                         "Yes" if results['p_value'] < 0.05 else "No")
            
            # Research implications
            st.subheader("🔬 Research Implications")
            st.markdown("""
            - **Low Kappa Values** indicate significant disagreement among AI models
            - **Inconsistent classifications** highlight bias in training data
            - **No Results patterns** show gaps in indigenous knowledge representation
            - **Findings support** the need for integrating Indigenous Knowledge Systems into AI training
            - **Cultural bias** is evident in the misclassification of indigenous plants
            """)

def main():
    # Main title and description
    st.title("Fleiss Kappa Analysis: AI Model Agreement on Plant Classification")
    st.markdown("""
    This application analyzes the inter-rater agreement between AI models (ChatGPT, Gemini, Mistral AI) 
    on classifying plants using Fleiss' Kappa statistic.
    
    **Start by exploring the original research data, then try with your own data!**
    """)
    
    # Installation notice if libraries are missing
    if not VISUALIZATION_AVAILABLE:
        st.warning("""
        ⚠️ **Visualization libraries not available** 
        For full functionality including charts and graphs, please install:
        ```bash
        pip install matplotlib seaborn scipy
        ```
        The app will continue with text-based analysis.
        """)
    
    # Initialize analyzer
    analyzer = FleissKappaAnalyzer()
    
    # Sidebar for navigation - Set default to Study Data Analysis
    st.sidebar.title("Navigation")
    
    # Set default selection to Study Data Analysis
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "Study Data Analysis"
    
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Study Data Analysis", "Insert Your Own Data", "About"],
        index=0  # Default to Study Data Analysis
    )
    
    # Update session state
    st.session_state.app_mode = app_mode
    
    if app_mode == "Study Data Analysis":
        # Show study data analysis by default
        run_study_data_analysis(analyzer)
    
    elif app_mode == "Insert Your Own Data":
        # User data input and analysis
        ready_for_analysis = user_data_input()
        
        if ready_for_analysis:
            st.header("Analyze Your Data")
            
            if st.button("Run Fleiss Kappa Analysis with Your Data", type="primary"):
                with st.spinner("Calculating Fleiss' Kappa..."):
                    # Prepare user data
                    ratings, plant_names = prepare_user_data()
                    model_names = ["ChatGPT", "Gemini", "Mistral AI"]
                    
                    # Run analysis
                    results = analyzer.calculate_fleiss_kappa(ratings, categories=[-1, 0, 1, 2])
                
                if results:
                    # Display results
                    st.header(" Your Analysis Results")
                    
                    # Key metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Fleiss' Kappa", f"{results['kappa']:.3f}")
                    
                    with col2:
                        st.metric("P-value", f"{results['p_value']:.4f}")
                    
                    with col3:
                        st.metric("Observed Agreement", f"{results['observed_agreement']:.3f}")
                    
                    with col4:
                        st.metric("Expected Agreement", f"{results['expected_agreement']:.3f}")
                    
                    # Interpretation
                    st.info(f"**Interpretation**: {results['interpretation']}")
                    
                    # Create visualizations
                    create_visualizations(results, ratings, plant_names, model_names)
                    
                    # Additional statistics
                    st.subheader("📈 Additional Statistics")
                    total_agreements = 0
                    for plant_ratings in ratings:
                        valid_ratings = [r for r in plant_ratings if r != -1]
                        if len(valid_ratings) > 0 and len(set(valid_ratings)) == 1:
                            total_agreements += 1
                    
                    agreement_rate = total_agreements / len(ratings)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Agreement Rate", f"{agreement_rate:.1%}")
                    with col2:
                        st.metric("Plants with Consensus", f"{total_agreements}/{len(ratings)}")
                    with col3:
                        st.metric("Statistical Significance", 
                                 "Yes" if results['p_value'] < 0.05 else "No")
    
    else:  # About mode
        st.header("ℹ️ About This Analysis")
        st.markdown("""
        ### Research Context
        This analysis is part of a study on **"A model for addressing AI algorithms biasness through indigenous South African knowledge systems."**
        
        ### Methodology
        - **Fleiss' Kappa**: Statistical measure for assessing inter-rater agreement
        - **AI Models**: ChatGPT, Google Gemini, Mistral AI
        - **Classifications**: Medicinal, Edible, Poisonous, or No Results
        
        ### Interpretation Guide
        - **< 0.00**: Poor agreement
        - **0.00 - 0.20**: Slight agreement  
        - **0.21 - 0.40**: Fair agreement
        - **0.41 - 0.60**: Moderate agreement
        - **0.61 - 0.80**: Substantial agreement
        - **0.81 - 1.00**: Almost perfect agreement
        
        ### Installation Requirements
        For full functionality with visualizations:
        ```bash
        pip install streamlit matplotlib seaborn scipy
        ```
        """)

if __name__ == "__main__":
    main()
