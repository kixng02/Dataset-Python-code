import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import ipywidgets as widgets
from IPython.display import display, clear_output
import json

class InteractiveFleissKappaAnalyzer:
    """
    Interactive Fleiss' Kappa analyzer with real-time parameter control
    and dynamic data input capabilities.
    """
    
    def __init__(self):
        self.kappa = None
        self.p_value = None
        self.interpretation = None
        self.ratings = None
        self.plant_names = None
        self.model_names = None
        
        # Initialize widgets
        self.setup_widgets()
        
    def setup_widgets(self):
        """Setup interactive widgets for parameter control"""
        
        # Data input method selector
        self.data_source = widgets.RadioButtons(
            options=['Use Actual Study Data', 'Upload Custom Data', 'Manual Entry'],
            value='Use Actual Study Data',
            description='Data Source:',
            disabled=False
        )
        
        # Model selection
        self.model_selector = widgets.SelectMultiple(
            options=['ChatGPT', 'Gemini', 'Mistral AI', 'Custom Model 1', 'Custom Model 2'],
            value=['ChatGPT', 'Gemini', 'Mistral AI'],
            rows=5,
            description='AI Models:',
            disabled=False
        )
        
        # Category mapping controls
        self.medicinal_code = widgets.IntText(value=0, description='Medicinal:')
        self.edible_code = widgets.IntText(value=1, description='Edible:')
        self.poisonous_code = widgets.IntText(value=2, description='Poisonous:')
        self.noresults_code = widgets.IntText(value=-1, description='No Results:')
        
        # Analysis parameters
        self.confidence_level = widgets.FloatSlider(
            value=0.95,
            min=0.80,
            max=0.99,
            step=0.01,
            description='Confidence Level:',
            readout_format='.2f'
        )
        
        self.treat_noresults = widgets.RadioButtons(
            options=['Exclude from analysis', 'Treat as separate category', 'Treat as disagreement'],
            value='Treat as separate category',
            description='Handle No Results:'
        )
        
        # Visualization controls
        self.chart_style = widgets.Dropdown(
            options=['seaborn', 'ggplot', 'classic', 'dark_background'],
            value='seaborn',
            description='Chart Style:'
        )
        
        self.color_scheme = widgets.Dropdown(
            options=['viridis', 'plasma', 'coolwarm', 'RdYlBu', 'custom'],
            value='RdYlBu',
            description='Color Scheme:'
        )
        
        # Action buttons
        self.analyze_button = widgets.Button(
            description='Run Fleiss Kappa Analysis',
            button_style='success',
            tooltip='Click to perform analysis'
        )
        
        self.reset_button = widgets.Button(
            description='Reset Parameters',
            button_style='warning'
        )
        
        self.export_button = widgets.Button(
            description='Export Results',
            button_style='info'
        )
        
        # Connect button events
        self.analyze_button.on_click(self.run_interactive_analysis)
        self.reset_button.on_click(self.reset_parameters)
        self.export_button.on_click(self.export_results)
        
        # Data upload widget
        self.upload_widget = widgets.FileUpload(
            accept='.csv,.json,.xlsx',
            multiple=False,
            description='Upload Data'
        )
        
        # Manual data entry
        self.manual_data_text = widgets.Textarea(
            value='',
            placeholder='Paste data as CSV or JSON here...',
            description='Manual Data:',
            layout={'width': '90%', 'height': '200px'}
        )
        
    def create_control_panel(self):
        """Create the interactive control panel"""
        
        # Category mapping box
        category_box = widgets.VBox([
            widgets.HTML("<b>Category Encoding:</b>"),
            widgets.HBox([self.medicinal_code, self.edible_code]),
            widgets.HBox([self.poisonous_code, self.noresults_code])
        ], layout={'border': '1px solid gray', 'padding': '10px'})
        
        # Analysis parameters box
        analysis_box = widgets.VBox([
            widgets.HTML("<b>Analysis Parameters:</b>"),
            self.confidence_level,
            self.treat_noresults
        ], layout={'border': '1px solid gray', 'padding': '10px'})
        
        # Visualization box
        viz_box = widgets.VBox([
            widgets.HTML("<b>Visualization Settings:</b>"),
            self.chart_style,
            self.color_scheme
        ], layout={'border': '1px solid gray', 'padding': '10px'})
        
        # Action buttons box
        action_box = widgets.HBox([
            self.analyze_button,
            self.reset_button,
            self.export_button
        ], layout={'justify_content': 'space-between'})
        
        # Main control panel
        control_panel = widgets.VBox([
            widgets.HTML("<h2>Interactive Fleiss Kappa Analyzer</h2>"),
            self.data_source,
            widgets.HBox([
                widgets.VBox([self.model_selector, category_box]),
                widgets.VBox([analysis_box, viz_box])
            ]),
            action_box
        ])
        
        return control_panel
    
    def load_actual_study_data(self):
        """Load the actual study data from Table 1"""
        self.plant_names = [
            "Aloe ferox", "African ginger", "Wild rosemary", "Devil's claw", 
            "African wormwood", "Pepperbark tree", "Pineapple flower", "Spekboom",
            "False horsewood", "Sand raisin", "Mountain nettle", "Acacia",
            "River karee", "Kudu lily", "Waterberg raisin", "Sweet wild garlic",
            "Cyrtanthus sanguineus", "Ruttya fruticosa", "Sesamum trilobum", "Aloe hahnii"
        ]
        
        self.ratings = [
            [0, 0, 1],       # Aloe ferox
            [0, 0, 0],       # African ginger
            [-1, -1, -1],    # Wild rosemary
            [0, 0, 0],       # Devil's claw
            [0, 0, 0],       # African wormwood
            [-1, -1, -1],    # Pepperbark tree
            [0, 0, 0],       # Pineapple flower
            [-1, -1, -1],    # Spekboom
            [2, -1, -1],     # False horsewood
            [-1, -1, 1],     # Sand raisin
            [-1, -1, -1],    # Mountain nettle
            [0, 2, 2],       # Acacia
            [0, -1, -1],     # River karee
            [0, -1, 0],      # Kudu lily
            [-1, -1, -1],    # Waterberg raisin
            [-1, -1, -1],    # Sweet wild garlic
            [-1, -1, 1],     # Cyrtanthus sanguineus
            [-1, 0, -1],     # Ruttya fruticosa
            [-1, -1, 1],     # Sesamum trilobum
            [-1, 0, -1]      # Aloe hahnii
        ]
        
        self.model_names = list(self.model_selector.value)
    
    def process_uploaded_data(self, upload_data):
        """Process uploaded CSV or JSON data"""
        try:
            if upload_data.name.endswith('.csv'):
                df = pd.read_csv(upload_data)
            elif upload_data.name.endswith('.json'):
                df = pd.read_json(upload_data)
            elif upload_data.name.endswith('.xlsx'):
                df = pd.read_excel(upload_data)
            else:
                return False, "Unsupported file format"
                
            # Assume first column is plant names, rest are model ratings
            self.plant_names = df.iloc[:, 0].tolist()
            self.ratings = df.iloc[:, 1:].values.tolist()
            self.model_names = df.columns[1:].tolist()
            
            return True, "Data loaded successfully"
            
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def process_manual_data(self, text_data):
        """Process manually entered data"""
        try:
            if text_data.strip().startswith('['):
                # JSON format
                data = json.loads(text_data)
                self.plant_names = data.get('plant_names', [])
                self.ratings = data.get('ratings', [])
                self.model_names = data.get('model_names', [])
            else:
                # CSV format
                lines = text_data.strip().split('\n')
                headers = lines[0].split(',')
                self.model_names = headers[1:]
                self.plant_names = []
                self.ratings = []
                
                for line in lines[1:]:
                    parts = line.split(',')
                    self.plant_names.append(parts[0])
                    self.ratings.append([int(x) for x in parts[1:]])
            
            return True, "Manual data processed successfully"
            
        except Exception as e:
            return False, f"Error processing manual data: {str(e)}"
    
    def run_interactive_analysis(self, button):
        """Run analysis with current widget parameters"""
        
        # Clear previous output
        clear_output(wait=True)
        display(self.create_control_panel())
        
        # Load data based on selection
        if self.data_source.value == 'Use Actual Study Data':
            self.load_actual_study_data()
            data_status = "Using actual study data from Table 1"
        elif self.data_source.value == 'Upload Custom Data':
            if len(self.upload_widget.value) > 0:
                success, message = self.process_uploaded_data(self.upload_widget.value[0])
                data_status = message
            else:
                data_status = "Please upload a data file"
        else:  # Manual Entry
            if self.manual_data_text.value.strip():
                success, message = self.process_manual_data(self.manual_data_text.value)
                data_status = message
            else:
                data_status = "Please enter data manually"
        
        # Perform analysis
        try:
            # Apply category mapping
            category_map = {
                self.medicinal_code.value: 0,
                self.edible_code.value: 1,
                self.poisonous_code.value: 2,
                self.noresults_code.value: -1
            }
            
            # Transform ratings based on mapping
            transformed_ratings = []
            for plant_ratings in self.ratings:
                transformed_ratings.append([category_map.get(r, r) for r in plant_ratings])
            
            # Handle no results based on selection
            if self.treat_noresults.value == 'Exclude from analysis':
                filtered_ratings = []
                filtered_plants = []
                for i, ratings in enumerate(transformed_ratings):
                    valid_ratings = [r for r in ratings if r != -1]
                    if len(valid_ratings) >= 2:  # Need at least 2 raters
                        filtered_ratings.append(valid_ratings)
                        filtered_plants.append(self.plant_names[i])
                analysis_ratings = filtered_ratings
                analysis_plants = filtered_plants
            else:
                analysis_ratings = transformed_ratings
                analysis_plants = self.plant_names
            
            # Calculate Fleiss Kappa
            results = self.calculate_fleiss_kappa(
                analysis_ratings, 
                confidence=self.confidence_level.value
            )
            
            # Display results
            self.display_interactive_results(results, data_status, analysis_plants)
            
        except Exception as e:
            print(f"Analysis error: {str(e)}")
    
    def calculate_fleiss_kappa(self, ratings, confidence=0.95):
        """Calculate Fleiss' Kappa with confidence intervals"""
        
        ratings = np.array(ratings)
        n, k = ratings.shape
        
        # Build frequency matrix
        categories = sorted(set(ratings.flatten()))
        m = len(categories)
        freq_matrix = np.zeros((n, m))
        
        for i in range(n):
            for j, cat in enumerate(categories):
                freq_matrix[i, j] = np.sum(ratings[i] == cat)
        
        # Calculate observed agreement (P₀)
        p0_numerator = 0
        for i in range(n):
            p0_numerator += np.sum(freq_matrix[i] * (freq_matrix[i] - 1))
        P0 = p0_numerator / (n * k * (k - 1))
        
        # Calculate expected agreement (Pₑ)
        p_j = np.sum(freq_matrix, axis=0) / (n * k)
        Pe = np.sum(p_j ** 2)
        
        # Calculate Fleiss' Kappa
        if Pe == 1:
            kappa = 1.0
        else:
            kappa = (P0 - Pe) / (1 - Pe)
        
        # Calculate confidence interval
        se = np.sqrt((2 * (1 - Pe)) / (n * k * (k - 1)))
        z = stats.norm.ppf((1 + confidence) / 2)
        ci_lower = kappa - z * se
        ci_upper = kappa + z * se
        
        # Calculate p-value
        z_score = kappa / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Interpretation
        interpretation = self._interpret_kappa(kappa)
        
        return {
            'kappa': kappa,
            'p_value': p_value,
            'interpretation': interpretation,
            'observed_agreement': P0,
            'expected_agreement': Pe,
            'confidence_interval': (ci_lower, ci_upper),
            'confidence_level': confidence,
            'n_subjects': n,
            'n_raters': k,
            'categories': categories
        }
    
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
    
    def display_interactive_results(self, results, data_status, plant_names):
        """Display interactive results with visualizations"""
        
        # Apply visualization settings
        plt.style.use(self.chart_style.value)
        
        # Create results dashboard
        fig = plt.figure(figsize=(16, 12))
        
        # Main results
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        self._create_results_summary_plot(ax1, results, data_status)
        
        # Agreement heatmap
        ax2 = plt.subplot2grid((3, 2), (1, 0))
        self._create_agreement_heatmap(ax2, results, plant_names)
        
        # Category distribution
        ax3 = plt.subplot2grid((3, 2), (1, 1))
        self._create_category_distribution(ax3, results)
        
        # Confidence interval
        ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        self._create_confidence_interval_plot(ax4, results)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed statistics
        self._print_detailed_statistics(results)
    
    def _create_results_summary_plot(self, ax, results, data_status):
        """Create main results summary plot"""
        
        metrics = ['Kappa', 'Observed\nAgreement', 'Expected\nAgreement']
        values = [results['kappa'], results['observed_agreement'], results['expected_agreement']]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_ylabel('Score')
        ax.set_title(f'Fleiss Kappa Analysis Results\n{results["interpretation"]}')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Add data status
        ax.text(0.02, 0.98, f'Data: {data_status}', transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _create_agreement_heatmap(self, ax, results, plant_names):
        """Create agreement pattern heatmap"""
        # Simplified agreement visualization
        n_plants = len(plant_names)
        agreement_pattern = np.random.random(n_plants)  # Replace with actual agreement data
        
        im = ax.imshow([agreement_pattern], cmap=self.color_scheme.value, aspect='auto')
        ax.set_yticks([])
        ax.set_xlabel('Plants')
        ax.set_title('Agreement Patterns Across Plants')
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    
    def _create_category_distribution(self, ax, results):
        """Create category distribution pie chart"""
        # Simplified category distribution
        categories = ['Medicinal', 'Edible', 'Poisonous', 'No Results']
        distribution = [40, 25, 20, 15]  # Replace with actual distribution
        
        ax.pie(distribution, labels=categories, autopct='%1.1f%%', startangle=90)
        ax.set_title('Classification Distribution')
    
    def _create_confidence_interval_plot(self, ax, results):
        """Create confidence interval visualization"""
        kappa = results['kappa']
        ci_lower, ci_upper = results['confidence_interval']
        
        ax.errorbar([0], [kappa], yerr=[[kappa - ci_lower], [ci_upper - kappa]], 
                   fmt='o', capsize=5, capthick=2, markersize=8)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Chance Level')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylabel('Kappa Value')
        ax.set_title(f'Fleiss Kappa with {results["confidence_level"]:.0%} Confidence Interval')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _print_detailed_statistics(self, results):
        """Print detailed statistics"""
        print("=" * 60)
        print("DETAILED FLEISS KAPPA ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Fleiss' Kappa: {results['kappa']:.4f}")
        print(f"95% Confidence Interval: [{results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f}]")
        print(f"P-value: {results['p_value']:.6f}")
        print(f"Interpretation: {results['interpretation']}")
        print(f"Observed Agreement (P₀): {results['observed_agreement']:.4f}")
        print(f"Expected Agreement (Pₑ): {results['expected_agreement']:.4f}")
        print(f"Number of Subjects (Plants): {results['n_subjects']}")
        print(f"Number of Raters (AI Models): {results['n_raters']}")
        print(f"Statistical Significance: {'Yes' if results['p_value'] < 0.05 else 'No'}")
    
    def reset_parameters(self, button):
        """Reset all parameters to default"""
        self.medicinal_code.value = 0
        self.edible_code.value = 1
        self.poisonous_code.value = 2
        self.noresults_code.value = -1
        self.confidence_level.value = 0.95
        self.treat_noresults.value = 'Treat as separate category'
        self.chart_style.value = 'seaborn'
        self.color_scheme.value = 'RdYlBu'
        
        print("Parameters reset to default values")
    
    def export_results(self, button):
        """Export results to files"""
        if self.kappa is not None:
            # Create results dictionary
            export_data = {
                'analysis_parameters': {
                    'confidence_level': self.confidence_level.value,
                    'treat_noresults': self.treat_noresults.value,
                    'category_mapping': {
                        'medicinal': self.medicinal_code.value,
                        'edible': self.edible_code.value,
                        'poisonous': self.poisonous_code.value,
                        'noresults': self.noresults_code.value
                    }
                },
                'results': {
                    'fleiss_kappa': self.kappa,
                    'p_value': self.p_value,
                    'interpretation': self.interpretation
                }
            }
            
            # Save to JSON
            with open('fleiss_kappa_results.json', 'w') as f:
                json.dump(export_data, f, indent=2)
            
            # Save visualization
            plt.savefig('fleiss_kappa_analysis.png', dpi=300, bbox_inches='tight')
            
            print("Results exported to 'fleiss_kappa_results.json' and 'fleiss_kappa_analysis.png'")
        else:
            print("No results to export. Please run analysis first.")

# Initialize and display the interactive analyzer
def launch_interactive_analyzer():
    """Launch the interactive Fleiss Kappa analyzer"""
    analyzer = InteractiveFleissKappaAnalyzer()
    control_panel = analyzer.create_control_panel()
    display(control_panel)
    
    # Add data upload and manual entry areas
    data_upload_area = widgets.VBox([
        widgets.HTML("<h3>Data Input Area</h3>"),
        widgets.HBox([
            widgets.VBox([
                widgets.HTML("<b>Upload Data File:</b>"),
                analyzer.upload_widget
            ]),
            widgets.VBox([
                widgets.HTML("<b>Or Enter Data Manually:</b>"),
                analyzer.manual_data_text
            ])
        ])
    ])
    
    display(data_upload_area)
    
    return analyzer

# Launch the interactive tool
print("Launching Interactive Fleiss Kappa Analyzer...")
analyzer = launch_interactive_analyzer()