import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import json
from datetime import datetime
import plotly.io as pio
#from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


pio.templates[pio.templates.default].layout.colorway = ['#79cac1', '#3b9153', '#009dd2', '#f78e82', '#d774ae', '#69008c']

# Configure the page
st.set_page_config(
    page_title="Business Rules and Escalation Reason Simulator",
    page_icon="📊",
    layout="wide"
)

def load_default_thresholds():
    """Load default thresholds from JSON file"""

    try:
        # Attempt to load thresholds from file
        with open('thresholds.json', 'r') as f:
            loaded_thresholds = json.load(f)
            
        # Get list of countries from loaded thresholds
        countries = list(loaded_thresholds.keys())
            
        # Validate and process loaded thresholds
        final_thresholds = {}
        
        # Process each country from the loaded file
        for country in countries:
            if country == 'DEFAULT':
                continue  # Already handled above
                
            final_thresholds[country] = {}
            for key in final_thresholds[0].keys():
                final_thresholds[country][key] = loaded_thresholds[country][key]
        
        return final_thresholds
        
    except FileNotFoundError:
        st.warning("Please upload thresholds and data files.")
        
    except json.JSONDecodeError:
        # Handle corrupted JSON file
        st.error("Error reading thresholds.json. File may be corrupted.")
    
    except Exception as e:
        # Handle any other errors
        st.warning(f"Please upload thresholds and data files. ")

def reload_thresholds_from_file():
    """Reload thresholds from JSON file."""
    try:
        with open('thresholds.json', 'r') as f:
            loaded_thresholds = json.load(f)
            st.session_state.thresholds = loaded_thresholds
            return True
    except Exception as e:
        st.error(f"Error reloading thresholds: {str(e)}")
        return False
    
def save_thresholds(thresholds):
    """Save thresholds to JSON file."""
    try:
        with open('thresholds.json', 'w') as f:
            json.dump(thresholds, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving thresholds: {str(e)}")
        return False

def initialize_session_state():
    if 'thresholds' not in st.session_state:
        st.session_state.thresholds = load_default_thresholds()


def display_threshold_controls(country):
    """Display and handle threshold controls for a specific country."""
    thresholds = st.session_state.thresholds[country].copy()

    # Gross Margin % Thresholds
    st.sidebar.subheader('Gross Margin % Thresholds')
    new_gm_l0 = st.sidebar.slider('Gross Margin % L0 More Than', 0, 100, int(thresholds['gm_l0']), key=f'gm_l0_{country}')
    new_gm_l1 = st.sidebar.slider('Gross Margin % L1 More Than', 0, 100, int(thresholds['gm_l1']), key=f'gm_l1_{country}')
    new_gm_l2 = st.sidebar.slider('Gross Margin % L2 More Than', 0, 100, int(thresholds['gm_l2']), key=f'gm_l2_{country}')
    new_gm_l3 = st.sidebar.slider('Gross Margin % L3 More Than', 0, 100, int(thresholds['gm_l3']), key=f'gm_l3_{country}')
    new_gm_l4 = st.sidebar.slider('Gross Margin % L4 More Than', 0, 100, int(thresholds['gm_l4']), key=f'gm_l4_{country}')

    # Deal Score Thresholds
    st.sidebar.subheader('Deal Score Thresholds')
    new_ds_l0 = st.sidebar.slider('Deal Score L0 More Than', 0, 100, int(thresholds['ds_l0']), key=f'ds_l0_{country}')
    new_ds_l1 = st.sidebar.slider('Deal Score L1 More Than', 0, 100, int(thresholds['ds_l1']), key=f'ds_l1_{country}')
    new_ds_l2 = st.sidebar.slider('Deal Score L2 More Than', 0, 100, int(thresholds['ds_l2']), key=f'ds_l2_{country}')
    new_ds_l3 = st.sidebar.slider('Deal Score L3 More Than', 0, 100, int(thresholds['ds_l3']), key=f'ds_l3_{country}')
    new_ds_l4 = st.sidebar.slider('Deal Score L4 More Than', 0, 100, int(thresholds['ds_l4']), key=f'ds_l4_{country}')

    # Deal Size (€) Thresholds
    st.sidebar.subheader('Deal Size (€) Thresholds')
    new_vol_l0 = st.sidebar.number_input('Deal Size (€) L0 Less Than', value=float(thresholds['vol_l0']), key=f'vol_l0_{country}')
    new_vol_l1 = st.sidebar.number_input('Deal Size (€) L1 Less Than', value=float(thresholds['vol_l1']), key=f'vol_l1_{country}')
    new_vol_l2 = st.sidebar.number_input('Deal Size (€) L2 Less Than', value=float(thresholds['vol_l2']), key=f'vol_l2_{country}')
    new_vol_l3 = st.sidebar.number_input('Deal Size (€) L3 Less Than', value=float(thresholds['vol_l3']), key=f'vol_l3_{country}')
    new_vol_l4 = st.sidebar.number_input('Deal Size (€) L4 Less Than', value=float(thresholds['vol_l4']), key=f'vol_l4_{country}')

    # International Finance VP Thresholds
    st.sidebar.subheader('International Finance VP Thresholds')
    new_l5_vol = st.sidebar.number_input('Deal Size (€) L5/L6 More Than', value=float(thresholds['l5_vol']), key=f'l5_vol_{country}')
    new_l5_gm = st.sidebar.number_input('Gross Margin % L5/L6 Less Than', value=float(thresholds['l5_gm']), key=f'l5_gm_{country}')

    # Compile new thresholds from UI inputs
    new_thresholds = {
        'vol_l4': new_vol_l4, 'vol_l3': new_vol_l3, 'vol_l2': new_vol_l2,
        'vol_l1': new_vol_l1, 'vol_l0': new_vol_l0,
        'gm_l4': new_gm_l4, 'gm_l3': new_gm_l3, 'gm_l2': new_gm_l2,
        'gm_l1': new_gm_l1, 'gm_l0': new_gm_l0,
        'ds_l4': new_ds_l4, 'ds_l3': new_ds_l3, 'ds_l2': new_ds_l2,
        'ds_l1': new_ds_l1, 'ds_l0': new_ds_l0,
        'l5_vol': new_l5_vol, 'l5_gm' : new_l5_gm
    }

    # Detect if any threshold has changed
    if any(thresholds[k] != new_thresholds[k] for k in new_thresholds):
        st.session_state.thresholds[country] = new_thresholds
        save_thresholds(st.session_state.thresholds)


def get_applicable_thresholds(country):
    """Retrieve the latest thresholds for a country, considering any temporary changes."""
    # Start with the permanent thresholds
    thresholds = st.session_state.thresholds[country].copy()

    return thresholds

# --- Business Logic Functions ---

def determine_vol_level(vol, thresholds):
    if vol > thresholds['vol_l4']:
        return 'L5'
    elif vol > thresholds['vol_l3']:
        return 'L4'
    elif vol > thresholds['vol_l2']:
        return 'L3'
    elif vol > thresholds['vol_l1']:
        return 'L2'
    elif vol > thresholds['vol_l0']:
        return 'L1'
    return 'L0'

def determine_gm_level(gm, thresholds):
    if gm > thresholds['gm_l0']:
        return 'L0'
    elif gm > thresholds['gm_l1']:
        return 'L1'
    elif gm > thresholds['gm_l2']:
        return 'L2'
    elif gm > thresholds['gm_l3']:
        return 'L3'
    elif gm > thresholds['gm_l4']:
        return 'L4'
    return 'L5'

def determine_ds_level(ds, thresholds):
    if ds > thresholds['ds_l0']:
        return 'L0'
    elif ds > thresholds['ds_l1']:
        return 'L1'
    elif ds > thresholds['ds_l2']:
        return 'L2'
    elif ds > thresholds['ds_l3']:
        return 'L3'
    elif ds > thresholds['ds_l4']:
        return 'L4'
    return 'L5'

def determine_L5_level(vol, gm, thresholds):
    if (vol > thresholds['l5_vol']) and (gm < thresholds['l5_gm']):
        return 'L5'
    else:
        return 'N/A'

def get_level_value(level):
    level_map = {'L0': 0, 'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4, 'L5': 5, 'N/A': -1}
    return level_map.get(level, -1)

# def determine_highest_level(row):
#     """Determine the highest escalation level based on thresholds."""
#     thresholds = get_applicable_thresholds(row['Country'])

#     levels = [
#         ('Deal Score', get_level_value(determine_ds_level(row['DS'], thresholds))),
#         ('Gross Margin %', get_level_value(determine_gm_level(row['GM'], thresholds))),
#         ('Deal size (€)', get_level_value(determine_vol_level(row['Vol'], thresholds))),
#         ('Other Business Rule', get_level_value(row['oBR_level'])),
#         ('International Final VP Rule', get_level_value(determine_L5_level(row['Vol'], row['GM'], thresholds)))
#     ]
#     max_level = max(levels, key=lambda x: x[1])
#     if max_level[0] == 'Other Business Rule' and row.get('FP', False) == True:
#         return 'Below Floor Price'
#     return max_level[0]

def determine_highest_level(row):
    """
    Determine the highest escalation level and all reasons that share that level.
    Returns a string of all reasons that share the maximum level, joined by ' & '.
    'Other Business Rule' is only included if it's the sole maximum reason.
    """
    thresholds = get_applicable_thresholds(row['Country'])
    
    # Define all level checks
    levels = [
        ('Deal Score', get_level_value(determine_ds_level(row['DS'], thresholds))),
        ('Gross Margin %', get_level_value(determine_gm_level(row['GM'], thresholds))),
        ('Deal size (€)', get_level_value(determine_vol_level(row['Vol'], thresholds))),
        ('Other Business Rule', get_level_value(row['oBR_level'])),
        ('International Final VP Rule', get_level_value(determine_L5_level(row['Vol'], row['GM'], thresholds)))
    ]
    
    # Find the maximum level value
    max_level_value = max(level[1] for level in levels)
    
    # Get all reasons that have the maximum level
    max_reasons = [reason for reason, value in levels if value == max_level_value]
    
    # Handle Other Business Rule:
    # - Remove it if there are other max reasons
    # - Keep it only if it's the sole max reason
    if 'Other Business Rule' in max_reasons and len(max_reasons) > 1:
        max_reasons.remove('Other Business Rule')
    
    # Special handling for Floor Price
    if 'Other Business Rule' in max_reasons and row.get('FP', False) == True:
        max_reasons.remove('Other Business Rule')
        max_reasons.append('Below Floor Price')
    
    # Join all maximum level reasons with ' & '
    return ' & '.join(max_reasons)

def get_approval_level(row):
    """Get approval level using the latest thresholds."""
    thresholds = get_applicable_thresholds(row['Country'])
    
    levels = [
        determine_ds_level(row['DS'], thresholds),
        determine_gm_level(row['GM'], thresholds),
        determine_vol_level(row['Vol'], thresholds),
        determine_L5_level(row['Vol'], row['GM'], thresholds),
        row['oBR_level']
    ]
    # Calculate the highest level based on defined hierarchy
    highest_level = max(levels, key=lambda x: get_level_value(x))
    return highest_level

def main():
    # Initialize session state
    initialize_session_state()

    st.title('Escalation Rules Analysis & Simulations')

    if 'previous_threshold_file' not in st.session_state:
        st.session_state.previous_threshold_file = None
    if 'loaded_thresholds' not in st.session_state:
        st.session_state.loaded_thresholds = None
        
    # Sidebar - Configuration
    st.sidebar.header('Configuration')

    # Upload thresholds JSON
    threshold_file = st.sidebar.file_uploader("Upload Thresholds JSON", type=['json'])

# Only load thresholds if the file has changed
    if threshold_file is not None and threshold_file != st.session_state.previous_threshold_file:
        try:
            st.session_state.loaded_thresholds = json.load(threshold_file)
            st.session_state.previous_threshold_file = threshold_file
            save_thresholds(st.session_state.loaded_thresholds)

            st.sidebar.success('Thresholds loaded successfully!')
        except json.JSONDecodeError:
            st.sidebar.error('Error: Invalid JSON file')
        except Exception as e:
            st.sidebar.error(f'Error loading file: {str(e)}')
                
        except Exception as e:
            st.error(f"Error processing uploaded thresholds: {str(e)}")

    # Upload data
    data_file = st.sidebar.file_uploader("Upload Data CSV", type=['csv'])

    if data_file is not None:
        try:
            data = pd.read_csv(data_file)

            # Sidebar Filters
            st.sidebar.header('Filters')
            if 'previous_countries' not in st.session_state:
                st.session_state.previous_countries = []
            
            selected_countries = st.sidebar.multiselect(
                'Select a Country to Adjust Business Rules',
                options=['All'] + list(data['Country'].unique()),
                default='All'
            )
            
            # Check if country selection has changed
            if selected_countries != st.session_state.previous_countries:
                # Reload thresholds from file
                reload_thresholds_from_file()
                st.session_state.previous_countries = selected_countries
                
            # Show threshold controls for selected countries
            if selected_countries and 'All' not in selected_countries:
                st.sidebar.header('Threshold Configuration')
                for country in selected_countries:
                    with st.sidebar.expander(f'Thresholds for {country}'):
                        display_threshold_controls(country)

            # Filter data based on country selection
            if 'All' in selected_countries or not selected_countries:
                filtered_data = data.copy()
            else:
                filtered_data = data[data['Country'].isin(selected_countries)]

            # Apply escalation rules
            filtered_data['Final_Escalation'] = filtered_data.apply(determine_highest_level, axis=1)

            # Additional Visualization Filters
            st.sidebar.header('Visualization Filters')
            vis_selected_countries = st.sidebar.multiselect(
                'Filter by Country',
                options=filtered_data['Country'].unique(),
                default=filtered_data['Country'].unique()
            )
            vis_selected_quote_types = st.sidebar.multiselect(
                'Filter by Quote Type',
                options=filtered_data['quote_type'].unique(),
                default=filtered_data['quote_type'].unique()
            )
            vis_selected_GM = st.sidebar.slider('Validation Check: Gross Margin Not More Than', 0, 1000, 150)
            vis_selected_min_GM = st.sidebar.slider('Validation Check: Gross Margin Not Less Than', -1000, 10, 10)

            vis_selected_quote_status = st.sidebar.multiselect(
                'Filter by Quote Status',
                options=filtered_data['status'].unique(),
                default=filtered_data['status'].unique()
            )

            filtered_data['CreatedDate'] = pd.to_datetime(filtered_data['CreatedDate'])
            filtered_data['MonthYear'] = filtered_data['CreatedDate'].dt.to_period('M')
            
            # Get unique month/years for the slider
            month_years = sorted(filtered_data['MonthYear'].unique())
            min_month = month_years[0]
            max_month = month_years[-1]
            
            # Create month range selector
            selected_months = st.sidebar.select_slider(
                'Select Month Range',
                options=month_years,
                value=(min_month, max_month),
                format_func=lambda x: x.strftime('%B %Y')
            )
    

            # Apply visualization filters
            vis_data = filtered_data[
                (filtered_data['MonthYear'] >= selected_months[0]) &
                (filtered_data['MonthYear'] <= selected_months[1]) &
                (filtered_data['Country'].isin(vis_selected_countries)) &
                (filtered_data['quote_type'].isin(vis_selected_quote_types)) &
                (filtered_data['status'].isin(vis_selected_quote_status)) &
                (filtered_data['GM'] < vis_selected_GM) &
                (filtered_data['GM'] >= vis_selected_min_GM)  

            ]

            # Generate Approval Levels based on latest thresholds
            vis_data['Approval_Level'] = vis_data.apply(get_approval_level, axis=1)

            # --- Visualizations ---
            st.header('Visualizations')
            vis_data = vis_data.rename(columns={'Approval_Level_Numeric': 'Approval Level', 'DS': 'Deal Score', 'Vol': 'Deal Size', 'GM': 'Gross Margin %','QuoteType__c':'Quote Type'})

            # First Row: Approval Level and Escalation Distributions
            col1, col2 = st.columns(2)

            with col1:
                # Pie chart of approval levels
                level_counts = vis_data['Approval_Level'].value_counts()
                fig1 = px.pie(
                    values=level_counts.values,
                    names=level_counts.index,
                    title=f'Distribution of Approval Levels ({len(vis_data)} deals)',
                    category_orders={"names": sorted(level_counts.index)}
                )
                st.plotly_chart(fig1)

            with col2:
                # Pie chart of escalation reasons
                escalation_counts = vis_data['Final_Escalation'].value_counts()
                fig2 = px.pie(
                    values=escalation_counts.values,
                    names=escalation_counts.index,
                    title='Distribution of Escalation Reasons',
                    category_orders={"names": sorted(escalation_counts.index)}

                )
                st.plotly_chart(fig2)

            # Second Row: Volume vs GM colored by Approval Level and Escalation Reason
            col1, col2 = st.columns(2)

            with col1:
                # Scatter plot with Volume vs GM colored by Approval Level
                fig3 = px.scatter(
                    vis_data,
                    x='Deal Size',
                    y='Gross Margin %',
                    color='Approval_Level',
                    title='Deal Size vs Gross Margin by Approval Level',
                    category_orders={"Approval_Level": sorted(vis_data['Approval_Level'].unique())},                    log_x=True
                )
                st.plotly_chart(fig3)

            with col2:
                # Scatter plot with Volume vs GM colored by Escalation Reason
                fig4 = px.scatter(
                    vis_data,
                    x='Deal Size',
                    y='Gross Margin %',
                    color='Final_Escalation',
                    title='Deal Size vs Gross Margin by Escalation Reason',
                    category_orders={"Final_Escalation": sorted(vis_data['Final_Escalation'].unique())},
                    log_x=True
                )
                st.plotly_chart(fig4)

            # Third Row: Approval Level vs Escalation Reason
            fig5 = px.histogram(
                vis_data,
                x='Final_Escalation',
                color='Approval_Level',
                title='Approval Levels by Escalation Reason',
                category_orders={"Approval_Level": sorted(vis_data['Approval_Level'].unique())},
                barmode='group'
            )
            st.plotly_chart(fig5)

            fig6 = px.histogram(
                vis_data,
                x='Deal Score',
                color='Approval_Level',
                title='Deal Score Distribution by Final Approval Level',
                category_orders={"Approval_Level": sorted(vis_data['Approval_Level'].unique())},
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig6)

            fig7 = px.scatter_3d(
            vis_data, 
            x='Deal Score', 
            y='Deal Size', 
            z='Gross Margin %',
            color='Approval_Level',category_orders={"Approval_Level": sorted(vis_data['Approval_Level'].unique())},
            title="Deal Distribution by Approval Level"
        )

            st.plotly_chart(fig7)

            # Cross-tabulation Table
            st.subheader('Approval Level vs Escalation Reason Cross-tabulation')
            crosstab = pd.crosstab(
                vis_data['Approval_Level'],
                vis_data['Final_Escalation'],
                margins=True
            )
            st.dataframe(crosstab)

            # Summary Statistics
            st.subheader('Summary by Approval Level')
            summary_stats = vis_data.groupby('Approval_Level').agg({
                'Deal Size': ['count', 'mean', 'median', 'sum'],
                'Gross Margin %': ['mean', 'median'],
                'Deal Score': ['mean', 'median']
            }).round(0)
            
            # Calculate totals with matching MultiIndex structure
            total_stats = pd.DataFrame([
                [int(vis_data['Deal Size'].count()), 
                round(vis_data['Deal Size'].mean(), 0), 
                round(vis_data['Deal Size'].median(), 0),
                round(vis_data['Deal Size'].sum(), 0),
                round(vis_data['Gross Margin %'].mean(), 0), 
                round(vis_data['Gross Margin %'].median(), 0),
                round(vis_data['Deal Score'].mean(), 0), 
                round(vis_data['Deal Score'].median(), 0)]
            ], columns=summary_stats.columns, index=pd.Index(['Total']))

            # Combine the grouped stats with the total row
            summary_stats = pd.concat([summary_stats, total_stats])
            summary_stats = summary_stats.astype(int)

            def format_eur(x):
                return f"€{x:,.0f}".replace(',', '.')
            
            summary_stats.columns = [
                'Count', 'Avg Volume', 'Median Volume', 'Total Sales',
                'Avg Gross Margin %', 'Median Gross Margin %',
                'Avg Deal Score', 'Median Deal Score'
            ]

            summary_stats['Total Sales'] = summary_stats['Total Sales'].apply(format_eur)
            summary_stats['Avg Volume'] = summary_stats['Avg Volume'].apply(format_eur)
            summary_stats['Median Volume'] = summary_stats['Median Volume'].apply(format_eur)
                                                                                      
            # Create a styled dataframe
            styled_df = summary_stats.style.set_properties(**{
               # 'background-color': 'white',
                #'color': 'black',
                'border-color': 'black',
                'border-style': 'solid',
                'border-width': '1px'
            })

            # Highlight the total row
            styled_df = styled_df.set_properties(**{
                'background-color': '#79cac1',
                'font-weight': 'bold'
            }, subset=pd.IndexSlice['Total', :])
            st.dataframe(styled_df)

            def create_correlation_analysis(df):

                st.header("Correlation Analysis")
                
                # Prepare numerical data
                df['Approval_Level_Numeric'] = df['Approval_Level'].apply(get_level_value)
                df = df.rename(columns={'Approval_Level_Numeric': 'Approval Level', 'DS': 'Deal Score', 'Vol': 'Deal Size', 'GM': 'Gross Margin %','QuoteType__c':'Quote Type'})

                numeric_cols = ['Deal Score', 'Deal Size', 'Approval Level', 'Gross Margin %']
                numeric_data = df[numeric_cols]

                # 1. Scatter plots with regression lines for top correlations
                plt.rcParams.update({'font.size': 8}) 
                st.subheader("Scatter Plots for Key Relationships")
                
                targets = ['Approval Level', 'Gross Margin %']
                features = ['Deal Score', 'Deal Size']

                fig = make_subplots(rows=2, cols=2, subplot_titles=[f'{feature} vs {target}' for target in targets for feature in features])

                for i, target in enumerate(targets):
                    for j, feature in enumerate(features):
                        # Remove NaN values
                        valid_data = df[[feature, target]].dropna()
                        
                        # Calculate correlation
                        corr, _ = pearsonr(valid_data[feature], valid_data[target])
                        
                        # Create scatter plot
                        scatter = go.Scatter(
                            x=valid_data[feature],
                            y=valid_data[target],
                            mode='markers',
                            name=f'{feature} vs {target}',
                            marker=dict(opacity=0.5),
                            hovertemplate=f'{feature}: %{{x}}<br>{target}: %{{y}}<extra></extra>'
                        )
                        
                        # Calculate regression line
                        z = np.polyfit(valid_data[feature], valid_data[target], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(valid_data[feature].min(), valid_data[feature].max(), 100)
                        
                        # Create regression line
                        line = go.Scatter(
                            x=x_range,
                            y=p(x_range),
                            mode='lines',
                            name='Regression Line',
                            line=dict(color='red'),
                            hoverinfo='skip'
                        )
                        
                        # Add traces to subplot
                        fig.add_trace(scatter, row=i+1, col=j+1)
                        fig.add_trace(line, row=i+1, col=j+1)
                        
                        # Add correlation annotation
                        fig.add_annotation(
                            xref=f'x{i*2+j+1}', yref=f'y{i*2+j+1}',
                            x=0.05, y=0.95, xanchor='left', yanchor='top',
                            text=f'Correlation: {corr:.2f}',
                            showarrow=False,
                            bgcolor='rgba(255,255,255,0.8)',
                            bordercolor='rgba(0,0,0,0.3)',
                            borderwidth=1
                        )

                # Update layout
                fig.update_layout(height=800, width=1000, title_text="Key Feature Relationships")
                fig.update_xaxes(title_text="Deal Score", row=1, col=1)
                fig.update_xaxes(title_text="Deal Size", row=1, col=2)
                fig.update_xaxes(title_text="Deal Score", row=2, col=1)
                fig.update_xaxes(title_text="Deal Size", row=2, col=2)
                fig.update_yaxes(title_text="Approval Level", row=1, col=1)
                fig.update_yaxes(title_text="Approval Level", row=1, col=2)
                fig.update_yaxes(title_text="Gross Margin %", row=2, col=1)
                fig.update_yaxes(title_text="Gross Margin %", row=2, col=2)

                # Display the plot
                st.plotly_chart(fig, use_container_width=True)

                def create_correlation_heatmap(numeric_data):
                    """Create an interactive correlation heatmap using Plotly"""
                    corr_matrix = numeric_data.corr()
                    
                    # # Create mask for upper triangle
                    # mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    
                    # # Convert to array and mask upper triangle
                    # z = np.array(corr_matrix)
                    # z[mask] = np.nan
                    
                    # Create hover text
                    hover_text = [[f'{col1} vs {col2}<br>Correlation: {val:.2f}'
                                for col2, val in zip(corr_matrix.columns, row)]
                                for col1, row in zip(corr_matrix.index, corr_matrix.values)]
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        zmin=-1,
                        zmax=1,
                        text=np.around(corr_matrix, decimals=2),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        hoverongaps=False,
                        hoverinfo='text',
                        hovertext=hover_text,
                        colorscale='RdBu',
                        colorbar=dict(
                            title='Correlation',
                            titleside='right',
                            thickness=15,
                            len=0.5,
                            x=1.1
                        )
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=dict(
                            text='Feature Correlation Matrix',
                            x=0.5,
                            y=0.95,
                            font=dict(size=16)
                        ),
                        width=600,
                        height=500,
                        xaxis_showgrid=False,
                        yaxis_showgrid=False,
                        xaxis_title=None,
                        yaxis_title=None,
                        showlegend=False,
                        xaxis={'side': 'bottom'}
                    )
                    
                    return fig, corr_matrix
                
                fig, corr_matrix = create_correlation_heatmap(numeric_data)
                st.plotly_chart(fig, use_container_width=True)

                gm_correlations = corr_matrix['Gross Margin %'].drop('Gross Margin %')
                approval_correlations = corr_matrix['Approval Level'].drop('Approval Level')
                
                # Function to describe correlation strength
                def describe_correlation(value):
                    abs_val = abs(value)
                    if abs_val >= 0.7:
                        return "very strong"
                    elif abs_val >= 0.5:
                        return "strong"
                    elif abs_val >= 0.3:
                        return "moderate"
                    elif abs_val >= 0.1:
                        return "weak"
                    else:
                        return "very weak"
                
                # Analyze GM correlations
                gm_insights = []
                for feature, corr in gm_correlations.items():
                    if abs(corr) >= 0.1:  # Only include meaningful correlations
                        direction = "positive" if corr > 0 else "negative"
                        strength = describe_correlation(corr)
                        gm_insights.append(f"- {feature} shows a {strength} {direction} correlation ({corr:.2f}) with Gross Margin")
                
                # Analyze Approval Level correlations
                approval_insights = []
                for feature, corr in approval_correlations.items():
                    if abs(corr) >= 0.1:  # Only include meaningful correlations
                        direction = "positive" if corr > 0 else "negative"
                        strength = describe_correlation(corr)
                        approval_insights.append(f"- {feature} shows a {strength} {direction} correlation ({corr:.2f}) with Approval Level")
                
                #return gm_insights, approval_insights


                # Add correlation insights
                st.subheader("Key Correlation Insights")
                #gm_insights, approval_insights = analyze_correlations(corr_matrix)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Gross Margin Correlations:**")
                    for insight in gm_insights:
                        st.write(insight)
                        
                with col2:
                    st.write("**Approval Level Correlations:**")
                    for insight in approval_insights:
                        st.write(insight)
                                
                # 3. Categorical Analysis
                plt.rcParams.update({'font.size': 9}) 

                st.subheader("Target Variables by Categories")

                categorical_cols = ['Cluster', 'Country', 'Quote Type']
                targets_display = ['Approval Level', 'Gross Margin %']
                colors = ['#79cac1', '#3b9153', '#009dd2', '#f78e82', '#d774ae', '#69008c']

                for target in targets_display:
                    # Create subplot with 3 columns
                    fig = make_subplots(rows=1, cols=3, subplot_titles=categorical_cols)

                    for i, cat_col in enumerate(categorical_cols):
                        # Calculate statistics for each category
                        grouped_data = df.groupby(cat_col)[target]
                        means = grouped_data.mean().reset_index()
                        sems = grouped_data.sem().reset_index()
                        
                        # Merge means and sems
                        plot_data = means.merge(sems, on=cat_col, suffixes=('_mean', '_sem'))
                        
                        # Create the bar trace
                        trace = go.Bar(
                            x=plot_data[cat_col],
                            y=plot_data[f"{target}_mean"],
                            error_y=dict(type='data', array=plot_data[f"{target}_sem"]),
                            name=cat_col,
                            text=plot_data[f"{target}_mean"].round(2),
                            textposition='outside',
                            marker_color=colors[i]
                        )
                        
                        # Add the trace to the appropriate subplot
                        fig.add_trace(trace, row=1, col=i+1)
                        
                        # Update layout for each subplot
                        fig.update_xaxes(title_text=cat_col, row=1, col=i+1)
                        fig.update_yaxes(title_text=target.replace('_Numeric', ''), row=1, col=i+1)
                        
                        # Customize y-axis for Approval Level
                        if target == 'Approval Level':
                            fig.update_yaxes(tickmode='array', 
                                            tickvals=list(range(-1, 6)),
                                            ticktext=['N/A', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5'],
                                            row=1, col=i+1)

                    # Update overall layout
                    fig.update_layout(
                        height=500, 
                        width=1000,
                        title_text=f"{target.replace('_Numeric', '')} Distribution by Categories",
                        showlegend=False
                    )

                    # Display the plot
                    st.plotly_chart(fig, use_container_width=True)

                # st.subheader("Target Variables by Categories")
                
                # categorical_cols = ['Cluster', 'Country', 'Quote Type']
                # targets_display = ['Approval Level', 'Gross Margin %']

                # for target in targets_display:
                #     fig, axes = plt.subplots(1, len(categorical_cols), figsize=(20, 6))
                #     title = f'{target.replace("_Numeric", "")} Distribution by Categories'
                #     fig.suptitle(title, fontsize=16)
                    
                #     for i, cat_col in enumerate(categorical_cols):
                #         # Calculate statistics for each category
                #         grouped_data = df.groupby(cat_col)[target]
                #         means = grouped_data.mean()
                #         sems = grouped_data.sem()  # Standard error of the mean
                        
                #         # Create the bar plot without error bars first
                #         ax = axes[i]
                #         bars = ax.bar(range(len(means)), means)
                        
                #         # Add error bars manually
                #         ax.errorbar(range(len(means)), means, yerr=sems, fmt='none', color='black', capsize=5)
                        
                #         # Customize the plot
                #         ax.set_title(f'{target.replace("_Numeric", "")} by {cat_col}')
                #         ax.set_xticks(range(len(means)))
                #         ax.set_xticklabels(means.index, rotation=45, ha='right')
                        
                #         # Add custom y-ticks for Approval_Level
                #         if target == 'Approval Level':
                #             ax.set_yticks(range(-1, 6))
                #             ax.set_yticklabels(['N/A', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5'])
                
                #     plt.tight_layout()
                #     st.pyplot(fig)
                #     plt.close()
                
                # 4. Time Series Analysis
                st.subheader("Time Series Trends")
                if hasattr(df['MonthYear'].dtype, 'freq'):  # Check if it's Period
                    df['MonthYear'] = df['MonthYear'].dt.to_timestamp()
                # Convert MonthYear to datetime if not already
                #df['MonthYear'] = pd.to_datetime(df['MonthYear'])
                
                # # Plot time series for both target variables
                # fig, axes = plt.subplots(2, 1, figsize=(15, 10))
                
                # for i, target in enumerate(targets):
                #     monthly_avg = df.groupby('MonthYear')[target].mean()
                    
                #     axes[i].plot(monthly_avg.index, monthly_avg.values, marker='o')
                #     axes[i].set_title(f'{target} Trend Over Time')
                #     axes[i].set_xlabel('Month-Year')
                #     axes[i].set_ylabel(target)
                #     axes[i].tick_params(axis='x', rotation=45)
                
                # plt.tight_layout()
                # st.pyplot(fig)
                # plt.close()

                fig = make_subplots(rows=2, cols=1, subplot_titles=[f'{target} Trend Over Time' for target in targets])

                for i, target in enumerate(targets):
                    # Calculate monthly average
                    monthly_avg = df.groupby('MonthYear')[target].mean().reset_index()
                    
                    # Create line plot
                    trace = go.Scatter(
                        x=monthly_avg['MonthYear'],
                        y=monthly_avg[target],
                        mode='lines+markers',
                        name=target,
                        hovertemplate='%{x}<br>' + f'{target}: ' + '%{y:.2f}<extra></extra>'
                    )
                    
                    # Add trace to subplot
                    fig.add_trace(trace, row=i+1, col=1)
                    
                    # Update y-axis label
                    fig.update_yaxes(title_text=target, row=i+1, col=1)

                # Update layout
                fig.update_layout(
                    height=800, 
                    width=1000, 
                    title_text="Target Variables Trend Over Time",
                    showlegend=False
                )

                # Update x-axis properties
                fig.update_xaxes(
                    title_text="Month-Year",
                    tickangle=45,
                    tickformat="%b %Y"
                )

                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                # fig = sns.pairplot(df, hue='quote_type', vars=['Deal Score', 'Deal Size', 'Gross Margin %'])
                # st.pyplot(fig)
                # plt.close()

            create_correlation_analysis(vis_data)
            
            vis_data = vis_data.rename(columns={'Approval_Level_Numeric': 'Approval Level', 'DS': 'Deal Score', 'Vol': 'Deal Size', 'GM': 'Gross Margin %','QuoteType__c':'Quote Type'})

            def compare_floor_price_quotes(df):
                
                st.write("### Below Hard Floor Price Quotes Analysis")

                # Create comparison summary
                numerical_cols = ['Deal Score', 'Deal Size', 'Gross Margin %']
                
                # 1. Summary Statistics Comparison
                summary_stats = df.groupby('FP')[numerical_cols].agg([
                    'mean', 'median', 'std', 'count'
                ]).round(2)
                
                        # 2. Distribution Comparisons
                col1, col2, col3 = st.columns(3)
                
                # Create and display plots for all features
                for idx, (col_name, plot_col) in enumerate(zip(numerical_cols, [col1, col2, col3])):
                    if col_name not in df.columns:
                        plot_col.warning(f"Column {col_name} not found in data")
                        continue
                        
                    # Calculate averages
                    avg_by_group = df.groupby('FP')[col_name].mean().reset_index()
                    
                    # Create bar plot
                    fig_bar = px.bar(avg_by_group, 
                                    x='FP',
                                    y=col_name,
                                    title=f'Average {col_name}',
                                    text=avg_by_group[col_name].round(0))
                    
                    fig_bar.update_traces(textposition='outside',
                                        textfont=dict(size=12))
                    
                    fig_bar.update_layout(
                        xaxis_title="Below Floor Price",
                        yaxis_title=f"Average {col_name}",
                        bargap=0.3,
                        showlegend=False,
                        yaxis_range=[0, avg_by_group[col_name].max() * 1.1],
                        height=400,  # Set consistent height
                        width=400,   # Set consistent width
                        margin=dict(t=50, l=50, r=20, b=50)  # Adjust margins for better fit
                    )
                

                    plot_col.plotly_chart(fig_bar, use_container_width=True)
                
                # 3. Categorical Feature Analysis
                categorical_cols = ['Approval Level', 'Quote Type', 'Country', 'status']
                
                for cat_col in categorical_cols:
                    # Calculate absolute counts for each category
                    counts = pd.crosstab(df[cat_col], 
                                        df['FP'])
                    
                    # Create bar plot
                    fig_bar = px.bar(counts,
                                    title=f'{cat_col} Distribution by Floor Price Flag',
                                    labels={'value': 'Number of Quotes',
                                        'Below Hard Floor Price': 'Below Floor Price'},
                                    barmode='group')
                    
                    # Update layout for better readability
                    fig_bar.update_layout(
                        xaxis_title=cat_col,
                        yaxis_title="Number of Quotes",
                        legend_title="Below Floor Price")
                    
                    fig_bar.for_each_trace(lambda trace: trace.update(visible="legendonly")
                        if trace.name in ['False'] else ())
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # 4. Feature Importance Analysis using Random Forest
                def prepare_data_for_modeling(df):
                    X = df.copy()
                    y = X.pop('FP')
                    
                    # Encode categorical variables
                    le = LabelEncoder()
                    for col in categorical_cols:
                        X[col] = le.fit_transform(X[col])
                        
                    return X[numerical_cols + categorical_cols], y
                
                X, y = prepare_data_for_modeling(df)
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X, y)
                
                # Create feature importance plot
                importance_df = pd.DataFrame({
                    'Feature': numerical_cols + categorical_cols,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_importance = px.bar(importance_df,
                                    x='Feature',
                                    y='Importance',
                                    title='Feature Importance for Below Floor Price Prediction')
                st.plotly_chart(fig_importance, use_container_width=True)

                
                return summary_stats, importance_df

            # Add button to trigger analysis
            if st.button("Below Hard Floor Price Quotes Analysis"):

                summary_stats,  importance_df = compare_floor_price_quotes(vis_data)

            def segmentation_tool(vis_data):
                st.title("Quote Segmentation Tool")
                st.write("""
                         This tool combines user control with automated analysis by allowing manual weight assignment to features and desired segment sizes, creating a weighted scoring system based on normalized rankings. 
                         After segmenting the data according to these user-defined parameters, it employs a classification model to discover and extract the underlying rules that characterize each segment, 
                         making the complex segmentation logic interpretable and actionable.
                         """)
                st.write("Set Feature Priority Weights:")
                # Create two columns for weights
                col1, col2, col3 = st.columns(3)

                with col1:
                    weight_gm = st.number_input("Weight for Gross Margin %", value=0.6, format="%.2f")
                with col2:
                    weight_ds = st.number_input("Weight for Deal Size", value=0.25, format="%.2f")
                with col3:
                    weight_dscore = st.number_input("Weight for Deal Score", value=0.15, format="%.2f")

                st.write("Define Desired Segment Sizes by Quantile Thresholds :")

                # Create two columns for segment quantiles
                col3, col4 = st.columns(2)

                with col3:
                    quantile_a = st.number_input("Quantile for Segment 5", value=0.05, format="%.2f", min_value=0.01)
                    quantile_b = st.number_input("Quantile for Segment 4", value=0.20, format="%.2f", min_value=quantile_a + 0.01)

                with col4:
                    quantile_c = st.number_input("Quantile for Segment 3", value=0.50, format="%.2f", min_value=quantile_b + 0.01)
                    quantile_d = st.number_input("Quantile for Segment 2", value=0.80, format="%.2f", min_value=quantile_c + 0.01)

                # Step to ensure valid weights and quantiles
                if weight_gm + weight_ds + weight_dscore != 1.0:
                    st.error("The sum of weights must equal to 1. Please adjust the values.")
                    st.stop()

                # Step to ensure valid quantiles
                if not (0 < quantile_a < quantile_b < quantile_c < quantile_d < 1):
                    st.error("Quantiles must be between (0 and 1) and in increasing order.")
                    st.stop()
                
                # Step 1: Rank each parameter
                vis_data['GM_rank'] = vis_data['Gross Margin %'].rank(ascending=True)
                vis_data['DS_rank'] = vis_data['Deal Size'].rank(ascending=False)
                vis_data['DScore_rank'] = vis_data['Deal Score'].rank(ascending=True)

                # Step 2: Normalize ranks to a scale of 0 to 100
                vis_data['GM_rank_norm'] = (vis_data['GM_rank'] / vis_data['GM_rank'].max()) * 100
                vis_data['DS_rank_norm'] = (vis_data['DS_rank'] / vis_data['DS_rank'].max()) * 100
                vis_data['DScore_rank_norm'] = (vis_data['DScore_rank'] / vis_data['DScore_rank'].max()) * 100

                # Step 3: Calculate the final score for segmentation
                vis_data['Score'] = (weight_ds * vis_data['DS_rank_norm'] + 
                            weight_dscore * vis_data['DScore_rank_norm'] + 
                            weight_gm * vis_data['GM_rank_norm'])

                # Step 4: Define Segment 4oundaries based on user-defined quantiles
                segment_boundaries = {
                    'Segment 5': vis_data['Score'].quantile(quantile_a),
                    'Segment 4': vis_data['Score'].quantile(quantile_b),
                    'Segment 3': vis_data['Score'].quantile(quantile_c),
                    'Segment 2': vis_data['Score'].quantile(quantile_d),
                    'Segment 1': vis_data['Score'].max()   # Last segment includes everything above Segment 2
                }

                # Assign segments based on score
                def assign_segment(score):
                    if score <= segment_boundaries['Segment 5']:
                        return 'Segment 5'
                    elif score <= segment_boundaries['Segment 4']:
                        return 'Segment 4'
                    elif score <= segment_boundaries['Segment 3']:
                        return 'Segment 3'
                    elif score <= segment_boundaries['Segment 2']:
                        return 'Segment 2'
                    else:
                        return 'Segment 1'

                vis_data['Segment'] = vis_data['Score'].apply(assign_segment)
                
                col1, col2 = st.columns(2)

                with col1:
                    level_counts = vis_data['Segment'].value_counts()
                    fig1 = px.pie(
                        values=level_counts.values,
                        names=level_counts.index,
                        title=f'Distribution of Calculated Segments ({len(vis_data)} deals)',
                        category_orders={"names": sorted(level_counts.index, reverse=False)}
                    )
                    st.plotly_chart(fig1)

                # Create 3D scatter plot
                with col2:
                    fig2 = px.scatter_3d(
                        vis_data, 
                        x='Deal Score', 
                        y='Deal Size', 
                        z='Gross Margin %',
                        color='Segment',category_orders={"Segment": sorted(vis_data['Segment'].unique(), reverse=False)},

                        title="Deal Distribution by Calculated Segments"
                    )
                    st.plotly_chart(fig2)

                # Display the segmented DataFrame in Streamlit
                st.write(vis_data[['Gross Margin %', 'Deal Size', 'Deal Score', 
                            'GM_rank_norm', 'DS_rank_norm', 'DScore_rank_norm', 
                            'Score', 'Segment', 'Country', 'Approval Level']])

            def analyze_approval_rules(df):


                label_encoder = LabelEncoder()
                df['Segment_encoded'] = label_encoder.fit_transform(df['Segment'])

                # Features and target
                X = df[['Gross Margin %', 'Deal Size', 'Deal Score']]
                y = df['Segment_encoded']
                
                max_depth = st.number_input("Tree Complexity Level - Set how many layers of rules to create. More layers mean more precise but complex patterns", value=3,  min_value=1)

                # Train a Decision Tree Classifier
                clf = DecisionTreeClassifier(random_state=42,max_depth=max_depth)
                clf.fit(X, y)

                # Visualize the Decision Tree
                fig, ax = plt.subplots(figsize=(15, 10))
                #plt.figure(figsize=(12,8))
                plot_tree(clf,
                        feature_names=X.columns,
                        class_names=label_encoder.classes_,
                        filled=True,
                        rounded=True)
                plt.title("Decision Tree for Segment Classification")
                st.pyplot(fig)
                plt.show()

                # Streamlit UI for user input
                st.title("Decision Rules for Quote Segmentation")

                # 3. Train decision tree (with controlled depth for interpretable rules)
                #dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2,random_state=42)
                # dt = DecisionTreeClassifier(random_state=42,max_depth=5)#, min_samples_leaf=10, min_samples_split=10)

                # try:
                #     dt.fit(X, y)
                #     #st.success("Decision tree fitted successfully")
                # except Exception as e:
                #     st.error(f"Error fitting decision tree: {str(e)}")
                #     return None, None

                # try:
                #     #probs = dt.predict_proba(X)
                #     st.write(f"Number of leaves: {dt.get_n_leaves()}")
                #     #st.write("Prediction probabilities shape:", probs.shape)
                #     #st.write("Unique predicted classes:", np.unique(dt.predict(X)))
                # except Exception as e:
                #     st.error(f"Error in predictions: {str(e)}")
                #     return None, None
                
                # # Display decision tree visualization
                # st.subheader("Decision Tree Visualization")
                # try:
                #     fig, ax = plt.subplots(figsize=(15, 10))
                #     plot_tree(dt, 
                #             feature_names=['Deal Score', 'Deal Size', 'Gross Margin %'],
                #             #class_names=['Unsuccessful', 'Successful'],
                #             filled=True, 
                #             rounded=True,
                #             fontsize=10)
                #     st.pyplot(fig)
                # except Exception as e:
                #     st.error(f"Error plotting decision tree: {str(e)}")


                def recurse(tree_, feature_name, node, path, paths):
                    if tree_.feature[node] != -2:
                        # Get feature index and ensure it's within bounds
                        feature_idx = tree_.feature[node]
                        if feature_idx < len(feature_name):
                            name = feature_name[feature_idx]
                            threshold = tree_.threshold[node]
                            p1, p2 = list(path), list(path)
                            
                            # Check if there are existing rules to group
                            last_feature = None
                            if path and isinstance(path[-1], str):
                                last_rule = path[-1].strip('()')
                                if '<=' in last_rule or '>' in last_rule:
                                    last_feature = last_rule.split()[0]
                            
                            # Group rules with the same feature
                            if last_feature == name:
                                p1.pop()
                                p2.pop()
                                prev_rule = path[-1].strip('()')
                                p1 += [f"({prev_rule} AND {name} <= {threshold:.2f})"]
                                p2 += [f"({prev_rule} AND {name} > {threshold:.2f})"]
                            else:
                                p1 += [f"({name} <= {threshold:.2f})"]
                                p2 += [f"({name} > {threshold:.2f})"]
                            
                            recurse(tree_, feature_name, tree_.children_left[node], p1, paths)
                            recurse(tree_, feature_name, tree_.children_right[node], p2, paths)
                    else:
                        value = tree_.value[node]
                        samples = tree_.n_node_samples[node]
                        path += [(value, samples)]
                        paths += [path]

                def extract_rules(tree_, feature_name, class_names):
                    paths = []
                    recurse(tree_, feature_name, 0, [], paths)
                    
                    rules = []
                    for path in paths:
                        try:
                            value = path[-1][0]
                            # Ensure value is in the correct format for probability calculation
                            if isinstance(value, np.ndarray):
                                total_samples = float(np.sum(value))
                                if total_samples > 0:
                                    # Get probabilities for each class
                                    probs = value.flatten() / total_samples
                                    
                                    # Create the rule string
                                    rule = "IF "
                                    for i, p in enumerate(path[:-1]):
                                        if i > 0:
                                            rule += " AND "
                                        rule += str(p)
                                    
                                    # Get the dominant class
                                    dominant_class_idx = np.argmax(probs)
                                    prob = probs[dominant_class_idx]
                                    
                                    rule += f" THEN {class_names[dominant_class_idx]} (Probability: {prob:.2f})"
                                    rules.append(rule)
                        except Exception as e:
                            print(f"Error processing path: {e}")
                            continue
                    
                    return rules
                feature_names = list(X.columns)
                class_names = label_encoder.classes_

                # Get rules from the decision tree
                rules = extract_rules(clf.tree_, feature_name=feature_names,
                                class_names=class_names)

                # User input for segment selection
                selected_segment = st.selectbox("Select a Segment", class_names)

                # Display rules for the selected segment
                st.write(f"Rules for {selected_segment}:")

                for rule in rules:
                    if selected_segment in rule:
                        st.text(rule)

                # # Get feature names and class names
                # feature_names = X.columns.tolist()
                # class_names = ['Unsuccessful', 'Successful']

                # # Get the rules for unsuccessful leaves
                # unsuccessful_rules = get_rules(dt, feature_names, class_names)

                # # Print the rules
                # st.write("Rules leading to not Accepted/Approved leaves:")
                # for i, rule in enumerate(unsuccessful_rules, 1):
                #     st.write(f"{i}. {rule}")

                return clf, X       
                    
            def create_rule_based_recommendations(df):
                """Main function to create and display rules"""
                st.title("Segmentation Rules Analysis")
                
                #df['is_successful'] = df['status'].isin(['Approved','Accepted'])
                df = df.loc[df['status'].isin(['Approved','Accepted', 'Rejected','Declined'])]

                df = df.rename(columns={'Approval_Level_Numeric': 'Approval Level', 'DS': 'Deal Score', 'Vol': 'Deal Size', 'GM': 'Gross Margin %','QuoteType__c':'Quote Type'})

                #df = df[['Deal Score', 'Deal Size', 'Gross Margin %','is_successful']].dropna()
                df = df[['Deal Score', 'Deal Size', 'Gross Margin %','Segment']].dropna()

                # Get rules and decision tree
                dt, X = analyze_approval_rules(df)
                
                if dt is None:
                    st.error("Could not create decision tree. Please check the data.")
                    return

                # # Create scatter plot of deals colored by zones
                # st.subheader("Deal Distribution")
                # try:
                #     probs = dt.predict_proba(X)
                #     df['predicted_success'] = probs[:, 1]  # Probability of success
                    
                #     # Create custom zones based on predicted probabilities
                #     df['zone'] = pd.cut(
                #         df['predicted_success'],
                #         bins=[-np.inf, 0.3, 0.6, 0.8, np.inf],
                #         labels=['High Risk', 'Warning', 'Moderate', 'Safe']
                #     )
                    
                    # Show distribution of zones
                    #st.write("Distribution of deals across zones:")
                   # st.write(df['zone'].value_counts())
                    
                    # # Create 3D scatter plot
                    # fig = px.scatter_3d(
                    #     df, 
                    #     x='Deal Score', 
                    #     y='Deal Size', 
                    #     z='Gross Margin %',
                    #     color='zone',
                    #     color_discrete_map={
                    #         'Safe': 'rgba(0, 255, 0, 0.3)',
                    #         'Moderate': 'blue',
                    #         'Warning': 'orange',
                    #         'High Risk': 'red'
                    #     },
                    #     title="Deal Distribution by Risk Zones"
                    # )
                    # fig.for_each_trace(lambda trace: trace.update(visible="legendonly") 
                    #     if trace.name in ['Safe'] else ())
                    # st.plotly_chart(fig)
                    
                # except Exception as e:
                #     st.error(f"Error creating scatter plot: {str(e)}")

                # Show some basic statistics for each zone
                # try:
                #     st.subheader("Zone Statistics")
                #     table_data = []

                #     for zone in df['zone'].unique():
                #         zone_data = df[df['zone'] == zone]
                #         table_data.append({
                #             'Zone': zone,
                #             'Number of deals': len(zone_data),
                #             'Predicted Success Prob From': f"{zone_data['predicted_success'].min():.2f}",
                #             'Predicted Success Prob To': f"{zone_data['predicted_success'].max():.2f}",
                #             'Average Gross Margin %': f"{zone_data['Gross Margin %'].mean():.2f}",
                #             'Average Deal Score': f"{zone_data['Deal Score'].mean():.2f}",
                #             'Average Deal Size': f"{zone_data['Deal Size'].mean():.2f}"
                #         })
                    
                #     # Create a DataFrame from the table_data
                #     table_df = pd.DataFrame(table_data)

                #     # Display the table using Streamlit
                #     st.table(table_df)
                #     # st.subheader("Zone Statistics")
                #     # for zone in df['zone'].unique():
                #     #     zone_data = df[df['zone'] == zone]
                #     #     st.write(f"\n**{zone} Zone:**")
                #     #     st.write(f"Number of deals: {len(zone_data)}")
                #     #     st.write(f"Average GM: {zone_data['GM'].mean():.2f}")
                #     #     st.write(f"Average DS: {zone_data['DS'].mean():.2f}")
                #     #     st.write(f"Average Vol: {zone_data['Vol'].mean():.2f}")
                # except Exception as e:
                #     st.error(f"Error calculating zone statistics: {str(e)}")

            def deal_classification_model(df):
                np.random.seed(42)
                X = df[['Gross Margin %', 'Deal Size', 'Deal Score', 'Country', 'FP']]
                y = df['Approval_Level']

                le_country = LabelEncoder()
                le_fp = LabelEncoder()
                X_encoded = X.copy()
                X_encoded['Country'] = le_country.fit_transform(X['Country'])
                X_encoded['FP'] = le_fp.fit_transform(X['FP'])

                # Scale numerical features
                scaler = StandardScaler()
                numerical_cols = ['Gross Margin %', 'Deal Size', 'Deal Score']
                X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

                # Train the model
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_encoded)

                # Find misclassified samples
                misclassified_mask = y_pred != y
                misclassified_df = df[misclassified_mask].copy()
                misclassified_df['Predicted Approval Level'] = y_pred[misclassified_mask]

                color_map = {
                    'L0': '#79cac1',  
                    'L1': '#3b9153',  
                    'L2': '#009dd2',  
                    'L3': '#f78e82',  
                    'L4': '#d774ae',
                    'L5':'#69008c'
                }

                # Create a list of all possible levels
                all_levels = sorted(list(set(df['Approval_Level'].unique()) | set(misclassified_df['Predicted Approval Level'].unique())))

                # Streamlit app
                st.title('Quote Classification Analysis Tool')
                st.write(
                    """This tool uses a classification model to identify quotes that exhibit characteristics more typical of different approval levels than their assigned ones. 
                    By analyzing the model's predictions, we can surface quotes that may be "borderline" or share strong similarities with other categories.
                    """
                )
                # Create 3D scatter plots
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader('Misclassified Quotes - Actual Approval Levels')
                    fig1 = px.scatter_3d(
                        misclassified_df,
                        x='Gross Margin %',
                        y='Deal Size',
                        z='Deal Score',
                        color='Approval_Level',
                        color_discrete_map=color_map,
                        category_orders={'Approval_Level': all_levels},
                        title='Actual Approval Levels',
                        labels={'Approval_Level': 'Actual Approval Level'}
                    )
                    fig1.update_layout(height=700)
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    st.subheader('Misclassified Quotes - Predicted Approval Levels')
                    fig2 = px.scatter_3d(
                        misclassified_df,
                        x='Gross Margin %',
                        y='Deal Size',
                        z='Deal Score',
                        color='Predicted Approval Level',
                        color_discrete_map=color_map,
                        category_orders={'Predicted Approval Level': all_levels},
                        title='Predicted Approval Levels',
                        labels={'Predicted Approval Level': 'Predicted Approval Level'}
                    )
                    fig2.update_layout(height=700)
                    st.plotly_chart(fig2, use_container_width=True)

                # Display model performance metrics
                st.write('Model Performance:')
                accuracy = (y_pred == y).mean()
                st.write(f'Overall Accuracy: {accuracy:.2%}')
                st.write(f'Number of misclassified quotes: {len(misclassified_df)}')

                 # Display misclassified samples
                st.header('Misclassified Quotes')
                st.dataframe(misclassified_df[['Gross Margin %', 'Deal Size', 'Deal Score', 'Country', 'FP', 'Approval_Level','Predicted Approval Level']])

            deal_classification_model(vis_data)
            segmentation_tool(vis_data)
            create_rule_based_recommendations(vis_data)

            # Raw Data Table
            st.header('Raw Data')
            st.dataframe(vis_data)


            def prepare_export_data():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'thresholds_{timestamp}.json'
                json_data = json.dumps(st.session_state.thresholds, indent=4)
                st.session_state.export_data = json_data
                st.session_state.export_filename = filename
            
            st.sidebar.info("You can export changes made in country thresholds.")

            # Export button
            if st.sidebar.button('Prepare Threshold JSON Export'):
                prepare_export_data()
                st.sidebar.success("Export data prepared. Click 'Download Thresholds' to download.")

            # Download button (always visible)
            if 'export_data' in st.session_state and 'export_filename' in st.session_state:
                st.sidebar.download_button(
                    label="Download Thresholds",
                    data=st.session_state.export_data,
                    file_name=st.session_state.export_filename,
                    mime="application/json"
                )
            else:
                st.sidebar.info("Click 'Prepare Threshold JSON Export' to generate the export file.")
        except Exception as e:
            st.error(f"Error processing data file: {str(e)}")

if __name__ == '__main__':
    main()
