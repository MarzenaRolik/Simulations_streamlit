import streamlit as st
import pandas as pd
import plotly.express as px
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
            vis_selected_min_GM = st.sidebar.slider('Validation Check: Gross Margin Not Less Than', -1000, 0, 0)

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
                    x='Vol',
                    y='GM',
                    color='Approval_Level',
                    title='Volume vs Gross Margin by Approval Level',
                    category_orders={"Approval_Level": sorted(vis_data['Approval_Level'].unique())},                    log_x=True
                )
                st.plotly_chart(fig3)

            with col2:
                # Scatter plot with Volume vs GM colored by Escalation Reason
                fig4 = px.scatter(
                    vis_data,
                    x='Vol',
                    y='GM',
                    color='Final_Escalation',
                    title='Volume vs Gross Margin by Escalation Reason',
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
                x='DS',
                color='Approval_Level',
                title='Deal Score Distribution by Final Approval Level',
                category_orders={"Approval_Level": sorted(vis_data['Approval_Level'].unique())},
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig6)

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
                'Vol': ['count', 'mean', 'median', 'sum'],
                'GM': ['mean', 'median'],
                'DS': ['mean', 'median']
            }).round(0)
            
            # Calculate totals with matching MultiIndex structure
            total_stats = pd.DataFrame([
                [int(vis_data['Vol'].count()), 
                round(vis_data['Vol'].mean(), 0), 
                round(vis_data['Vol'].median(), 0),
                round(vis_data['Vol'].sum(), 0),
                round(vis_data['GM'].mean(), 0), 
                round(vis_data['GM'].median(), 0),
                round(vis_data['DS'].mean(), 0), 
                round(vis_data['DS'].median(), 0)]
            ], columns=summary_stats.columns, index=pd.Index(['Total']))

            # Combine the grouped stats with the total row
            summary_stats = pd.concat([summary_stats, total_stats])
            summary_stats = summary_stats.astype(int)

            def format_eur(x):
                return f"€{x:,.0f}".replace(',', '.')
            
            # Rename the columns for clarity
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
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle('Key Feature Relationships', fontsize=16)
                
                for i, target in enumerate(targets):
                    for j, feature in enumerate(features):
                        ax = axes[i, j]
                        sns.regplot(data=df, x=feature, y=target, ax=ax, scatter_kws={'alpha':0.5})
                        ax.set_title(f'{feature} vs {target}')
                        
                        # Add correlation coefficient
                        corr, _ = pearsonr(df[feature], df[target])
                        ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                            transform=ax.transAxes, 
                            fontsize=10,
                            verticalalignment='top')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

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

                for target in targets_display:
                    fig, axes = plt.subplots(1, len(categorical_cols), figsize=(20, 6))
                    title = f'{target.replace("_Numeric", "")} Distribution by Categories'
                    fig.suptitle(title, fontsize=16)
                    
                    for i, cat_col in enumerate(categorical_cols):
                        # Calculate statistics for each category
                        grouped_data = df.groupby(cat_col)[target]
                        means = grouped_data.mean()
                        sems = grouped_data.sem()  # Standard error of the mean
                        
                        # Create the bar plot without error bars first
                        ax = axes[i]
                        bars = ax.bar(range(len(means)), means)
                        
                        # Add error bars manually
                        ax.errorbar(range(len(means)), means, yerr=sems, fmt='none', color='black', capsize=5)
                        
                        # Customize the plot
                        ax.set_title(f'{target.replace("_Numeric", "")} by {cat_col}')
                        ax.set_xticks(range(len(means)))
                        ax.set_xticklabels(means.index, rotation=45, ha='right')
                        
                        # Add custom y-ticks for Approval_Level
                        if target == 'Approval Level':
                            ax.set_yticks(range(-1, 6))
                            ax.set_yticklabels(['N/A', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5'])
                
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # 4. Time Series Analysis
                st.subheader("Time Series Trends")
                if hasattr(df['MonthYear'].dtype, 'freq'):  # Check if it's Period
                    df['MonthYear'] = df['MonthYear'].dt.to_timestamp()
                # Convert MonthYear to datetime if not already
                #df['MonthYear'] = pd.to_datetime(df['MonthYear'])
                
                # Plot time series for both target variables
                fig, axes = plt.subplots(2, 1, figsize=(15, 10))
                
                for i, target in enumerate(targets):
                    monthly_avg = df.groupby('MonthYear')[target].mean()
                    
                    axes[i].plot(monthly_avg.index, monthly_avg.values, marker='o')
                    axes[i].set_title(f'{target} Trend Over Time')
                    axes[i].set_xlabel('Month-Year')
                    axes[i].set_ylabel(target)
                    axes[i].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            create_correlation_analysis(vis_data)

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
