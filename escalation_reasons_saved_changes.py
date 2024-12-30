import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime

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
        st.warning("thresholds.json not found.")
        
    except json.JSONDecodeError:
        # Handle corrupted JSON file
        st.error("Error reading thresholds.json. File may be corrupted.")
    
    except Exception as e:
        # Handle any other errors
        st.error(f"Please upload thresholds and data files. ")

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

    # Compile new thresholds from UI inputs
    new_thresholds = {
        'vol_l4': new_vol_l4, 'vol_l3': new_vol_l3, 'vol_l2': new_vol_l2,
        'vol_l1': new_vol_l1, 'vol_l0': new_vol_l0,
        'gm_l4': new_gm_l4, 'gm_l3': new_gm_l3, 'gm_l2': new_gm_l2,
        'gm_l1': new_gm_l1, 'gm_l0': new_gm_l0,
        'ds_l4': new_ds_l4, 'ds_l3': new_ds_l3, 'ds_l2': new_ds_l2,
        'ds_l1': new_ds_l1, 'ds_l0': new_ds_l0
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

def get_level_value(level):
    level_map = {'L0': 0, 'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4, 'L5': 5, 'N/A': -1}
    return level_map.get(level, -1)

def determine_highest_level(row):
    """Determine the highest escalation level based on thresholds."""
    thresholds = get_applicable_thresholds(row['Country'])

    levels = [
        ('Deal Score', get_level_value(determine_ds_level(row['DS'], thresholds))),
        ('Gross Margin %', get_level_value(determine_gm_level(row['GM'], thresholds))),
        ('Deal size (€)', get_level_value(determine_vol_level(row['Vol'], thresholds))),
        ('Other Business Rule', get_level_value(row['oBR_level']))
    ]
    max_level = max(levels, key=lambda x: x[1])
    if max_level[0] == 'Other Business Rule' and row.get('FP', False) == True:
        return 'Below Floor Price'
    return max_level[0]

def get_approval_level(row):
    """Get approval level using the latest thresholds."""
    thresholds = get_applicable_thresholds(row['Country'])
    
    levels = [
        determine_ds_level(row['DS'], thresholds),
        determine_gm_level(row['GM'], thresholds),
        determine_vol_level(row['Vol'], thresholds),
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

            # Apply visualization filters
            vis_data = filtered_data[
                (filtered_data['Country'].isin(vis_selected_countries)) &
                (filtered_data['quote_type'].isin(vis_selected_quote_types)) &
                (filtered_data['GM'] < vis_selected_GM)
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
                    title=f'Distribution of Approval Levels ({len(vis_data)} deals)'
                )
                st.plotly_chart(fig1)

            with col2:
                # Pie chart of escalation reasons
                escalation_counts = vis_data['Final_Escalation'].value_counts()
                fig2 = px.pie(
                    values=escalation_counts.values,
                    names=escalation_counts.index,
                    title='Distribution of Escalation Reasons'
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
                    log_x=True
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
                    log_x=True
                )
                st.plotly_chart(fig4)

            # Third Row: Approval Level vs Escalation Reason
            fig5 = px.histogram(
                vis_data,
                x='Final_Escalation',
                color='Approval_Level',
                title='Approval Levels by Escalation Reason',
                barmode='group'
            )
            st.plotly_chart(fig5)

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
                'Vol': ['count', 'mean', 'median'],
                'GM': ['mean', 'median'],
                'DS': ['mean', 'median']
            }).round(2)
            summary_stats.columns = ['Count', 'Avg Volume', 'Median Volume', 
                                     'Avg GM', 'Median GM',
                                     'Avg DS', 'Median DS']
            st.dataframe(summary_stats)

            # Raw Data Table
            st.header('Raw Data')
            st.dataframe(vis_data)

            # Export Current Thresholds
            if st.sidebar.button('Export Current Thresholds'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'thresholds_{timestamp}.json'
                try:
                    with open(filename, 'w') as f:
                        json.dump(st.session_state.thresholds, f, indent=4)
                    st.sidebar.success(f'Thresholds exported to {filename}')
                except Exception as e:
                    st.sidebar.error(f'Error exporting thresholds: {str(e)}')

        except Exception as e:
            st.error(f"Error processing data file: {str(e)}")

if __name__ == '__main__':
    main()
