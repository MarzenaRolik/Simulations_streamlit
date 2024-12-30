import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime

def load_default_thresholds():
    """Load default thresholds from JSON file or use built-in defaults."""
    default_thresholds = {
        "DEFAULT": {
            "vol_l4": 10000000.01,
            "vol_l3": 500000.01,
            "vol_l2": 120000.01,
            "vol_l1": 80000.01,
            "vol_l0": 50000.01,

            "gm_l4": 0,
            "gm_l3": 67,
            "gm_l2": 70,
            "gm_l1": 74,
            "gm_l0": 77,

            "ds_l4": 10,
            "ds_l3": 21,
            "ds_l2": 30,
            "ds_l1": 55,
            "ds_l0": 81
        }
    }

    try:
        with open('thresholds.json', 'r') as f:
            saved_thresholds = json.load(f)
            # Ensure all required keys exist
            for country, thresholds in saved_thresholds.items():
                for key in default_thresholds["DEFAULT"].keys():
                    if key not in thresholds:
                        thresholds[key] = default_thresholds["DEFAULT"][key]
            return saved_thresholds
    except FileNotFoundError:
        return default_thresholds

def save_thresholds(thresholds):
    """Save thresholds to JSON file."""
    with open('thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=4)

def initialize_session_state():
    """Initialize or reset session state variables."""
    if 'permanent_thresholds' not in st.session_state:
        st.session_state.permanent_thresholds = load_default_thresholds()
    if 'temp_thresholds' not in st.session_state:
        st.session_state.temp_thresholds = {}
    if 'modified_countries' not in st.session_state:
        st.session_state.modified_countries = set()
    if 'initial_thresholds' not in st.session_state:
        st.session_state.initial_thresholds = st.session_state.permanent_thresholds.copy()

def display_threshold_controls(country):
    """Display and handle threshold controls for a specific country."""
    # Ensure country exists in permanent_thresholds
    if country not in st.session_state.permanent_thresholds:
        st.session_state.permanent_thresholds[country] = st.session_state.permanent_thresholds['DEFAULT'].copy()
    
    # Retrieve current thresholds by merging permanent and temporary thresholds
    current_thresholds = st.session_state.permanent_thresholds[country].copy()
    if country in st.session_state.temp_thresholds:
        current_thresholds.update(st.session_state.temp_thresholds[country])

    modified = False

    # Gross Margin % Thresholds
    st.sidebar.subheader('Gross Margin % Thresholds')
    new_gm_l0 = st.sidebar.slider('Gross Margin % L0 More Than', 0, 100, int(current_thresholds['gm_l0']), key=f'gm_l0_{country}')
    new_gm_l1 = st.sidebar.slider('Gross Margin % L1 More Than', 0, 100, int(current_thresholds['gm_l1']), key=f'gm_l1_{country}')
    new_gm_l2 = st.sidebar.slider('Gross Margin % L2 More Than', 0, 100, int(current_thresholds['gm_l2']), key=f'gm_l2_{country}')
    new_gm_l3 = st.sidebar.slider('Gross Margin % L3 More Than', 0, 100, int(current_thresholds['gm_l3']), key=f'gm_l3_{country}')
    new_gm_l4 = st.sidebar.slider('Gross Margin % L4 More Than', 0, 100, int(current_thresholds['gm_l4']), key=f'gm_l4_{country}')

    # Deal Score Thresholds
    st.sidebar.subheader('Deal Score Thresholds')
    new_ds_l0 = st.sidebar.slider('Deal Score L0 More Than', 0, 100, int(current_thresholds['ds_l0']), key=f'ds_l0_{country}')
    new_ds_l1 = st.sidebar.slider('Deal Score L1 More Than', 0, 100, int(current_thresholds['ds_l1']), key=f'ds_l1_{country}')
    new_ds_l2 = st.sidebar.slider('Deal Score L2 More Than', 0, 100, int(current_thresholds['ds_l2']), key=f'ds_l2_{country}')
    new_ds_l3 = st.sidebar.slider('Deal Score L3 More Than', 0, 100, int(current_thresholds['ds_l3']), key=f'ds_l3_{country}')
    new_ds_l4 = st.sidebar.slider('Deal Score L4 More Than', 0, 100, int(current_thresholds['ds_l4']), key=f'ds_l4_{country}')

    # Deal Size (€) Thresholds
    st.sidebar.subheader('Deal Size (€) Thresholds')
    new_vol_l0 = st.sidebar.number_input('Deal Size (€) L0 Less Than', value=float(current_thresholds['vol_l0']), key=f'vol_l0_{country}')
    new_vol_l1 = st.sidebar.number_input('Deal Size (€) L1 Less Than', value=float(current_thresholds['vol_l1']), key=f'vol_l1_{country}')
    new_vol_l2 = st.sidebar.number_input('Deal Size (€) L2 Less Than', value=float(current_thresholds['vol_l2']), key=f'vol_l2_{country}')
    new_vol_l3 = st.sidebar.number_input('Deal Size (€) L3 Less Than', value=float(current_thresholds['vol_l3']), key=f'vol_l3_{country}')
    new_vol_l4 = st.sidebar.number_input('Deal Size (€) L4 Less Than', value=float(current_thresholds['vol_l4']), key=f'vol_l4_{country}')

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
    if any(current_thresholds[k] != new_thresholds[k] for k in new_thresholds):
        st.session_state.temp_thresholds[country] = new_thresholds.copy()
        st.session_state.modified_countries.add(country)
        modified = True

    return modified

def save_permanent_thresholds():
    """Save all temporary thresholds to permanent_thresholds and persist to file."""
    try:
        # Update permanent_thresholds with temp_thresholds for each modified country
        for country in st.session_state.modified_countries:
            st.session_state.permanent_thresholds[country] = st.session_state.temp_thresholds[country].copy()

        # Save the updated permanent_thresholds to thresholds.json
        save_thresholds(st.session_state.permanent_thresholds)

        # Clear modification tracking
        st.session_state.modified_countries.clear()
        st.session_state.temp_thresholds.clear()

        st.success("All changes saved successfully.")
        return True
    except Exception as e:
        st.error(f"Error saving thresholds: {str(e)}")
        return False

def reset_to_initial_thresholds():
    """Reset permanent_thresholds to initial_thresholds and persist to file."""
    try:
        st.session_state.permanent_thresholds = st.session_state.initial_thresholds.copy()
        st.session_state.temp_thresholds.clear()
        st.session_state.modified_countries.clear()
        save_thresholds(st.session_state.permanent_thresholds)
        st.success("All thresholds have been reset to initial values.")
    except Exception as e:
        st.error(f"Error resetting thresholds: {str(e)}")

def get_applicable_thresholds(country):
    """Retrieve the latest thresholds for a country, considering any temporary changes."""
    # Start with the permanent thresholds
    thresholds = st.session_state.permanent_thresholds.get(country, st.session_state.permanent_thresholds['DEFAULT']).copy()
    # Apply any temporary changes
    if country in st.session_state.temp_thresholds:
        thresholds.update(st.session_state.temp_thresholds[country])
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

    st.title('Escalation Rules Analysis and Simulation')

    # Sidebar - Configuration
    st.sidebar.header('Configuration')

    # Upload thresholds JSON
    threshold_file = st.sidebar.file_uploader("Upload Thresholds JSON", type=['json'])
    if threshold_file is not None:
        try:
            loaded_thresholds = json.load(threshold_file)
            # Update permanent_thresholds with the loaded thresholds
            st.session_state.permanent_thresholds = loaded_thresholds.copy()
            # Clear any temporary thresholds and modified countries
            st.session_state.temp_thresholds.clear()
            st.session_state.modified_countries.clear()
            # Update initial_thresholds to the loaded thresholds
            st.session_state.initial_thresholds = loaded_thresholds.copy()
            st.sidebar.success('Thresholds loaded successfully.')
        except json.JSONDecodeError:
            st.sidebar.error('Invalid JSON file uploaded.')
        except Exception as e:
            st.sidebar.error(f'Error loading thresholds: {str(e)}')

    # Add Reset to Initial Thresholds button
    if st.sidebar.button('Reset to Initial JSON Thresholds'):
        reset_to_initial_thresholds()

    # Upload data
    data_file = st.sidebar.file_uploader("Upload Data CSV", type=['csv'])

    if data_file is not None:
        try:
            data = pd.read_csv(data_file)

            # Sidebar Filters
            st.sidebar.header('Filters')
            countries = list(data['Country'].unique())
            selected_countries = st.sidebar.multiselect(
                'Select Countries',
                options=['All'] + countries,
                default=['All']
            )

            # Show threshold controls for selected countries
            if selected_countries and 'All' not in selected_countries:
                st.sidebar.header('Threshold Configuration')
                for country in selected_countries:
                    with st.sidebar.expander(f'Thresholds for {country}'):
                        display_threshold_controls(country)

            # Show modified countries status
            if st.session_state.modified_countries:
                st.sidebar.markdown("---")
                st.sidebar.subheader("Modified Countries")
                st.sidebar.info(f"Countries with unsaved changes: {', '.join(st.session_state.modified_countries)}")

                # Global save and reset buttons
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    if st.button('Save All Changes'):
                        save_permanent_thresholds()
                with col2:
                    if st.button('Reset All Changes'):
                        reset_to_initial_thresholds()

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
            vis_selected_GM = st.sidebar.slider('Gross Margin Not More Than', 0, 1000, 150)

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
                        json.dump(st.session_state.permanent_thresholds, f, indent=4)
                    st.sidebar.success(f'Thresholds exported to {filename}')
                except Exception as e:
                    st.sidebar.error(f'Error exporting thresholds: {str(e)}')

        except Exception as e:
            st.error(f"Error processing data file: {str(e)}")

if __name__ == '__main__':
    main()
