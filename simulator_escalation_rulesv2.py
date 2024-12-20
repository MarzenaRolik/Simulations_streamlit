import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def determine_vol_level(vol, thresholds):
    if vol > thresholds['vol_l4']:
        return 'L4'
    elif vol > thresholds['vol_l3']:
        return 'L3'
    elif vol > thresholds['vol_l2']:
        return 'L2'
    elif vol > thresholds['vol_l1']:
        return 'L1'
    elif vol > 0:
        return 'L0'
    return 'N/A'

def determine_gm_level(gm, thresholds):
    if gm < thresholds['gm_l4']:
        return 'L4'
    elif gm < thresholds['gm_l3']:
        return 'L3'
    elif gm < thresholds['gm_l2']:
        return 'L2'
    elif gm < thresholds['gm_l1']:
        return 'L1'
    elif gm < 100:
        return 'L0'
    return 'N/A'

def determine_ds_level(ds, thresholds):
    if ds < thresholds['ds_l5']:
        return 'L5'
    elif ds < thresholds['ds_l4']:
        return 'L4'
    elif ds < thresholds['ds_l3']:
        return 'L3'
    elif ds < thresholds['ds_l2']:
        return 'L2'
    elif ds < thresholds['ds_l1']:
        return 'L1'
    elif ds <= 100:
        return 'L0'
    return 'N/A'

def get_level_value(level):
    level_map = {'L0': 0, 'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4, 'L5': 5, 'N/A': -1}
    return level_map.get(level, -1)

def determine_highest_level(row, thresholds):
    levels = [
        ('DS', get_level_value(determine_ds_level(row['Deal Score'], thresholds))),
        ('GM', get_level_value(determine_gm_level(row['Gross Margin %'], thresholds))),
        ('VOL', get_level_value(determine_vol_level(row['sales_eur'], thresholds))),
        ('OBR', get_level_value(row['Business Rule']))
    ]
    
    max_level = max(levels, key=lambda x: x[1])
    if max_level[0] == 'OBR' and row['Below Hard Floor Price Rule Violation']:
        return 'FP'
    return max_level[0]

def main():
    st.title('Escalation Rules Analysis and Simulation')
    
    # Sidebar
    st.sidebar.header('Filters and Rules')
    
    # File upload in sidebar
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Country filter in sidebar
        countries = ['All'] + list(data['Sales Org Country'].unique())
        selected_country = st.sidebar.selectbox('Select Country', countries)
        
        # Rule simulation section in sidebar
        st.sidebar.header('Rule Simulation')
        
        # Volume thresholds
        st.sidebar.subheader('Volume Thresholds')
        vol_l4 = st.sidebar.number_input('VOL L4 Threshold', value=500000.01)
        vol_l3 = st.sidebar.number_input('VOL L3 Threshold', value=120000.01)
        vol_l2 = st.sidebar.number_input('VOL L2 Threshold', value=80000.01)
        vol_l1 = st.sidebar.number_input('VOL L1 Threshold', value=50000.01)
        
        # GM thresholds with sliders
        st.sidebar.subheader('GM Thresholds')
        gm_l4 = st.sidebar.slider('GM L4 Threshold', 0, 100, 67)
        gm_l3 = st.sidebar.slider('GM L3 Threshold', 0, 100, 70)
        gm_l2 = st.sidebar.slider('GM L2 Threshold', 0, 100, 74)
        gm_l1 = st.sidebar.slider('GM L1 Threshold', 0, 100, 77)
        
        # DS thresholds with sliders
        st.sidebar.subheader('DS Thresholds')
        ds_l5 = st.sidebar.slider('DS L5 Threshold', 0, 100, 10)
        ds_l4 = st.sidebar.slider('DS L4 Threshold', 0, 100, 21)
        ds_l3 = st.sidebar.slider('DS L3 Threshold', 0, 100, 30)
        ds_l2 = st.sidebar.slider('DS L2 Threshold', 0, 100, 55)
        ds_l1 = st.sidebar.slider('DS L1 Threshold', 0, 100, 81)
        
        # Collect thresholds in a dictionary
        thresholds = {
            'vol_l4': vol_l4, 'vol_l3': vol_l3, 'vol_l2': vol_l2, 'vol_l1': vol_l1,
            'gm_l4': gm_l4, 'gm_l3': gm_l3, 'gm_l2': gm_l2, 'gm_l1': gm_l1,
            'ds_l5': ds_l5, 'ds_l4': ds_l4, 'ds_l3': ds_l3, 'ds_l2': ds_l2, 'ds_l1': ds_l1
        }
        
        # Filter data based on country selection
        if selected_country != 'All':
            filtered_data = data[data['Sales Org Country'] == selected_country]
        else:
            filtered_data = data.copy()
        
        # Apply rules and calculate distributions
        filtered_data['VOL_Level'] = filtered_data['sales_eur'].apply(lambda x: determine_vol_level(x, thresholds))
        filtered_data['GM_Level'] = filtered_data['Gross Margin %'].apply(lambda x: determine_gm_level(x, thresholds))
        filtered_data['DS_Level'] = filtered_data['Deal Score'].apply(lambda x: determine_ds_level(x, thresholds))
        filtered_data['Final_Escalation'] = filtered_data.apply(lambda row: determine_highest_level(row, thresholds), axis=1)
        
        # Main content area - Visualizations
        st.header('Visualizations')
        
        # First row of visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of final escalation distribution
            level_counts = filtered_data['Final_Escalation'].value_counts()
            fig1 = px.pie(
                values=level_counts.values,
                names=level_counts.index,
                title='Distribution of Final Escalation Reasons'
            )
            st.plotly_chart(fig1)
        
        with col2:
            # Histogram of DS by final escalation
            fig2 = px.histogram(
                filtered_data,
                x='Deal Score',
                color='Final_Escalation',
                title='Deal Score Distribution by Final Escalation',
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig2)
        
        # Second row - Scatter plot
        fig3 = px.scatter(
            filtered_data,
            x='sales_eur',
            y='Gross Margin %',
            color='Final_Escalation',
            title='Volume vs Gross Margin by Final Escalation',
            log_x=True
        )
        st.plotly_chart(fig3)
        
        # Additional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig4 = px.box(
                filtered_data,
                x='Final_Escalation',
                y='Deal Score',
                title='Deal Score Distribution by Final Escalation'
            )
            st.plotly_chart(fig4)
        
        with col2:
            fig5 = px.box(
                filtered_data,
                x='Final_Escalation',
                y='Gross Margin %',
                title='Gross Margin Distribution by Final Escalation'
            )
            st.plotly_chart(fig5)
        
        # Display raw data
        st.header('Raw Data')
        st.dataframe(filtered_data)

if __name__ == '__main__':
    main()