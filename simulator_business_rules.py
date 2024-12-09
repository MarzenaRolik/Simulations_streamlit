import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Configure the page
st.set_page_config(
    page_title="Business Rules Simulator",
    page_icon="📊",
    layout="wide"
)
# Initialize session states
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None

# with open('./config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)
config = {
    'credentials': {
        'usernames': {
            st.secrets['username']: {
                'name': st.secrets['name'],
                'password': st.secrets['password']
            }
        }
    },
    'cookie': {
        'name': 'your_cookie_name',
        'key': st.secrets['cookie_key'],
        'expiry_days': 60
    }
}
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

def show_login():
    st.title("Deal Analysis Dashboard")
    st.markdown("Please log in to access the dashboard.")
    return authenticator.login( key='Login2')

def calculate_score_level(score, thresholds):
    for level, threshold in thresholds.items():
        if score < threshold:
            return level
    return 'L0'

def calculate_size_margin_level(deal_size, gross_margin):
    # L5 and L6 rules (AND condition)
    if deal_size >= 1000000 and gross_margin <= 65:
        return 'L5'
    
    # L0 to L4 rules (OR condition)
    if ((deal_size >= 500000.01 and deal_size <= 10000000) or (gross_margin  >= 0 and gross_margin <= 67)):
        return 'L4'
    elif ((deal_size >= 120000.01 and deal_size <= 500000) or (gross_margin >=67.01 and gross_margin <= 70)):
        return 'L3'
    elif ((deal_size >= 80000.01 and deal_size <= 120000) or (gross_margin >= 70.01 and gross_margin <= 74)):
        return 'L2'
    elif ((deal_size >=50000.01 and deal_size <= 80000) or (gross_margin >= 74.01 and gross_margin <= 77)):
        return 'L1'
    elif ((deal_size >= 0 and deal_size <= 50000) or (gross_margin >= 77.01)):
        return 'L0'

def get_final_level(score_level, size_margin_level):
    levels = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']
    score_idx = levels.index(score_level)
    size_margin_idx = levels.index(size_margin_level)
    return levels[max(score_idx, size_margin_idx)]

def show_dashboard():
        # Initialize session state variables if they don't exist
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'display_df' not in st.session_state:
        st.session_state.display_df = None

    # Logout button in sidebar
    with st.sidebar:
        st.write(f'Welcome *{st.session_state["name"]}*')
        authenticator.logout('Logout', 'sidebar')

        # Add a reset button in sidebar
        if st.session_state.file_uploaded:
            if st.button('Upload New File'):
                # Clean up all session state variables
                st.session_state.file_uploaded = False
                st.session_state.display_df = None
                if 'data' in st.session_state:
                    del st.session_state.data
                st.rerun()

    st.title('Deal Approval Level Simulator')
    
    # Initialize file_uploaded state if it doesn't exist
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    
    # Show upload section only if no file has been uploaded
    if not st.session_state.file_uploaded:
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                st.session_state.data = pd.read_csv(uploaded_file, decimal='.')
                st.session_state.display_df = st.session_state.data
                st.session_state.file_uploaded = True
                st.success('File successfully uploaded!')
                st.rerun()
            except Exception as e:
                st.error(f'Error reading the file: {str(e)}')
        else:
            st.info('Please upload a CSV file to view the dashboard')
    
    # Show dashboard only after file is uploaded
    if st.session_state.file_uploaded and 'data' in st.session_state:
        st.write(f"Using dataset with {len(st.session_state.data)} rows")
        data = st.session_state.data

        st.sidebar.header('Deal Score Thresholds - Simulate Various Thresholds Levels:')
        score_thresholds = {}
        for level in ['L5', 'L4', 'L3', 'L2', 'L1']:
            default_value = {'L5': 10, 'L4': 20, 'L3': 29, 'L2': 54, 'L1': 80}[level]
            score_thresholds[level] = st.sidebar.slider(
                f'{level} - Deal Score From',
                0, 100, default_value,
                help=f'Deals below this score require {level} approval'
            )
        
        st.sidebar.header('Deal Size & Margin Rules')
        st.sidebar.markdown('Current rules shown in the table below')
        
        # Display the rules table
        rules_data = {
            'Level': ['L0', 'L1', 'L2', 'L3', 'L4', 'L5/L6'],
            'Deal Size From': ['0', '50,000.01', '80,000.01', '120,000.01', '500,000.01', '1,000,000'],
            'Deal Size To': ['50,000', '80,000', '120,000', '500,000', '10,000,000', '-'],
            'Condition': ['OR', 'OR', 'OR', 'OR', 'OR', 'AND'],
            'Gross Margin From': ['77.01', '74.01', '70.01', '67.01', '0', '0'],
            'Gross Margin To': ['100', '77', '74', '70', '67', '65']
        }
        st.sidebar.dataframe(pd.DataFrame(rules_data))
        
        # Calculate approval levels
        df_raw = data.copy()
        df = df_raw[(df_raw['Gross Margin %'] <= 100) & (df_raw['Gross Margin %'] >= 0)]
        # Calculate levels based on both rule sets
        df['Creation date'] = pd.to_datetime(df['creation_date'])
        df['Sales EUR'] = pd.to_numeric(df['Sales EUR'])
        df['Gross Margin %'] = pd.to_numeric(df['Gross Margin %'])
        df['Score_Level'] = df['Deal Score'].apply(
            lambda x: calculate_score_level(x, score_thresholds)
        )
        
        df['Size_Margin_Level'] = df.apply(
            lambda row: calculate_size_margin_level(row['Sales EUR'], row['Gross Margin %']),
            axis=1
        )
        
        # Calculate final level
        df['Final_Level'] = df.apply(
            lambda row: get_final_level(row['Score_Level'], row['Size_Margin_Level']),
            axis=1
        )
        
        # Distribution plots
        tab1, tab2, tab3 = st.tabs(["Final Approval Levels", "Score Distribution", "Size vs Margin"])
        
        with tab1:
            level_counts = df['Final_Level'].value_counts()
            fig1 = px.pie(
                values=level_counts.values,
                names=level_counts.index,
                title='Distribution of Final Approval Levels'
            )
            st.plotly_chart(fig1)
        
        with tab2:
            fig2 = px.histogram(
                df,
                x='Deal Score',
                color='Final_Level',
                title='Deal Score Distribution by Final Approval Level',
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig2)
        
        with tab3:
            fig3 = px.scatter(
                df,
                x='Sales EUR',
                y='Gross Margin %',
                color='Final_Level',
                title='Deal Size vs Gross Margin by Final Approval Level',
                log_x=True
            )
            st.plotly_chart(fig3)
        
        # Data table
        st.header('Deal Details')
        display_df = df.copy()
        display_df['Sales EUR'] = display_df['Sales EUR'].apply(lambda x: f"€{x:,.2f}")
        display_df['Gross Margin %'] = display_df['Gross Margin %'].apply(lambda x: f"{x:.2f}%")
        display_df['Deal Score'] = display_df['Deal Score'].apply(lambda x: f"{x:.1f}")
        st.dataframe(display_df)

def main():
    # Check authentication status
    if st.session_state['authentication_status'] != True:
        st.title("Deal Analysis Dashboard")
        st.markdown("Please log in to access the dashboard.")
        
        # Handle authentication
        try:
            name, authentication_status, username = authenticator.login( key='Login1')
            if authentication_status:
                st.session_state['authentication_status'] = True
                st.session_state['name'] = name
                st.session_state['username'] = username
                st.rerun()
            elif authentication_status == False:
                st.error('Username/password is incorrect')
            elif authentication_status is None:
                st.warning('Please enter your username and password')
        except Exception as e:
            st.error(f"Please provide correct username and password")
    else:
        show_dashboard()

if __name__ == '__main__':
    main()