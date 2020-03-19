import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt

def main():
    df_c, df_d, df_r, df_act, df_rate = load_data()
    st.sidebar.markdown('Last update: {}'.format(df_c.index.max()))
    page = st.sidebar.selectbox("Choose a page", ["Graph", "Data"])
    log = st.sidebar.radio('Scale', ['Normal','Log'], index=0)
    option = st.sidebar.selectbox('Option', ['Confirmed','Active','Death Rate','Deaths','Recovered'], index=0)

    if page == "Data":
        #st.header("This is your data explorer.")
        st.write("Please select a page on the left.")
        st.write(df_c)
        st.write('Data from https://github.com/CSSEGISandData/COVID-19')
    elif page == "Graph":
        #st.header('COVID-19 time exploration')
        st.title("COVID-19 Time Exploration")
        paises = st.multiselect("Choose a Country", df_c.columns)
        if option == 'Confirmed':
            visualize_data(df_c, paises, log, option)
        if option == 'Deaths':
            visualize_data(df_d, paises, log, option)
        if option == 'Recovered':
            visualize_data(df_r, paises, log, option)
        if option == 'Active':
            visualize_data(df_act, paises, log, option)
        if option == 'Death Rate':
            visualize_data(df_rate, paises, log, option)
            
def make_data(df):
    df.set_index(['Province/State', 'Country/Region', 'Lat', 'Long'], inplace=True)
    df = (df.
         stack().
        reset_index().
        rename(columns={'level_4':'issue_date', 0:'cant', 'Country/Region':'Country'})
    )
    df = df.groupby(['Country','issue_date']).sum().drop(['Lat','Long'], axis=1).reset_index()
    df.issue_date = pd.to_datetime(df.issue_date)
    df = df.set_index(['issue_date','Country']).unstack()
    df = df.loc[:, 'cant']
    return df

@st.cache
def load_data():
    df_c = pd.read_csv('./data/time_series_19-covid-Confirmed.csv')
    df_d = pd.read_csv('./data/time_series_19-covid-Deaths.csv')
    df_r = pd.read_csv('./data/time_series_19-covid-Recovered.csv')
    df_c = make_data(df_c)
    df_d = make_data(df_d)
    df_r = make_data(df_r)
    df_act = df_c - df_d - df_r
    df_rate = (df_d / df_c).fillna(0).replace(np.inf, 0)
    return df_c, df_d, df_r, df_act, df_rate

def visualize_data(df, paises, log, option):
    if len(paises) > 0:
        paises.sort()
        df = df.loc[:, paises]
        plt.plot(df)
        plt.title('COVID19 -- '+option+' Cases')
        plt.xticks(rotation=20)
        if log == 'Log':
            plt.yscale('log')
        plt.legend(paises)
        st.pyplot()

def visualize_data2(df, paises, log, option): 
        df.columns = [x[:3] for x in df.columns]
        if log == 'Log':
            df = np.log(df)
        st.line_chart(df)

        
if __name__ == "__main__":
    main()

