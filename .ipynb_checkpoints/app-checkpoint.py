import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt

def main():
    df_c, df_c_r1, df_c_r100, df_d, df_r, df_act, df_rate = load_data()
    st.sidebar.markdown('Last day in data: {}'.format(df_c.index.max()))
    option = st.sidebar.selectbox('Option', ['Confirmed',
                                             'Confirmed from 1st case',
                                             'Confirmed from 100 cases',
                                             'Active',
                                             'Death Rate',
                                             'Deaths',
                                             'Recovered'], index=0)
    log = st.sidebar.radio('Scale', ['Normal','Log'], index=0)
    page = st.sidebar.selectbox("Choose a page", ["Graph", "Data"])


    if page == "Data":
        #st.header("This is your data explorer.")
        st.write("Please select a page on the left.")
        st.write(df_c)
        st.write('Data from https://github.com/CSSEGISandData/COVID-19')
        st.write('This repo https://github.com/guidolo/COVID-DASH')
    elif page == "Graph":
        #st.header('COVID-19 time exploration')
        st.title("COVID-19 Time Exploration")
        paises = st.multiselect("Choose a Country", df_c.columns)
        if option == 'Confirmed':
            visualize_data(df_c, paises, log, 'Confirmed Cases', 'Date', '# Cases')
        if option == 'Confirmed from 1st case':
            visualize_data(df_c_r1, paises, log, option, 'Days', '# Cases')
        if option == 'Confirmed from 100 cases':
            visualize_data(df_c_r100, paises, log, option, 'Days', '# Cases')
        if option == 'Deaths':
            visualize_data(df_d, paises, log, 'Death Cases', 'Date', '# Cases')
        if option == 'Recovered':
            visualize_data(df_r, paises, log, 'Recovered Cases', 'Date', '# Cases')
        if option == 'Active':
            visualize_data(df_act, paises, log, 'Active Cases', 'Date', '# Cases')
        if option == 'Death Rate':
            visualize_data(df_rate, paises, log, option, 'Date', 'Death rate')

def rescale(df, n):
    for x in range(0,n):
        df = df.reset_index(drop=True).replace(x, np.nan)
    for col in df:
        if col == 'China' and n == 100:
            df.loc[:, col] = pd.concat([df['China'].iloc[:-5], pd.DataFrame([0,0,0,0,0,0,0,0], index=[0,0,0,0,0,0,0,0])]).rename(columns={0:'China'}).China.sort_values().reset_index(drop=True)
        else:
            df.loc[:, col] = df.loc[:,col].sort_values().reset_index(drop=True)
    return df

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
    df_c = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
    df_d = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
    df_r = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
    df_c = make_data(df_c)
    df_d = make_data(df_d)
    df_r = make_data(df_r)
    df_act = df_c - df_d - df_r
    df_rate = (df_d / df_c).fillna(0).replace(np.inf, 0)
    df_c_r100 = rescale(df_c, 100)
    df_c_r1 = rescale(df_c, 1)
    return df_c, df_c_r1, df_c_r100, df_d, df_r, df_act, df_rate

def visualize_data(df, paises, log, title, xlabel='', ylabel=''):
    if len(paises) > 0:
        paises.sort()
        df = df.loc[:, paises]
        plt.plot(df)
        plt.title('COVID19 -- '+ title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
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

