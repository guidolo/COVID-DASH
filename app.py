import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import datetime


def main():
    global WIDTH, HEIGHT
    df_c, df_c_r1, df_c_r100, df_c_pc, df_d, df_r, df_act, df_rate = load_data()
    st.sidebar.markdown('Last day in data: {}'.format(df_c.index.max()))
    option = st.sidebar.selectbox('Option', ['Confirmed',
                                             'Confirmed from 1st case',
                                             'Confirmed from 100 cases',
                                             'Confirmed Percent Change',
                                             'Active',
                                             'Death Rate',
                                             'Deaths',
                                             'Recovered'], index=0)
    log = st.sidebar.radio('Scale', ['Normal','Log'], index=0)
    #get width and heigh
    fig = plt.figure()
    #WIDTH, HEIGHT = fig.get_size_inches()*fig.dpi
    WIDTH, HEIGHT = [640, 480]
    if option in ['Confirmed', 'Active','Death Rate','Deaths','Recovered','Confirmed Percent Change']:
        start = st.sidebar.date_input("Start at",datetime.date(2020, 1, 22))
        end = st.sidebar.date_input("Finish at",datetime.date.today())
    else:
        start = st.sidebar.number_input('Starting at day:', 0, 100, step=1, value=0)
        end = st.sidebar.number_input('Ending at day:', 0, 100, step=1, value=100)
        
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
            visualize_data2(df_c.loc[start:end,], paises, log, 'Confirmed Cases', 'Date', '# Cases')
        if option == 'Confirmed from 1st case':
            visualize_data(df_c_r1.loc[start:end,], paises, log, option, 'Days', '# Cases')
        if option == 'Confirmed from 100 cases':
            visualize_data(df_c_r100.loc[start:end,], paises, log, option, 'Days', '# Cases')
        if option == 'Confirmed Percent Change':
            visualize_data(df_c_pc.loc[start:end,], paises, log, option, 'Date', 'Percent Change')
        if option == 'Deaths':
            visualize_data2(df_d.loc[start:end,], paises, log, 'Death Cases', 'Date', '# Cases')
        if option == 'Recovered':
            visualize_data2(df_r.loc[start:end,], paises, log, 'Recovered Cases', 'Date', '# Cases')
        if option == 'Active':
            visualize_data2(df_act.loc[start:end,], paises, log, 'Active Cases', 'Date', '# Cases')
        if option == 'Death Rate':
            visualize_data(df_rate.loc[start:end,], paises, log, option, 'Date', 'Death rate')
        
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
    df_c_pc = df_c.pct_change()
    return df_c, df_c_r1, df_c_r100, df_c_pc, df_d, df_r, df_act, df_rate

def visualize_data(df, paises, log, title, xlabel='', ylabel=''):
    global WIDTH, HEIGHT
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
        
def visualize_data2(df, paises, log, option, xlabel='', ylabel=''): 
    global WIDTH, HEIGHT
    if len(paises) > 0:
        source= df[paises].reset_index().melt('issue_date', var_name='Country', value_name='Cases').rename(columns={'issue_date':'Date'})

        # Create a selection that chooses the nearest point & selects based on x-value
        nearest = alt.selection(type='single', nearest=True, on='mouseover',fields=['Date'], empty='none')

        # The basic line
        line = alt.Chart(source).mark_line(interpolate='basis').encode(
            x='Date:T',
            y='Cases:Q',
            #y=alt.Y('Cases', scale=alt.Scale(type='log')),
            color=alt.Color('Country', legend=alt.Legend(orient="bottom"))
        )

        # Transparent selectors across the chart. This is what tells us
        # the x-value of the cursor
        selectors = alt.Chart(source).mark_point().encode(
            x='Date:T',
            opacity=alt.value(0),
        ).add_selection(
            nearest
        )

        # Draw points on the line, and highlight based on selection
        points = line.mark_point().encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )

        # Draw text labels near the points, and highlight based on selection
        text = line.mark_text(align='right', dx=5, dy=-5).encode(
            text=alt.condition(nearest, 'Cases:Q', alt.value(' '))
        )

        # Draw a rule at the location of the selection
        rules = alt.Chart(source).mark_rule(color='gray').encode(
            x='Date:T',
        ).transform_filter(
            nearest
        )

        # Put the five layers into a chart and bind the data
        graph = alt.layer(line, selectors, points, rules, text).properties(
                width=WIDTH, height=HEIGHT
                )

        st.write(graph)
        
if __name__ == "__main__":
    WIDTH = HEIGHT = 0
    main()

