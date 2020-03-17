import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt

def main():
    df = load_data()
    page = st.sidebar.selectbox("Choose a page", ["Graph", "Data"])

    if page == "Data":
        st.header("This is your data explorer.")
        st.write("Please select a page on the left.")
        st.write(df)
    elif page == "Graph":
        st.title("Data Exploration")
        paises = st.multiselect("Choose a Country", df.columns)
        log = st.radio('Scale', ['Normal','Log'], index=0)
        visualize_data(df, paises, log)

@st.cache
def load_data():
    df = pd.read_csv('./data/time_series_19-covid-Confirmed.csv')
    
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

def visualize_data(df, paises, log):
    #paises = ["Spain", "Italy", "Argentina","Japan","Germany", "Netherlands", "France", "China"]
    #paises = ["Argentina", "Chile", "Brazil", "Colombia", "Costa Rica", "Peru", "Mexico"]
    if len(paises) > 0:
        paises.sort()
        df = df.loc[:, paises]
        plt.plot(df)
        if log == 'Log':
            plt.yscale('log')
        plt.legend(paises)
        st.pyplot()

if __name__ == "__main__":
    main()

