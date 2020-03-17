import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt

def main():
    df = load_data()
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Exploration"])

    if page == "Homepage":
        st.header("This is your data explorer.")
        st.write("Please select a page on the left.")
        st.write(df)
    elif page == "Exploration":
        st.title("Data Exploration")
        x_axis = st.selectbox("Choose a Country", df.columns, index=3)
        visualize_data(df, x_axis)

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

def visualize_data(df, x_axis):
    paises = ["Spain", "Italy", "Argentina","Japan","Germany", "Netherlands", "France", "China"]
    paises = ["Argentina", "Chile", "Brazil", "Colombia", "Costa Rica", "Peru", "Mexico"]
    paises.sort()

    df = df.loc[:, paises]
    plt.plot(df)
    st.pyplot()

if __name__ == "__main__":
    main()

