import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from PIL import Image
import datetime 
import pickle

from datetime import datetime as dt
from sklearn import metrics
import seaborn as sns
#import pandas_profiling
from scipy import stats

from sklearn.ensemble import RandomForestRegressor

#====================================================================================================
df=pd.read_csv(r'/Users/karthikeyank/Desktop/Project/final1.csv')



#------------------------------------------------------------------------------------------------------------------------------------------

#page congiguration
st.set_page_config(layout="wide", page_title="Retail_Sales_Prediction", page_icon="random", initial_sidebar_state="expanded")

st.markdown("<h1 style='text-align: center; color: violet;'>Retail Weekly Sales Predicton</h1>",unsafe_allow_html=True)

#===============================================================================================================================================

tab1,tab2,tab3=st.tabs(['HOME','PREDICTION','CONCLUSION'])

with tab1:
    col1=st.columns(2,gap="large")

    with col1:
        col1.markdown("#### :red[Domain]")
        st.write("##### Retail Industry")
        st.write()
        col1.markdown("#### :red[Technologies  and Tools used]")
        st.write("##### Python, Pandas, numpy, matplotlib, seaborn, Plotly, Streamlit, sklearn.")
        st.write()
        col1.markdown("#### :red[Overview of the Project] ")
        st.write("##### *  Predict the weekly sales of a retail store based on historical sales using Machine Learning techniques. ")
        st.write("##### *  To perform Data cleaning, Exploratory Data Analysis, Feature Engineering, Hypothesis Testing for the ML model. ")
        st.write("##### *  In this use case I used the :violet[RANDOM FOREST REGRESSOR] model to predict the weekly sales of the retail store. ")




with tab2:
    option = st.radio('**Select your option**',('Processed Data', 'Prediction Process',),horizontal=True)

    if option == 'Processed Data':
        st.header("Processed  Final Data")
        st.write(df)

    if option == 'Prediction Process':

        col3,col4,col5,col6,col10=st.columns(5,gap="large")

        with col3:

            st.header(":violet[Store info]")

            # store
            store_no = [i for i in range(1, 46)] 

            store = st.selectbox('Select the **:red[Store Number]**', store_no)

            # department
            dept_no = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,33, 34, 35, 36, 37, 38, 40, 
                        41, 42, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55,56, 58, 59, 60, 67, 71, 72, 
                        74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92, 93, 94, 95, 97, 78, 96, 99, 39,
                        77, 50, 43, 65, 98]

            dept = st.selectbox('Select the **:red[Department Number]**', dept_no)

            # Type
            Type = {'A':0,"B":1,"C":2}

            type_ = st.selectbox('Select the **:red[Store Type]**', ['A', 'B', 'C'])

            # Size
            size=[151315, 202307,  37392, 205863,  34875, 202505,  70713, 155078,
                  125833, 126512, 207499, 112238, 219622, 200898, 123737,  57197,
                  93188, 120653, 203819, 203742, 140167, 119557, 114533, 128107,
                  152513, 204184, 206302,  93638,  42988, 203750, 203007,  39690,
                  158114, 103681,  39910, 184109, 155083, 196321,  41062, 118221]
            
            siz=st.selectbox('Select the **:red[Store Size]**', size)



        with col4:

            st.header(":violet[Indirect Impact on Store Sales]")

            # TEMPERATURE
            temp = st.number_input('Enter the ****:red[Temperature]**** in fahrenheit -----> **:green[(min=5.54 & max=100.14)]**',value=89.0,min_value=5.54,max_value=100.14)
            
            # Fuel Price
            fuel = st.number_input('Enter the **:red[Fuel Price]** ---> **:green[(min=2.472 & max=4.468)]**',value=3.00,min_value=2.472,max_value=4.468)

            # Consumer Price Index
            CPI = st.number_input('Enter the **:red[CPI]** ----------> **:green[(min=126.0 & max=227.47)]**',value=200.33,min_value=126.0,max_value=227.47)

                # min : 126.064
                # max : 227.2328068
            # Unemployment
            unemp = st.number_input('Enter the **:red[Unemployment Rate]** in percentage **:green[(min=3.879 & max=14.313)]**',value=10.00,min_value=3.879,max_value=14.313)

            # min : 3.879
            # max : 14.313    


        with col5: 
            st.header(":violet[Markdown are Impact on Store Sales]")   

            # markown
            def inv_trans(x):
                return 1/x

            markdown1 = st.number_input('Enter the **:red[Markdown1]** in dollars -------- **:green[(min=0.27,max=88646.76)]**',value=2000.00,min_value=0.27,max_value=88646.76)
            markdown1=inv_trans(markdown1)

            markdown2 = st.number_input('Enter the **:red[Markdown2]** in dollars -------- **:green[(min=0.02,max=104519.54)]**',value=65000.00,min_value=0.02,max_value=104519.54)
            markdown2=inv_trans(markdown2)

            markdown3=st.number_input('Enter the **:red[Markdown3]** in dollars -------- **:green[(min=0.01,max=141630.61)]**',value=27000.00,min_value=0.01,max_value=141630.61)
            markdown3=inv_trans(markdown3)

            markdown4=st.number_input('Enter the **:red[Markdown4]** in dollars -------- **:green[(min=0.22,max=67474.85))]**',value=11200.00,min_value=0.22,max_value=67474.85)
            markdown4=inv_trans(markdown4)

            markdown5=st.number_input('Enter the **:red[Markdown5]** in dollars -------- **:green[(min=135.06,max=108519.28)]**',value=89000.00,min_value=135.06,max_value=108519.28)
            markdown5=inv_trans(markdown5)



        with col6:
            st.header(":violet[Direct Impact on Store Sales]")
            
            # Date
            duration = st.date_input("Select the **:red[Date]**", datetime.date(2012, 7, 20), min_value=datetime.date(2010, 2, 5), max_value=datetime.date.today())

            # Holiday
            holi={"YES":1,"NO":0}

            holiday = st.selectbox('Select the **:red[Holiday]**', ["YES","NO"])     
            




        with col10:
            st.header(":red[Weekly Sales]")  

            import pickle
            model=pickle.load(open(r'/Users/karthikeyank/Desktop/Project/model.pkl', 'rb'))

            if st.button('Predict'):

                result = model.predict([[store,dept,siz,temp, fuel,CPI, unemp, markdown1,markdown2,markdown3,markdown4,markdown5, duration.year,duration.month,duration.day,Type[type_] ,holi[holiday]]])
                
                #predicted_price = str(result)[1:-1]
                price=result.round(2)

                st.warning(f'Predicted weekly sales of the retail store is: $ {price}')

                #st.success('Predicted weekly sales is {}'.format(price))
                #st.success(f'Predicted weekly sales of the retail  store is     : $ {result}')

    with tab3:

        col8,col9=st.columns(2,gap="large")
        with col8:

            st.subheader("My observation from analysis and prediction of this data...") 

            st.write(" * The **:red[Weekly Sales]** of the retail store is depened on many of the factors ")
            st.write(" * These factors are Directly or Indirectly affecting the Weekly Sales. ")
            st.write(" * **:red[Size of the store]** is playing a major role.")
            st.write(" * Combination of **:violet[Fuel Price]** and **:green[Unemployment rate]** is significantly impacting the Weekly Sales.")
            st.write(" * **:red[Temprature]** and **:green[Markdown]** are in indirect relation and some times in direct relationship. ")
            st.write(" * Both of them hughly impacting the Weekly Sales of the retail store. ")

        #with col9:
            #st.image(Image.open(r"C:\Users\Muthusamy\Pictures\logo\download.png"), width=300)    





            
      


    