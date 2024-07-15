import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrices import mean_squared_error
import streamlit as st

train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.cv')

#data cleaning
missing_val=train_data.isnull().sum()
high_missing_col=missing_val[missing_val>train_data.shape[0]*0.4].index
train_data.drop(columns=high_missing_col,inplace=True)
test_data.drop(columns=high_missing_col,inplace=True)

#fill the missing values using numerical columns
for col in train_data.select_dtypes(include=['float64','int64']).columns:
    fill_value=train_data[col].median()
    train_data[col].fillna(fill_value,inplace=True)
    test_data[col].fillna(fill_value,inplace=True)

#fill missing values for categorical columns
for col in train_data.select_dtypes(include=['object']).columns:
    fill_value=train_data[col].mode()[0]
    train_data[col].fillna(fill_value,inplace=True)
    test_data[col].fillna(fill_value,inplace=True)

# Exploratory Data Analysis(EDA)
plt.figure(figsize=(0,4))
sns.highplot(train_data['SalesPrice'],kde=True)
plt.title('Distribution of Sale Price')
plt.show()

# Correlation Heatmap
correlation= train_data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation,annot=False,cmap='coolware')
plt.title('Correlation Heatmap')
plt.show()

# Sale Price Vs Overall Quality
sns.boxplot(x='OverallQual',y='SalePrice',data=train_data)
plt.title('Sale Price Vs Overall Quality')
plt.show()

# Featuring 
train_data['TotalSF']=train_data['TotalBsmtSF']+train_data['1stFlrF']
test_data['TotalSF']=test_data['TotalBsmtSF']+test_data['1stFlrSF']

# Selecting significant features
features=['OverallQual','GrLivArea','TotalSF','GarageCars','FullBath','YearBull']
x_train=train_data[features]
y_train=train_data['SalePrice']
x_test=test_data[features]

# Modelling using RandomForestRegressor
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_train)
rase=mean_squared_error(y_train,y_pred,squared=False)
print(f'RMSE on training set: {rase}')

# Streamlit Dashboard
def load_streamlit_interface():
    st.title('House Price Prediction')
    overall_qual=st.slider('Overall Quality',1,10,5)
    gr_liv_area=st.number_input('Ground Living Area(sq ft)',value=1500)
    total_sf=st.number_input('Total Square Feet(sq ft)',value=2000)
    garage_cars=st.slider('Size of Garagr(Car Capacity)',0,5,2)
    Full_bath=st.slider('Number of Full Bathrooms',0,4,2)
    year_built=st.slider('Year of Construction',1850,2020,1990)

    if st.button('Predict Price'):
        input_data=pd.DataFrame(overall_qual,gr_liv_area,total_sf,garage_cars, 
                                       columns=features)
        prediction=model.predict(input_data)
        st.success(f'Predicted Sale Price:$(Prediction[0]:,.2f)')

if __name__=='__main__':
    load_streamlit_interface
