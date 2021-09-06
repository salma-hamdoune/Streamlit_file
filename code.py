import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


header = st.container()
dataset = st.container()
features = st.container()

modelTraining = st.container()

# Code executed once
@st.cache
def get_data(filename):
	housing_data = pd.read_csv(filename)
	return housing_data

@st.cache
def train_reg(data):
	reg = LinearRegression()

	df_predictors = data[['NOX' ,'INDUS', 'ZN ']].values
	Y = data['MEDV'].values.reshape(-1,1)
	reg.fit(df_predictors, Y)

	predictions = reg.predict(df_predictors)
	return predictions
# write in header

with header:
	st.title("Welcome to my first ineraction with Streamlit!")

with dataset:
	st.header('Boston house pricing')
	st.text('This is kaggle dataset')
	df = get_data('data/housing_nona.csv')

	st.write(df.head())
	st.subheader("Values distributions of MEDV variable")
	medv_dist =np.histogram(df['MEDV'], bins=15, range=(0,24))[0]  

	# # pd.DataFrame(df['MEDV'].value_counts())
	st.bar_chart(medv_dist)



with features:
	st.header('Features')


with modelTraining:
	st.header('Model training and testing')

	disp_col, test_col = st.columns(2)
	# set_col	.slider('W')

	Y = df['MEDV'].values.reshape(-1,1)
	predictions = train_reg(df)

	disp_col.subheader('Mean absolute error of the model is: ')
	disp_col.write(mean_absolute_error(Y, predictions))

	disp_col.subheader('Mean squared error of the model is: ')
	disp_col.write(mean_squared_error(Y, predictions))

	disp_col.subheader('R squared score of the model is: ')
	disp_col.write(r2_score(Y, predictions))


	# inserting predictors by user
	test_col.subheader('Please fill in values according to each feature name')

	test_col.write('NOX')
	val_nox = test_col.number_input(label="NOX value",step=1.,format="%.2f")

	test_col.write('INDUS')
	val_indus = test_col.number_input(label="INDUS value",step=1.,format="%.2f")

	test_col.write('ZN')
	val_zn = test_col.number_input(label="ZN value",step=1.,format="%.2f")


	test_col.subheader('Model prediction')
	test_col.write(reg.predict(np.array([[val_nox, val_indus, val_zn]]))[0])
