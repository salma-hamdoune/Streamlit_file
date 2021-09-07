import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


header = st.container()
dataset = st.container()
# features = st.container()

modelTraining = st.container()
modelTesting= st.container()

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

	train_pred = reg.predict(df_predictors)
	return reg, train_pred
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



# with features:
# 	st.header('Features')


with modelTraining:
	st.header('Model training results')

	# disp_col, test_col = st.columns(2)
	# set_col	.slider('W')

	Y = df['MEDV'].values.reshape(-1,1)
	predictions = train_reg(df)[1]

	st.subheader('Mean absolute error of the model is: ')
	st.write(mean_absolute_error(Y, predictions))

	st.subheader('Mean squared error of the model is: ')
	st.write(mean_squared_error(Y, predictions))

	st.subheader('R squared score of the model is: ')
	st.write(r2_score(Y, predictions))

with modelTesting:
	st.header('Predict')
	# inserting predictors by user
	st.subheader('Please fill in values according to each feature name')

	st.write('NOX')
	val_nox = st.number_input(label="NOX value",step=1.,format="%.2f")

	st.write('INDUS')
	val_indus = st.number_input(label="INDUS value",step=1.,format="%.2f")

	st.write('ZN')
	val_zn = st.number_input(label="ZN value",step=1.,format="%.2f")


	st.subheader('Model prediction')
	st.write(train_reg(df)[0].predict(np.array([[val_nox, val_indus, val_zn]]))[0])

