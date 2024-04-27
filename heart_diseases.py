import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
# %pip install h5py graphviz pydot


data = pd.read_csv('heart.csv')
data.head()
df = data.copy()
ds = data.copy()
# ds.info()

# categorical = ds.select_dtypes(exclude = 'number')
# numerical = ds.select_dtypes(include = 'number')

# print(f"\t\t\t\t\tCategorical ds")
# display(categorical.head(3))

# print(f"\n\n\t\t\t\t\tNumerical ds")
# display(numerical.head(3))

import pickle
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Create LabelEncoder and StandardScaler instances
lbl_en = defaultdict(LabelEncoder)
scaler = StandardScaler()

columns_to_encode = ['Sex', 'ChestPainType','ST_Slope']
columns_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'] 

# Apply LabelEncoder to categorical columns
ds[columns_to_encode] = ds[columns_to_encode].apply(lambda x: lbl_en[x.name].fit_transform(x))

# Apply StandardScaler to numeric columns
ds[columns_to_scale] = scaler.fit_transform(ds[columns_to_scale])

# Save LabelEncoder, StandardScaler, and other necessary information
filename = 'labSca.sav'
data_to_save = {
    'label_encoders': dict(lbl_en),
    'scaler': scaler,
    'columns_to_encode': columns_to_encode,
    'columns_to_scale': columns_to_scale
    # Add any other information you want to save
}

pickle.dump(data_to_save, open(filename, 'wb'))

# x = ds
# y = df.HeartDisease
# #using XGBOOST to find feature importance
# import xgboost as xgb
# model = xgb.XGBClassifier()
# model.fit(x,y)

# # first feature importance scores
# xgb.plot_importance(model)


# # feature selection
# selected_columns = ['Age', 'Sex', 'Cholesterol', 'MaxHR', 'RestingBP', 'Oldpeak', 'ChestPainType', 'ST_Slope']
# new_ds = ds[selected_columns]
# new_ds.head()


# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report


# x_train, x_test, y_train, y_test = train_test_split(new_ds, y, test_size = 0.10, random_state = 47, stratify = y)
# print(f'x_train: {x_train.shape}')
# print(f'x_test: {x_test.shape}')
# print('y_train: {}'.format(y_train.shape))
# print('y_test: {}'.format(y_test.shape))


# DEEPE LEARNING MODEL

# model = tf.keras.Sequential([ 
#     tf.keras.layers.Dense(units=12, activation='relu'),
#     tf.keras.layers.Dense(20, activation='relu'), 
#     tf.keras.layers.Dense(20, activation='relu'), 
#     tf.keras.layers.Dense(1, activation='sigmoid') 
# ])
# model.compile(optimizer='adam',
#               loss = 'binary_crossentropy', 
#               metrics=['accuracy']) 

# model.fit(x_train, y_train, epochs=15) 

# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.metrics import classification_report

# revealer = confusion_matrix(y_pred, y_test)
# sns.set(style = 'darkgrid')
# sns.heatmap(revealer/np.sum(revealer), annot=True, cmap='crest', fmt='.1%', linewidth=1)

# print(classification_report(y_pred, y_test))

# model.save('heartfailurepred.h5')




import streamlit as st
import pickle
from tensorflow.keras.models import load_model
model = load_model('heartfailurepred.h5')

st.sidebar.image('pngwing.com.png', width = 200,)
st.sidebar.markdown('<br>', unsafe_allow_html=True)
selected_page = st.sidebar.radio('Navigation', ['Home', 'Modeling'])

def HomePage():
    # Streamlit app header
    st.markdown("<h1 style = 'color: #2B2A4C; text-align: center; font-family:montserrat'>Heart Failure Prediction Model</h1>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h6 style = 'margin: -15px; color: #2B2A4C; text-align: center ; font-family:montserrat'>The Heart Failure Prediction Model is a machine learning algorithm designed to forecast the likelihood of heart failure in individuals based on various health-related factors. By analyzing features such as age, Sex, medical history, and lifestyle habits, the model provides valuable insights into the risk of heart failure, allowing for early intervention and preventive measures.</h6>",unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.image('2150941719.jpg', width=700, align="center")
    st.markdown('<br>', unsafe_allow_html= True)



    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h3 style='color: #2B2A4C;text-align: center; font-family:montserrat'>The Model Features</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Sex</h3>", unsafe_allow_html=True)
    st.markdown("<p>Sex refers to the biological sex of the individual, which can have an impact on their susceptibility to diabetes. There are three</p>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html= True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Age</h3>", unsafe_allow_html=True)
    st.markdown("<p>The Age field in the Heart Diseases Model denotes the individual's chronological age, a key factor in assessing heart disease risk. Age ranges from 28-77 in our dataset.</p>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Cholesterol </h3>", unsafe_allow_html=True)
    st.markdown("<p>Cholesterol is a type of fat that is found in your blood. It's essential for building cells and making certain hormones, but having too much cholesterol can be harmful, especially for your heart. Levels between 100-129 mg/dL are considered near optimal, while Levels above 160 mg/dL are considered high.</p>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>MAXHR </h3>", unsafe_allow_html=True)
    st.markdown("<p>MAXHR stands for Maximum Heart Rate. It refers to the highest heart rate an individual can achieve during physical exertion. MAXHR is an important metric in assessing cardiovascular health and fitness levels.</p>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Chest pain type</h3>", unsafe_allow_html=True)
    st.markdown("<p>Chest pain or discomfort caused when your heart muscle doesn't get enough oxygen-rich blood.</p>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>RestingBP</h3>", unsafe_allow_html=True)
    st.markdown("<p>Resting blood pressure refers to the measurement of blood pressure when the body is at rest, typically in a seated or lying position and after a period of relaxation. Normal resting blood pressure typically falls below 120/80 mmHg</p>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #2B2A4C;text-align: left; font-family:montserrat'>Oldpeak </h3>", unsafe_allow_html=True)
    st.markdown("<p>Oldpeak is a term used when doctors are checking the heart's health. They might ask you to exercise on a treadmill or bike while they monitor your heart. They're looking for any unusual changes on a graph called an electrocardiogram (ECG). One thing they look for is if a part of the graph dips down (depression) when you're exercising compared to when you're resting. This dip is called Oldpeak. If there's a big dip, it could mean your heart isn't getting enough blood, which could signal a problem like blocked arteries.</p>", unsafe_allow_html=True)

   

    # Streamlit app footer
    st.markdown("<p style='text-align: LEFT; font-size: 12px;'>Created with ❤️ by the Orpheus Snipersat the Cavista Hackathon </p>", unsafe_allow_html=True)

# Function to define the modeling page content
def modeling_page():
    st.markdown("<h1 style='text-align: CENTER; color: #2B2A4C;'>Modelling Section </h1>", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: LEFT; color: #2B2A4C;'>Dataset Sample</h1>", unsafe_allow_html=True)
    # st.sidebar.markdown('<br><br><br>', unsafe_allow_html= True)
    st.write(df[['Age', 'Sex', 'Cholesterol', 'MaxHR','ChestPainType', 'RestingBP', 'Oldpeak', 'ST_Slope', 'HeartDisease']])



if selected_page == "Home":
    HomePage()
elif selected_page == "Modeling":
    st.sidebar.markdown('<br>', unsafe_allow_html= True)
    modeling_page()


if selected_page == "Modeling":
    st.sidebar.markdown("Add your input here")
    Age = st.sidebar.number_input("Age",0,1000)
    Sex = st.sidebar.selectbox("Sex", df['Sex'].unique())
    Cholesterol = st.sidebar.number_input("Cholesterol",0,1000)
    MaxHR = st.sidebar.number_input('MaxHR',0,1000)
    ChestPainType = st.sidebar.selectbox("ChestPainType", df['ChestPainType'].unique())
    RestingBP = st.sidebar.number_input("RestingBP",0,1000)
    Oldpeak = st.sidebar.number_input("Oldpeak",  0.0, 10.0, format="%.1f")      
    ST_Slope = st.sidebar.selectbox("ST_Slope", df['ST_Slope'].unique())
    st.sidebar.markdown('<br>', unsafe_allow_html= True)
    
    st.markdown("<h2 style='text-align: LEFT; color: #z2B2A4C;'>Input Patient Name</h2>", unsafe_allow_html=True)
    patient_name = st.text_input("")
    
    input_variables = pd.DataFrame([{
        'Age': Age,
        'Sex':Sex,
        'Cholesterol': Cholesterol,
        'MaxHR': MaxHR, 
        'ChestPainType': ChestPainType,
        'RestingBP': RestingBP,
        'Oldpeak': Oldpeak, 
        'ChestPainType': ChestPainType,
        'ST_Slope': ST_Slope
    }])


    st.markdown("<h2 style='text-align: LEFT; color: #z2B2A4C;'>Your Input Appears Here</h2>", unsafe_allow_html=True)
    st.write(input_variables)
    
# Assuming input_variables is your DataFrame with data to be transformed
    
    import pickle
    import pandas as pd
    
    # Load the saved model and scalers
    filename = 'labSca.sav'
    with open(filename, 'rb') as file:
        saved_data = pickle.load(file)
    
    label_encoders = saved_data['label_encoders']
    scaler = saved_data['scaler']
    columns_to_encode = saved_data['columns_to_encode']
    columns_to_scale = saved_data['columns_to_scale']


    # Transform categorical columns using label encoders
    for col, encoder in label_encoders.items():
        # Reorder input_variables columns to match the order used during fitting
        input_variables = input_variables[columns_to_encode + columns_to_scale]
        input_variables[col] = encoder.transform(input_variables[col])
    

    if patient_name:
        if st.button('Press To Predict'):
            st.markdown("<h4 style='color: #2B2A4C; text-align: left; font-family: montserrat;'>Model Report</h4>", unsafe_allow_html=True)
            predicted = model.predict(input_variables)
            st.toast('Predicted Successfully')
            st.image('check icon.png', width=100)
            if predicted >= 0.5:
                st.error(f"{patient_name} you may be at high risk of having Heart Disease!")
            else:
                st.success(f"{patient_name} is at low risk of Heart Disease")
    else:
        st.warning("Please enter the patient's name.")

    
    st.markdown('<hr>', unsafe_allow_html=True)
    
    st.markdown("<h8 style = 'color: #2B2A4C; text-align: LEFT; font-family:montserrat'>Heart Disease MODEL BUILT BY Pheau Snipers</h8>",unsafe_allow_html=True)

