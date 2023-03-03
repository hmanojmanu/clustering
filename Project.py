import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fancyimpute import IterativeImputer
from sklearn.cluster import Birch
import statistics as sts
import streamlit as st


st.set_page_config(layout="wide")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.marshallsindia.com/ImageBuckets/ItemImages/ZA%20403.jpg?id=75");
             background-attachment: fixed;
	     #background-position: 25% 75%;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

st.title('Model Deployment: Clustering')
st.sidebar.header('Input Country listed')

input = st.sidebar.selectbox("Select Country from list",('Afghanistan','Albania','Algeria','American Samoa','Andorra','Angola','Antigua and Barbuda',
							 'Argentina','Armenia','Aruba','Australia','Austria','Azerbaijan','Bahamas, The','Bahrain',
							 'Bangladesh','Barbados','Belarus','Belgium','Belize','Benin','Bermuda','Bhutan','Bolivia',
							 'Bosnia and Herzegovina','Botswana','Brazil','Brunei Darussalam','Bulgaria','Burkina Faso',
							 'Burundi','Cambodia','Cameroon','Canada','Cayman Islands','Central African Republic','Chad',
							 'Chile','China','Colombia','Comoros','Congo, Dem. Rep.','Congo, Rep.','Costa Rica','Cote d',
							 'Croatia','Cuba','Curacao','Cyprus','Czech Republic','Denmark','Djibouti','Dominica',
							 'Dominican Republic','Ecuador','Egypt, Arab Rep.','El Salvador','Equatorial Guinea',
							 'Eritrea','Estonia','Ethiopia','Faeroe Islands','Fiji','Finland','France','French Polynesia',
							 'Gabon','Gambia','Georgia','Germany','Ghana','Greece','Greenland','Grenada','Guam',
							 'Guatemala','Guinea','Guinea-Bissau','Guyana','Haiti','Honduras','Hong Kong SAR, China',
							 'Hungary','Iceland','India','Indonesia','Iran, Islamic Rep.','Iraq','Ireland','Isle of Man',
							 'Israel','Italy','Ivoire','Jamaica','Japan','Jordan','Kazakhstan','Kenya','Kiribati',
							 'Korea, Dem. Rep.','Korea, Rep.','Kosovo','Kuwait','Kyrgyz Republic','Lao PDR','Latvia',
							 'Lebanon','Lesotho','Liberia','Libya','Liechtenstein','Lithuania','Luxembourg','Macao SAR, China',
							 'Macedonia, FYR','Madagascar','Malawi','Malaysia','Maldives','Mali','Malta','Marshall Islands',
							 'Mauritania','Mauritius','Mexico','Micronesia, Fed. Sts.','Moldova','Monaco','Mongolia',
							 'Montenegro','Morocco','Mozambique','Myanmar','Namibia','Nepal','Netherlands','New Caledonia',
							 'New Zealand','Nicaragua','Niger','Nigeria','Norway','Oman','Pakistan','Panama','Papua New Guinea',
							 'Paraguay','Peru','Philippines','Poland','Portugal','Puerto Rico','Qatar','Romania',
							 'Russian Federation','Rwanda','Samoa','San Marino','Sao Tome and Principe','Saudi Arabia','Senegal',
							 'Serbia','Seychelles','Sierra Leone','Singapore','Sint Maarten (Dutch part)','Slovak Republic',
							 'Slovenia','Solomon Islands','Somalia','South Africa','South Sudan','Spain','Sri Lanka',
							 'St. Kitts and Nevis','St. Lucia','St. Martin (French part)','St. Vincent and the Grenadines',
							 'Sudan','Suriname','Swaziland','Sweden','Switzerland','Syrian Arab Republic','Tajikistan',
							 'Tanzania','Thailand','Timor-Leste','Togo','Tonga','Trinidad and Tobago','Tunisia','Turkey',
							 'Turkmenistan','Turks and Caicos Islands','Uganda','Ukraine','United Arab Emirates',
							 'United Kingdom','United States','Uruguay','Uzbekistan','Vanuatu','Venezuela, RB','Vietnam',
							 'Virgin Islands (U.S.)','Yemen, Rep.','Zambia','Zimbabwe'))

#Model Prediction - Deployment

data = pd.read_csv("https://raw.githubusercontent.com/abhishekgd96/Project-Classification/main/World_development_mesurement.csv")

data['Business Tax Rate'] = data['Business Tax Rate'].str.replace('%', '')
data['Business Tax Rate'] = data['Business Tax Rate'].astype(float)

data['GDP'] = data['GDP'].str.replace('$', '')
data['GDP'] = data['GDP'].str.replace(',', '')
data['GDP'] = data['GDP'].astype(float)

data['Health Exp/Capita'] = data['Health Exp/Capita'].str.replace('$', '')
data['Health Exp/Capita'] = data['Health Exp/Capita'].str.replace(',', '')
data['Health Exp/Capita'] = data['Health Exp/Capita'].astype(float)

data['Tourism Inbound'] = data['Tourism Inbound'].str.replace('$', '')
data['Tourism Inbound'] = data['Tourism Inbound'].str.replace(',', '')
data['Tourism Inbound'] = data['Tourism Inbound'].astype(float)

data['Tourism Outbound'] = data['Tourism Outbound'].str.replace('$', '')
data['Tourism Outbound'] = data['Tourism Outbound'].str.replace(',', '')
data['Tourism Outbound'] = data['Tourism Outbound'].astype(float)




data1 = data.drop('Country', axis=1)


# calling the  MICE class
mice_imputer = IterativeImputer()
# imputing the missing value with mice imputer
data1 = mice_imputer.fit_transform(data1)


df = pd.DataFrame(data1, columns = ['Birth Rate',
                                    'Business Tax Rate',
                                    'CO2 Emissions',
                                    'Days to Start Business',
                                    'Ease of Business',
                                    'Energy Usage',
                                    'GDP',
                                    'Health Exp % GDP',
                                    'Health Exp/Capita',
                                    'Hours to do Tax',
                                    'Infant Mortality Rate',
                                    'Internet Usage',
                                    'Lending Interest',
                                    'Life Expectancy Female',
                                    'Life Expectancy Male',
                                    'Mobile Phone Usage',
                                    'Number of Records',
                                    'Population 0-14',
                                    'Population 15-64',
                                    'Population 65+',
                                    'Population Total',
                                    'Population Urban',
                                    'Tourism Inbound',
                                    'Tourism Outbound'])

scaler = StandardScaler()
df_std = scaler.fit_transform(df)

df['Country'] = data['Country']
df = df[ ['Country'] + [ col for col in df.columns if col != 'Country' ] ]
df_country =df[df['Country']==input]

st.write(df_country)

df_country = df_country.drop(["Country"], axis = 1)

#Creating a BIRCH model 
model = Birch(branching_factor = 50, n_clusters = 4, threshold = 1.5)
model.fit(df_std)

pred = model.predict(df_country)
pred = sts.mode(pred)
st.write('This country belongs to cluster number :',pred)
