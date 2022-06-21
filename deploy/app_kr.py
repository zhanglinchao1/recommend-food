import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

st.title("음식 추천 시스템")
st.text("주문을 도와드리겠습니다")
st.image("food.jpg")

## nav = st.sidebar.radio("Navigation",["Home","IF Necessary 1","If Necessary 2"])
st.subheader("  <내용 기반 으로 추천>")
st.subheader("당신의 좋아하는 음식은 무엇입니까?")
vegn = st.radio("Vegetables or none!",["veg","non-veg"],index = 1) 

st.subheader("어떤 요리를 좋아하세요?")
cuisine = st.selectbox("Choose your favourite!",['Korea', 'Snack', 'Dessert', 'Japanese', 'Indian', 'Healthy Food',
       'Mexican', 'Italian', 'Chinese', 'Beverage', 'Thai'])


st.subheader("요리 평가 점수를 선택하세요.")  #RATING
val = st.slider("from poor to the best!",0,10)

food = pd.read_csv("../input/food.csv")
ratings = pd.read_csv("../input/ratings.csv")
combined = pd.merge(ratings, food, on='Food_ID')
#ans = food.loc[(food.C_Type == cuisine) & (food.Veg_Non == vegn),['Name','C_Type','Veg_Non']]

ans = combined.loc[(combined.C_Type == cuisine) & (combined.Veg_Non == vegn)& (combined.Rating >= val),['Name','C_Type','Veg_Non']]
names = ans['Name'].tolist()
x = np.array(names)
ans1 = np.unique(x)

finallist = ""
bruh = st.checkbox("요리를 선택하세요.(이곳의 메뉴는 향후 개인 구매 기록을 통해 자동 생성된다.)")
if bruh == True:
    finallist = st.selectbox("수동 선텍 메뉴",ans1)


##### IMPLEMENTING RECOMMENDER ######
dataset = ratings.pivot_table(index='Food_ID',columns='User_ID',values='Rating')
dataset.fillna(0,inplace=True)
csr_dataset = csr_matrix(dataset.values)
dataset.reset_index(inplace=True)

model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model.fit(csr_dataset)

def food_recommendation(Food_Name):
    n = 10
    FoodList = food[food['Name'].str.contains(Food_Name)]  
    if len(FoodList):        
        Foodi= FoodList.iloc[0]['Food_ID']
        Foodi = dataset[dataset['Food_ID'] == Foodi].index[0]
        distances , indices = model.kneighbors(csr_dataset[Foodi],n_neighbors=n+1)    
        Food_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        Recommendations = []
        for val in Food_indices:
            Foodi = dataset.iloc[val[0]]['Food_ID']
            i = food[food['Food_ID'] == Foodi].index
            Recommendations.append({'Name':food.iloc[i]['Name'].values[0],'Distance':val[1]})
        df = pd.DataFrame(Recommendations,index=range(1,n+1))
        return df['Name']
    else:
        return "No Similar Foods."


display = food_recommendation(finallist)
#names1 = display['Name'].tolist()

#x1 = np.array(names)
#ans2 = np.unique(x1)
st.subheader("구매 내역 기반으로 추천:")
if bruh == True:
    bruh1 = st.button("Click it! ")
    n =0
    if bruh1 == True:
        for i in display:
            n=n+1
            st.write('추천메뉴',n,':',i)
