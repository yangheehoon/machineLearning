# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
#가설검정

#귀무가설: 직장에서 직급과 근무 시간, 정신적 피로도는 번아웃에 연관성이 없다. 
#대립가설: 직장에서 직급과 근무 시간, 정신적 피로도는 번아웃에 연관성이 있다. 

# Employee ID: 사원별로 부여된 고유 ID (예: fffe390032003000 )
# Date of Joining: 직원 입사 날짜-시간 (예: 2008-12-30 )
# Gender: 사원의 성별( 남/여 )
# Company Type: 직원이 근무하는 회사의 유형( 서비스/제품 )
# WFH Setup Available: 직원이 재택근무 시설을 이용할 수 있습니까( 예/아니오 )
# Designation: 조직에서 일하는 직원의 직급.
# [0.0, 5.0] 범위에서 클수록 높은 직급
# Resource Allocation: 직원에게 할당된 자원의 양, 즉. 근무 시간
# [1.0, 10.0] 범위 높을수록 더 많은 근무 시간
# Mental Fatigue Score: 직원이 직면하고 있는 정신적 피로도.
# [0.0, 10.0] 범위 에서 0.0은 피로가 없음을 의미하고 10.0은 완전히 피로함을 의미
# Burn Rate: 근무 중 Bur out의 비율을 말하는 각 직원에 대해 예측해야 하는 값 (종속변수임)
# [0.0, 1.0] 범위 에서 값이 높을수록 번아웃


import pandas  as  pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#데이터 로드
burn = pd.read_csv("e:/data/train.csv")
burn

#컬럼명에 공백이 들어간 컬럼 언더바로 대체
burn.columns = [i.replace(" ","_") for i in list(burn.columns)]

#결측치 확인
burn.isnull().sum()

# Employee_ID                0
# Date_of_Joining            0
# Gender                     0
# Company_Type               0
# WFH_Setup_Available        0
# Designation                0
# Resource_Allocation     1381
# Mental_Fatigue_Score    2117
# Burn_Rate               1124

#결측치 행 제거
burn=burn.dropna()

#이상치 확인
def outlier_value(x):  # 이상치를 확인하는 함수
    f, axes = plt.subplots(1,4,figsize=(16,4))  #다수의 그래프를 한행에 출력하기 위해 subplots 생성
    cnt=0 #숫자형 컬럼데이터의 초기 인덱스 값 설정
    for  i  in  x.columns[x.dtypes=='float64']:  # 숫자형 컬럼 반복문 실행
        ax = axes[cnt] # subplots의 인덱스 열 지정
        Q1 = x[i].quantile(0.25)  # 컬럼데이터의 25%
        Q3 = x[i].quantile(0.75)  # 컬럼데이터의 75%
        IQR = Q3 - Q1  # 사분위수 범위값
        print ('%s 이상치 개수 : %d개' %(i, x[i][ (x[i] > Q3 + IQR*5) | ( x[i] < Q1-IQR*5)].count()) ) 
        ax.boxplot(x[i]) #사분위수 상자 그래프 생성 
        ax.set_title(i,size = '20') # 그래프 제목   
        cnt+=1 #다음 열을 위한 인덱스값 1 증가 
    plt.style.use('bmh')    
    plt.show() #그래프 출력    
    
outlier_value(burn)

#종속변수의 분포 형태 확인
# sns.distplot(burn['Burn_Rate'], kde=True, rug=True , color='blue',kde_kws={"color": "red"},rug_kws={"color": "purple"})
# sns.displot(burn['Burn_Rate'], kde=True, rug=True , color='blue')
ax = sns.histplot(burn['Burn_Rate'], kde=True, color='green')
ax.lines[0].set_color('r')
plt.show()

#데이터 정규화
from  sklearn.preprocessing  import  MinMaxScaler 
burn2 = burn[['Designation', 'Resource_Allocation', 'Mental_Fatigue_Score', 'Burn_Rate']]
scaler = MinMaxScaler()
#정규화를 하기위한 계산
scaler.fit(burn2) 
burn = scaler.transform(burn2)
#정규화한 값으로 burn 재정의
burn=pd.DataFrame(burn,columns = burn2.columns)
display(burn.describe())

#독립변수와 종속변수 상관관게 확인 
sns.heatmap(burn.corr(),annot=True, cmap = 'Oranges', linewidths=0.2)
fig = plt.gcf()   
fig.set_size_inches(10,8)  
plt.show()

import statsmodels.formula.api as smf
import statsmodels.api as sm

#다중 회귀 모델을 생성
model = smf.ols( formula='Burn_Rate ~ Designation + Resource_Allocation + Mental_Fatigue_Score',   data=burn)

#다중공선성 여부 확인
from  statsmodels.stats.outliers_influence  import  variance_inflation_factor

model.exog_names  # 모델에서 보이는 컬럼명과 컬럼순서를 확인

variance_inflation_factor( model.exog, 1) # 직급의 팽창계수
variance_inflation_factor( model.exog, 2) # 근무 시간의 팽창계수
variance_inflation_factor( model.exog, 3) # 정신적 피로도의 팽창계수

for  i,  k  in  zip(model.exog_names[1:], range(1,4)):
    if variance_inflation_factor( model.exog, k) >10 :
        print('%s 다중공선성 있음  =====>  팽창계수 : %f' %(i,variance_inflation_factor( model.exog, k)))
    else :
        print('%s 다중공선성 없음  =====>  팽창계수 : %f' %(i,variance_inflation_factor( model.exog, k)))    
        
#모델 훈련
result = model.fit()

#분석결과를 해석
display(result.summary() )  #p_value가 0.05보다 작아 대립가설 채택

#테스트 데이터 로드
test_data = pd.read_csv('e:/data/test.csv')

#컬럼명에 공백이 들어간 컬럼 언더바로 대체
test_data.columns = [i.replace(" ","_") for i in list(test_data.columns)]

#테스트 데이터 정규화
from  sklearn.preprocessing  import  MinMaxScaler 
test_data2 = test_data[['Designation','Resource_Allocation','Mental_Fatigue_Score']]
scaler = MinMaxScaler()
#정규화를 하기위한 계산
scaler.fit(test_data2) 
test_data = scaler.transform(test_data2)
#정규화한 값으로 test_data 재정의
test_data=pd.DataFrame(test_data,columns = test_data2.columns)
display(test_data.describe())

#생성된 다중 회귀 모델을 통해 burn out 지수 예측
test_result = result.predict(test_data[['Designation','Resource_Allocation','Mental_Fatigue_Score']])
display(pd.DataFrame(test_result*10 ,columns=['Predict_Burn_Rate']))

from   sklearn.model_selection  import  train_test_split

#독립변수 
x=burn.iloc[:,:-1]
#종속변수
y=burn.iloc[:,-1]

#해당 데이터를 훈련과 테스트 9대1로 나눔
x_train, x_test, y_train, y_test = train_test_split( x, y ,test_size=0.1, random_state=1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# LinearRegression 다중회귀 분석 모델을 생성
from  sklearn.linear_model   import  LinearRegression

model2 = LinearRegression()
model2.fit( x_train, y_train )

#LinearRegression 모델의 번아웃 지수 예측
test_result2 = model2.predict(x_test)
test_result2

#결정계수 
print('LinearRegression 모델의 결정계수 : =========>%.16f' %(model2.score(x_train,y_train)))
#상관관계
print ('LinearRegression 모델의 상관관계 : =========>%.16f' %(np.corrcoef( y_test, test_result2 )[0][1]))   


#실제 정답값과 예측값의 오차 소수점 16자리까지 출력
from  sklearn  import  metrics
print('평균 제곱 오차 : =========> %.16f' %(metrics.mean_squared_error(y_test, test_result2)))

#한글 코드
from matplotlib import font_manager, rc
font = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font)

#예측값과 실제값의 일부 비교 라인그래프
plt.figure(figsize=(13,4))
#예측값과 실제값 각각 100개 그래프화
plt.plot(test_result2[:100],marker='o',color='r',alpha=0.5,label='예측값')
plt.plot(np.array(y_test)[:100],marker='o',color='b',alpha=0.5,label='실제값')
plt.ylabel('번아웃 지수')
plt.xlabel('데이터 넘버')
plt.legend()
plt.title("실제값과 예측값 일부 비교")  
plt.show()

#예측값과 실제값 산점도 확인 
#그래프 정사각형
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
#산포도 그래프
plt.scatter(np.array(y_test), test_result2, c='green', alpha=0.4, edgecolor='white')    
plt.plot(np.array(y_test), np.array(y_test),color='red')
plt.title("실제값과 예측값 산포도 그래프")
plt.ylim(0.0,1.0)
plt.xlim(0.0,1.0)
plt.ylabel('예측값')
plt.xlabel('실제값')
plt.show()

#가설검정을 하기 위한 p_value값을 담을 딕셔너리 변수 선언
null_hypothesis={}
alternative_hypothesis={}

# p_value값 출력 및 각각의 가설 딕셔너리에 맞게 저장
for i in burn.columns[burn.dtypes=='float64'][0:-1].tolist():
    #과학적 표기법 대신 소수점 16자리까지 나타낸다.
    print ('%s의 p_value 값 : %s' %(i,format(result.pvalues[i],'.16f')))
    #가설검정 위해 컬럼명과 p_value 값을 유의수준 0.05에 비교하여 각각의 딕셔너리에 담는다
    if float(format(result.pvalues[i],'.16f')) > 0.05:
        null_hypothesis[i]=format(result.pvalues[i],'.16f')
    else:
        alternative_hypothesis[i]=format(result.pvalues[i],'.16f')

#가설검정
if not null_hypothesis :
    if len(alternative_hypothesis) > 1 :
        print('%s의 p_value 값들이 각각 %s므로 유의수준 0.05보다 작아 대립가설 채택' 
              %(list(alternative_hypothesis.keys()),list(alternative_hypothesis.values())))
    else:
        print('%s의 p_value 값이 %s므로 유의수준 0.05보다 작아 대립가설 채택' 
              %(list(alternative_hypothesis.keys()),list(alternative_hypothesis.values())))
else :
    if len(null_hypothesis) > 1 :
        print('%s의 p_value 값들이 각각 %s므로 유의수준 0.05보다 커 귀무가설 채택' 
              %(list(null_hypothesis.keys()),list(null_hypothesis.values())))
    else:
        print('%s의 p_value 값이 %s므로 유의수준 0.05보다 커 귀무가설 채택' 
              %(list(null_hypothesis.keys()),list(null_hypothesis.values())))

