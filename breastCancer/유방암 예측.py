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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    

#한글 코드
from matplotlib import font_manager, rc
font = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font)

#데이터 로드
b_cancer = pd.read_csv("e:/cancer/breast-cancer.csv")
b_cancer.head()

#결측치 확인

print(f'{b_cancer.isnull().sum()}\n')

#독립변수 선언
x = b_cancer.iloc[ :  , 2: ]
#종속변수 선언
y = b_cancer['diagnosis']

#몇몇 분류 알고리즘의 경우 분류일지라도 문자대신 숫자 형태로 데이터를 학습시켜야하기에
#종속변수의 악성을 1 양성을 0으로 형변환 시켜줍니다, ex)리퍼알고리즘, xgboost 
cnt =0
for i in b_cancer['diagnosis']:
    if i =='M':
        b_cancer['diagnosis'].values[cnt]=1
    else :
        b_cancer['diagnosis'].values[cnt]=0
    cnt+=1
    
plt.style.use('bmh') #그래프 시트 설정

#이상치 행들을 담을 리스트 선언
outlier_row=[]

#이상치 확인 함수 정의
def outlier_value(x):  # 이상치를 확인하는 함수
    f, axes = plt.subplots(6,5,figsize=(16,14),constrained_layout=True,facecolor='#A8F552')  #subplots 생성
    cnt=0 #숫자형 컬럼데이터의 서브플롯 초기 인덱스 열값 설정
    cnt2=0 #숫자형 컬럼데이터의 서브플롯 초기 인덱스 행값 설정
    for  i  in  x.columns[x.dtypes=='float64']:  # 숫자형 컬럼 반복문 실행
        ax = axes[cnt2][cnt] # 서브플롯의 인덱스 행열 지정
        Q1 = x[i].quantile(0.25)  # 컬럼데이터의 25%
        Q3 = x[i].quantile(0.75)  # 컬럼데이터의 75%
        IQR = Q3 - Q1  # 사분위수 범위값
        #보통의 경우 Q3 +1.5 * IQR 이상 Q1 - 1.5 * IQR 이하인 경우 이상치로 판단하지만 
        #해당 데이터의 경우 최극단의 이상치만 거르기위해 1.5 대신 5을 곱하여 이상치 검증을 하였다
        print ('%s 이상치 개수 : %d개' %(i, x[i][ (x[i] > Q3 + IQR*5) | ( x[i] < Q1-IQR*5)].count()) )
        #이상치 행들을 리스트에 담아준다
        for j in x[ (x[i] > Q3 + IQR*5) |  (x[i] < Q1-IQR*5) ].index:
            outlier_row.append(j)
        ax.boxplot(x[i]) #사분위수 상자 그래프 생성 
        ax.set_title(i,size = '16',color='r') # 그래프 제목 
        ax.set_xticklabels([i]) # 그래프의 x축명 변경
        cnt+=1 #다음 열을 위한 인덱스값 1 증가 
        if cnt == 5: #서브플롯 행열값 변경을 위한 조건문
            cnt = 0
            cnt2+=1           
    plt.show() #그래프 출력    
    
#이상치 확인 함수 호출
outlier_value(x)    

#이상치 제거
x.drop(index=set(outlier_row),inplace=True)
y.drop(index=set(outlier_row),inplace=True)    

#이상치 제거 후 인덱스 초기화
x=x.reset_index(drop=True)
y=y.reset_index(drop=True)

# 정규화 작업 수행
from sklearn.preprocessing  import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x) 

x2= scaler.transform(x)
x2=pd.DataFrame(x2,columns=x.columns)

import seaborn as sns

#전처리 완료 후 다중상자그림
plt.figure(figsize=(17,7), facecolor='#BEEFFF')
sns.boxplot(data=x2) 
plt.xticks(rotation = +65, fontweight='bold') 
plt.show()

from  sklearn.naive_bayes  import  GaussianNB    # 나이브 베이즈 분류 모델
from  sklearn.linear_model  import  LogisticRegression  # 로지스틱회귀 분류 모델
from  sklearn.ensemble  import  RandomForestClassifier  #배깅 앙상블이 적용된 랜덤포레스트 분류 모델
from sklearn.neural_network import MLPClassifier # 다층 신경망 분류 모델
from sklearn.svm import SVC # 서포트 벡터 분류 모델
from sklearn.neighbors import KNeighborsClassifier # knn 분류 모델
from sklearn.tree import DecisionTreeClassifier # 의사결정트리 분류 모델
# import  wittgenstein  as  lw  # 리퍼 분류 모델
import xgboost as xgb  #부스팅 앙상블이 적용된 xgboost 분류 모델
from  sklearn.ensemble  import  VotingClassifier   # 여러개의 분류 모델들을 앙상블로 쓰기 위한 모듈
                                                                
from  sklearn  import  metrics    

#알고리즘별 모델 구현
NaiveBayes = GaussianNB(var_smoothing=0.0001)
LogisticRegression = LogisticRegression(solver='lbfgs', max_iter = 900,random_state =1)
RandomForest_C = RandomForestClassifier(random_state =1,n_estimators=100)
SupportVector_C = SVC(kernel='rbf', C=100, gamma=0.5, probability=True)
MLPerceptrons_C = MLPClassifier(hidden_layer_sizes=(10,10), activation='relu', solver='sgd', 
                    learning_rate_init=0.1, max_iter=1000, random_state=1)
KNN = KNeighborsClassifier(n_neighbors=7)
DecisionTree_C = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=1)
# Ripper = lw.RIPPER(random_state=1)
XGboost = xgb.XGBClassifier() 

Voting_C = VotingClassifier( estimators = [ ('gnb',NaiveBayes),('lr',LogisticRegression),('rf',RandomForest_C),
                                           ('svm', SupportVector_C ),('mlp', MLPerceptrons_C ),('knn',KNN),
                                           ('dt',DecisionTree_C),('xgb', XGboost) ], voting='soft' )


#훈련데이터와 테스트데이터 분리
from  sklearn.model_selection  import  train_test_split
x_train, x_test, y_train, y_test = train_test_split( x2, y, test_size=0.1, random_state=1)


m_algorithm = [NaiveBayes,LogisticRegression,RandomForest_C,SupportVector_C,
               MLPerceptrons_C,KNN,DecisionTree_C,XGboost,Voting_C]
algorithm_n = ['NaiveBayes','LogisticRegression','RandomForest_C','SupportVector_C',
               'MLPerceptrons_C','KNN','DecisionTree_C','XGboost','Voting_C']

m_accuracy = {}

#각 분류 알고리즘 별 혼동행렬,ROC곡선,실제값과예측값 비교 시각화
for i,j,c in zip(m_algorithm,algorithm_n,range(len(algorithm_n))) :
    i.fit( x_train, list(y_train) )
    test_result = i.predict( x_test )
    accuracy = sum( test_result == y_test ) / len(y_test)
    m_accuracy[j]=accuracy
    
    from  sklearn.metrics  import  confusion_matrix
    tn, fp, fn, tp =  confusion_matrix( list(y_test), test_result ).ravel() 

    from  sklearn.metrics  import  cohen_kappa_score
    
    # 모델 평가
    kappa = cohen_kappa_score( list(y_test), test_result )
    sensitivity =  tp /(tp+fn)                                              
    specificity =  tn /(tn+fp)
    precision =   tp /(tp+fp)
    f1_score = (2*tp) / ( 2*tp+fp+fn )

    #혼동행렬
    matrix=confusion_matrix(list(y_test), test_result)
    
    TFPN = ['True Neg','False Pos','False Neg','True Pos']
    TFPNcnt = list(matrix.flatten())
    
    labels = [f'{i}\n{j}' for i, j in zip(TFPN,TFPNcnt)]
    labels = np.asarray(labels).reshape(2,2)
    
    plt.figure(figsize=(7,5))
    plt.title('[ '+j+' ] confusion matrix')
    sns.heatmap(matrix,annot=labels ,fmt='s')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()
    
    #AUC 스코어
    from  sklearn  import  metrics

    fpr, tpr , threshold  = metrics.roc_curve( list(y_test), test_result )
    roc_auc = metrics.auc( fpr, tpr )
    roc_auc  

    #ROC 곡선
    plt.figure(figsize=(7,4),facecolor='pink')
    plt.title('[ '+j+' ] ROC 곡선')
    plt.plot( fpr, tpr , 'blue', label='AUC = %0.2f' % roc_auc )
    plt.legend(bbox_to_anchor=(1.02, 1.13))
    plt.plot([0,1], [0,1], 'r--') 
    plt.show()
    
    #알고리즘별 예측값과 실제값 각각 50개 비교 그래프
    plt.figure(figsize=(17,4),facecolor='black')
    plt.plot(np.array(y_test)[:50],linestyle='none',marker='o',color='b')
    plt.plot(test_result[:50],linestyle='none',marker='o', color='r')

    above_threshold = np.array(y_test)[:50] == test_result[:50]

    kyo=[]
    if len(np.where(above_threshold==False)[0]) != 0:
        for i in np.where(above_threshold==False)[0]:
            sub_kyo=[]
            if len(kyo) == 0 :
                sub_kyo.append(0)
                sub_kyo.append(i)
                kyo.append(sub_kyo)
            else :
                sub_kyo.append(kyo[len(kyo)-1][1]+1)
                sub_kyo.append(i)
                kyo.append(sub_kyo)

        if np.where(above_threshold==False)[0].max() != len(np.array(y_test)[:50]) and len(kyo) != 0:
            sub_kyo=[]
            sub_kyo.append(kyo[len(kyo)-1][1]+1)
            sub_kyo.append(len(np.array(y_test)[:50]))
            kyo.append(sub_kyo)

        for i in range(len(kyo)):
            plt.plot(np.array(range(kyo[i][0],kyo[i][1])),test_result[kyo[i][0]:kyo[i][1]], 
                     linestyle='none', marker='o', color='purple', lw=2.9) 

    else:
        plt.plot(test_result[:50], linestyle='none', marker='o', color='purple', lw=2.9) 

    plt.ylabel('유방암',color='white', fontweight='bold')
    plt.xlabel('데이터 넘버',color='white', fontweight='bold')
    plt.xticks(list(range(50)),fontweight='bold',color='white')
    plt.yticks([0,1],fontweight='bold', color='white', labels=['양성','악성'])
    # plt.tick_params(axis='both', labelcolor='white')
    plt.title('[ '+j+' ] 실제값과 예측값 일부 비교',color='white',size=20 ,fontweight='bold',pad=12)
    plt.legend(['실제값','예측값','일치값'],bbox_to_anchor=(1.0, 1.3), handlelength=4, edgecolor='black',
               facecolor='#F5F5F5',framealpha=True)    
    plt.show()
    
    
    su = [[j,accuracy,precision,sensitivity,specificity,kappa,f1_score]]
    
    df= pd.DataFrame(su, columns=['알고리즘','정확도','정밀도','민감도','특이도','카파통계량','f1_score'])
    globals()[f'df_{c}']=df # 동적 전역변수 선언
    
df = pd.concat([ globals()[f'df_{j}'] for j in range(len(algorithm_n)) ], ignore_index=True) #동적 전역변수를 사용해 데이터 프레임을 합침

#데이터 프레임 css 설정
styles = [dict(selector="caption", props=[("text-align", "center"),("color", 'gold'),("font-weight", 'bold'),("background-color", "gray")])]
#데이터 프레임 css적용 출력
display(df.style.set_caption('모델별 평가지표').set_table_styles(styles).hide_index())

#알고리즘별 평가지표 막대그래프
df.plot(kind='bar',x='알고리즘', rot=0,figsize=(17,6))
plt.ylabel('수치')
plt.legend(bbox_to_anchor = (1.0, 1.3))
plt.title('알고리즘별 평가지표')
plt.show()


#알고리즘 별 모델 정확도 막대 그래프
plt.figure(figsize=(17,6),facecolor='#ffd700')
bar = plt.bar(m_accuracy.keys(), m_accuracy.values())

#막대 그래프에 숫자 표기
for val in bar:
    if val.get_height() == 1 : 
        plt.text(val.get_x() + val.get_width()/2.0, val.get_height(), f'{val.get_height():.6f}', ha='center', va='bottom', 
                 size = 12, color='r', fontweight='bold')
    else:
        plt.text(val.get_x() + val.get_width()/2.0, val.get_height(), f'{val.get_height():.6f}', ha='center', va='bottom', size = 12)

plt.ylim(0,1.1)
plt.xticks(fontweight='bold')
plt.title("알고리즘별 정확도",size=20, fontweight='bold',pad=10)
plt.show()
