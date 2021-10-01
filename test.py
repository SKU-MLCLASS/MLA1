import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

csv_file = pd.read_csv("koreahouse.csv", encoding="euc-kr")


def delete_data(data):
    print(data.isnull().sum())
    del data["대지권면적"]
    del data["Unnamed: 7"]
    print(data)
    return data


def pre_processing(first_data):
    csv_file = delete_data(first_data)
    csv_save = pd.DataFrame(csv_file).to_csv("pre_seoul_aprtment.csv",
                                             index=False,
                                             index_label=False)
    return csv_save


first_data_presing = delete_data(csv_file)

seoul_index = first_data_presing["자치구명"][:10000]
X = first_data_presing["건물면적"][:10000]
y = first_data_presing["물건금액"][:10000]
plt.rc('font', family='Malgun Gothic')
plt.plot(X, seoul_index, "..b", "..r")
plt.xlabel("자치구")
plt.ylabel("면적")
plt.show()

plt.bar(seoul_index, y)
plt.xlabel("자치구")
plt.ylabel("가격")
plt.show()

# X : 건물면적, Y: 가격
plt.plot(y, X, "..b")
plt.xlabel("가격별 면적")
plt.show()

plt.plot(X, y, "..b", label="면적 대비 가격 ")
plt.xlabel("면적 대비 가격")
plt.show()
"""
선형 회귀(linear regression)이란게 결국 
y 라는 종속변수를 설명할려고하는거잖아 그러면 결론적으로 독립변수 x값을 뭘로 하냐에 따라서 
y값이 심하게 요동칠 수 있다는거 데이터분석으로 해서 특징추출이나 데이터 해석을 할때 막하냐 ?
그건 아니라는거지 어떻게 해야하냐? 
1. 모델에 맞춰서 하는경우 
    1.1 모델에 맞춰서 하는경우는 x값 하고 y값을 쓸려고하는 당위성 확보하기 위해서 
        ex) 주택 면적에 비례해서 가격을 예측하고 싶을때 특징이 이렇게 떄문에 하겟다라는 당위성 
2. 데이터를 해석할려고 
    이건 말그대로 데이터를 보고서 현재 동향과 회사의 투명성 or 미래성을 보면서 비지니스적으로 접근
        ex) 법정동에 2020년 에 면적으로 보니깐 15억이더라 그러므로 투자가치 머머머더더 
"""
