import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 데이터 파일 경로 지정
file_path = "./data/concrete_data.csv"

# 데이터 불러오기
df = pd.read_csv(file_path)

# 전처리 결측값을 평균값으로 대체
df.fillna(df.mean(), inplace=True)

# 이상치 처리: IQR 기반
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# 모든 열에 대해 이상치 처리 적용
for column in df.columns:
    df = remove_outliers(df, column)

# 선택한 열들을 두 그룹으로 나누기
selected_columns_1 = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer']
selected_columns_2 = ['coarse_aggregate', 'age', 'concrete_compressive_strength', 'fine_aggregate ']

# 첫 번째 그룹의 산점도 매트릭스
scatter_1 = sns.pairplot(df[selected_columns_1], height=2, plot_kws={'s': 5})
scatter_1.fig.suptitle('Scatterplot Matrix - Part 1', y=1.02)
plt.show()

# 두 번째 그룹의 산점도 매트릭스
scatter_2 = sns.pairplot(df[selected_columns_2], height=2, plot_kws={'s': 5})
scatter_2.fig.suptitle('Scatterplot Matrix - Part 2', y=1.02)
plt.show()

# 열 상관관계 히트맵
plt.figure(figsize=[17, 8])
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# 독립 변수와 종속 변수 설정
x = df.drop(['concrete_compressive_strength'], axis=1)
y = df['concrete_compressive_strength']

# K-Fold Cross Validation을 위한 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 평균 제곱 오차(MSE)를 저장할 리스트
mse_scores = []

# CatBoost 모델 학습 및 평가
for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    cat = CatBoostRegressor(loss_function='RMSE', random_state=42)
    cat.fit(x_train, y_train, eval_set=(x_test, y_test), plot=False)

    y_pred = cat.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

    cat_score = cat.score(x_test, y_test)
    print(f'CatBoost Score: {cat_score}')

# K-Fold Cross Validation 결과 출력
print(f'Mean Squared Error for each fold: {mse_scores}')
print(f'Mean Squared Error (average): {np.mean(mse_scores)}')

# 예측값과 실제값 비교 시각화 (예측값: 빨강, 실제값: 파랑)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='red', alpha=0.5, label='Predicted')
plt.scatter(y_test, y_test, color='blue', alpha=0.5, label='Actual')
plt.xlabel("Concrete Compressive Strength")
plt.ylabel("Values")
plt.title("CatBoost Model - Actual vs Predicted")
plt.legend()
plt.show()
