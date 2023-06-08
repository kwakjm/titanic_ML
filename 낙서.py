from sklearn.model_selection import train_test_split

# 나이가 결측치인 행만 선택
df_age_missing = df[df['Age'].isnull()]

# 나이가 있는 행만 선택
df_age_available = df[df['Age'].notnull()]

# 훈련 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    df_age_available.drop('Age', axis=1),
    df_age_available['Age'],
    test_size=0.2,
    random_state=42
)

# 회귀 모델 학습
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 결측치 예측
age_predictions = regressor.predict(X_test)

# 예측된 결측치로 대체
df_age_missing['Age'] = age_predictions

# 원래 DataFrame에 결측치 대체 결과 반영
df_filled = pd.concat([df_age_available, df_age_missing])
df_filled.sort_index(inplace=True)