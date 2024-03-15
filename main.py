import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

dane = pd.read_csv("dane_gier_steam.csv", header=[0], on_bad_lines='skip', delimiter=';')

melt_df = pd.melt(dane, id_vars=['Title', 'Reviews Total', 'Reviews Score Fancy', 'Release Date', 'Launch Price', 'Revenue Estimated'], value_vars=['Tags'], value_name='Tag')
melt_df['Tag'] = melt_df['Tag'].str.split(', ')

exploded_df = melt_df.explode('Tag').drop(columns=['variable']).reset_index(drop=True)
exploded_df['Revenue Estimated'] = exploded_df['Revenue Estimated'].str.replace(' ', '').str.strip('$').str.replace(',', '.').astype(float)
exploded_df['Launch Price'] = exploded_df['Launch Price'].str.replace(' ', '').str.strip('$').str.replace(',', '.').astype(float)
exploded_df['Reviews Score Fancy']=exploded_df['Reviews Score Fancy'].str.strip('%').str.replace(',', '.').astype(float)

label_encoder = LabelEncoder()
exploded_df['Tag_encoded'] = label_encoder.fit_transform(exploded_df['Tag'])

X = exploded_df[['Launch Price', 'Tag_encoded']]
y = exploded_df['Reviews Score Fancy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Błąd średniokwadratowy (MSE) modelu:".format(mse))

new_data = pd.DataFrame({'Launch Price': [9.99],'Tag_encoded': [label_encoder.transform(['Action'])[0]]})
prediction = model.predict(new_data)
print(prediction)
