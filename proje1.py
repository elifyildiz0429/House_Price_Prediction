import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#veri yükleme
df=pd.read_csv("AmesHousing.csv")

#sayısal sütunlardaki boşlukların bulunup onların diğerlerinin ortalaması ile doldurulması, kategorik sütunlara ise None yazması
numeric_cols= df.select_dtypes(include=[np.number]).columns
df[numeric_cols]= df[numeric_cols].fillna(df[numeric_cols].mean())

categorical_cols= df.select_dtypes(include=['object','string']).columns
df[categorical_cols]= df[categorical_cols].fillna('None')

df= pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#X=Tahmin için kullanılacak özellik SalePrice olması için Id'yi devre dışı bıraktık.
#y=fiyat üzerinden tahmin üretecek
X = df.drop(columns=['SalePrice','Id'], errors='ignore') 
y=df['SalePrice']
print("X içindeki sütunlar:", X.columns) 
print("X veri sayısı:", X.shape)

#Veriyi ayırıp onu tablo için ölçeklendiriyor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20,  random_state=42 )

scaler=StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Modelleme ve tahmin etme
l_model= LinearRegression()
l_model.fit(X_train_scaled, y_train)
y_pred= l_model.predict(X_test_scaled)

#metrikler
print(f"Başarı Durumu(R2):{r2_score(y_test, y_pred):.4f}" )
print(f"Ortalama Hata(MAE):{mean_absolute_error(y_test,y_pred):.2f}")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE (Karekök Hata): {rmse:.2f}")

#Görselleştirme

#Figure1 Fiyat dağılımı
plt.figure(figsize=(6,4))
sns.histplot(y, color='yellow')
plt.title("Genel Fiyat Durumu")
plt.show

#Figure2 gerçek ve tahmin karşılaştırılması
plt.figure()
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot()
plt.xlabel("Gerçek Fiyatlar:")
plt.ylabel("Tahmin Edilen Fiyatlar:")
plt.title("Ev gerçek ve tahmin fiyatları")
plt.show()

#figure3 hata dağılımı
plt.figure()
residuals=y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals, color='black')
plt.title("hata dağilimi")
plt.show()
