# House_Price_Prediction
# Projenin Amacı:
Bu projenin amacı, gerçek bir veri  seti üzerinden Linear Regresyon bir model geliştirerek hedef değişkeni tahmin etmek, modeli test etmektir.
# Model Performans:
Model, değişken olarak Id kullandık, SalePrice üzerinden yaparsak eğer fiyatı fiyat üzerinden tahmin etmiş olacaktı bu ise bize sonuçları tamamen doğru bulduğumuzu gösteriyor. Id isebize farklı sonuçlar verdi.
Id bize aşağıdaki sonuçları vermiştir:
** R² Skoru:-0.0098 ** (Id ve fiyat arasında doğrudan bir ilişki yok bu yüzden sonuç negatiftir.) 
** MAE (Ortalama Mutlak Hata):59739.89 ** 
** Kullanılan Girdi (X):Index(['Id']) **
** Veri Sayısı:(1459, 1) **
# Veri Seti
Projede kullanılan veri seti[House_Price_Prediction] (https://www.kaggle.com/datasets/carlmcbrideellis/house-prices-advanced-regression-solution-file)
# Teknik Yapı
Bu projeyi yapmak için bazı kütüphaneleri kullanırız:
** Pandas : Veri manipülasyonu
** Numpy : Veri manipülasyonu
** scikit-learn : Model Eğitimi
** matplotlib : Veri Görselleştirme
** seaborn : Veri Görselleştirme
# Kurulum
pip install pandas numpy scikit-learn matplotlib seaborn
