###### Customer Lifetime Value Prediction #####

### CLTV = (Costumer Value / Churn rate ) * Profit Margin

### Costumer Value = Purchase Frequency * Average order Value

#### CLTV Expected Number of Transaction * Expected Average profit

#### CLTV = BG / NBD Model * Gamma gamma Submodel

### BG / NBD ( Beta Geometric / Negative Binomial distribution via expected Number of Transaction )

#### Transaction Rate ( Gamma dağılır )

### Dropout Rate ( Beta dağılır )

#### BG- NBD ve Gamma gamma via CLTV prediction

#####1.  Data Preperation

#### 2 BG-NBD Modeli ile Expected Number of Transaction

#### 3. Gamma - Gamma Modeli ile Expected Average Profit.

#### 4. BG- NBD ve gamma -gamma modeli ile CLTV 'nin hesaplanması

#### 5. CLTV 'ye göre Segmentlerin oluşturulması

#### 6. Çalışmanın Fonksiyonlaştırılması..

import datetime as dt
from idlelib.replace import replace

import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes .plotting import plot_period_transactions
from pyparsing import replace_with

from RMF.rfm import today_date

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

from sklearn.preprocessing import MinMaxScaler


#### Reading Data

df_ = pd.read_excel("/Users/erdinc/PycharmProjects/pythonProject4/RMF/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()

df.describe().T
df.head()
df.isnull().sum()
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0 ]

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)


##### Lifetime Veri Yapısının Hazırlanması

#### recency : Son satın alma üzerinden geçen zaman. Haftalık. ( kullanıcı özelinde)

### T: Müşterinin yaşı . Haftalık. ( Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)

### Frequency : Tekrar eden toplam satın alma sayısı ( frequency > 1 )

### Monetary_value : satın alma başına ortalama kazanç

cltv_df = df.groupby("Customer ID").agg({
    "InvoiceDate": [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                    lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
    "Invoice": lambda num: num.nunique(),
    "TotalPrice": lambda TotalPrice: TotalPrice.sum()
})

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ["recency", "T", "frequency", "monetary"]

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df.describe().T

cltv_df = cltv_df[(cltv_df["frequency"] > 1 )]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7


#### 2 BG- NBD Modelin kurulması

bgf = BetaGeoFitter(penalizer_coef=0.001)
### parametre tahmin yönteminde kullanacağız daha çok

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

#### Modelimizi kurduk şimdiiii

### 1 hafta içersinde en çok satın alma beklediğimiz 10 müşteri kimdir...

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                               cltv_df["T"]).sort_values(ascending=False).head(10)

bgf.predict(1,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sort_values(ascending=False).head(10)

### aynısı sadece farklı bir kod ile yazıldı.... Ama bunu Gamma gamma da yapamayız.

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df["frequency"],
                                                  cltv_df["recency"],
                                                  cltv_df["T"])

### 1 ay içinde dersek eğer

bgf.predict(4,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sort_values(ascending=False).head(10)




cltv_df["expected_purc_1_month"] = bgf.predict(4,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sort_values(ascending=False).head(10)



### 1 aylık için toplam değerleri istersek eğer...

bgf.predict(4,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sum()

#### yapılan sonuçların tahmin başarısı

plot_period_transactions(bgf)
plt.show()


##### Gamma Gamma Modelin kurulması...


ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).head(10)

ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).sort_values(ascending=False).head(10)


cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                             cltv_df["monetary"])

cltv_df.sort_values("expected_average_profit", ascending=False).head(10)



