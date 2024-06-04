

##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
df= sns.load_dataset("titanic")
df.head()
#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

df["sex"].value_counts()

#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################

for col in df.columns:
    print(col, df[col].nunique())

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

df["pclass"].unique()
df["pclass"].nunique()

#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################
selected = ["pclass", "parch"]
for col in selected:
    print(col, df[col].nunique())

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

print("öncesi: ", df["embarked"].dtype)
df["embarked"] = df["embarked"].astype("category")
print("sonrası: ", df["embarked"].dtype)

#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

df["embarked"].value_counts()
df[df["embarked"] == "C"].head(15)
df[df["embarked"] == "C"].size
#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################

df[df["embarked"] != "S"].head(15)

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

df.loc[(df["age"] < 30) & (df["sex"] == "female"), :]

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

df.loc[(df["fare"] > 500) | (df["age"] > 70), :].head(15)

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

df.isnull().values.any()  # eksik veri olup olmadığını verir.
df.isnull().values.sum()  # toplam eksik veriyi verir.
df.isnull().sum()         # değişkenler bazında eksik veriyi verir.

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

new_df = df.drop("who", axis=1)
new_df.columns

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################

most_freauent = df["deck"].mode()[0]
df["deck"].fillna(most_freauent, inplace=True)

#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################
df["age"].isnull().values.any()

medyan_value_for_age = df["age"].median()
df["age"].fillna(medyan_value_for_age, inplace=True)

df["age"].head(20)
#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "mean", "count"]})
###
b = df.pivot_table("survived","pclass", "sex")

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################

df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)

#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################

df = sns.load_dataset("tips")

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.groupby("time").agg({"total_bill": ["sum", "min", "max", "mean"]})


#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.groupby(["day", "time"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

new_selected = df[(df["sex"] == "Female") & (df["time"] == "Lunch")]

new_selected.groupby("day").agg({"total_bill": ["sum", "min", "max", "mean"],
                                 "tip": ["sum", "min", "max", "mean"]})

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

df.loc[(df["size"] < 3) & (df["total_bill"] > 10), "total_bill"].mean()

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

df["total_bill_tip_sum"] =  df["total_bill"] + df["tip"]
df["total_bill_tip_sum"].head()

#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

total_bill_tip_sum = df["total_bill_tip_sum"].sort_values(ascending=False)  #ascending=False kullanmamızın nedeni kücükten büyüğe olmasını istememiz
lst = total_bill_tip_sum.index.values                                       # eğer ki yazmasaydık True olarak kalırdı ve büyükten kücüğe sıralanırdı
top_df = df.iloc[lst[:30]]
