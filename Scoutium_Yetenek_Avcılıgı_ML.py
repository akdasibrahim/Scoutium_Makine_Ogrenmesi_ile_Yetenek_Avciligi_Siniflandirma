import pandas as pd
import numpy as np
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

# Adım1:  scoutium_attributeasds.csv ve scoutium_potential_labelssdasd.csv dosyalarını okutunuz

attributes = pd.read_csv('datasets/scoutium_attributes.csv', sep= ';')
potential = pd.read_csv('datasets/scoutium_potential_labels.csv', sep= ';')

#  Adım2:  Okutmuşolduğumuzcsv dosyalarınımerge fonksiyonunu kullanarak birleştiriniz.
#  ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)

df= pd.merge(attributes, potential, how='left', on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])
df.shape
df.head()
df.info()
# Adım3:  position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.

df[df['position_id'] != 1].head()
df[df['position_id'] != 1].shape
df[df['position_id'] == 1].shape

df = df[df['position_id'] != 1]

#  Adım4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)

df[df["potential_label"] == "below_average"].shape[0]
df = df[df["potential_label"] != "below_average"]


#  Adım5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu
#  olacak şekilde manipülasyon yapınız.
#     Adım1: İndekste “player_id”,“position_id” ve “potential_label”,  sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
#     “attribute_value” olacak şekilde pivot table’ı oluşturunuz.####

pivot_tab= pd.pivot_table(df, values="attribute_value", columns="attribute_id", index=["player_id","position_id","potential_label"])
pivot_tab.head()

# Adım2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.

pivot_tab = pivot_tab.reset_index(drop=False)
pivot_tab.head()
pivot_tab.info()
pivot_tab.columns = pivot_tab.columns.map(str)

#  Adım6:  Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.

le = LabelEncoder()
pivot_tab['potential_label'] = le.fit_transform(pivot_tab['potential_label'])


#  Adım7:  Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız

def grab_col_names(dataframe, cat_th=8, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(pivot_tab)

num_cols = [col for col in num_cols if col not in ['position_id','player_id'] ]

# Adım8:  Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.

scaler = StandardScaler()
pivot_tab[num_cols] = scaler.fit_transform(pivot_tab[num_cols])

# Görev 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir
# makine öğrenmesi modeli geliştiriniz.

y = pivot_tab["potential_label"]
X = pivot_tab.drop(["potential_label", "player_id"], axis=1)

models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   #("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   #('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('CatBoost', CatBoostClassifier(verbose=False)),
                   ("LightGBM", LGBMClassifier(verbose=-1)),]


scores = ["roc_auc", "f1", "precision", "recall", "accuracy"]
rows = []

for name, model in models:
    row = {"model": name}
    for score in scores:
        row[score] = cross_val_score(model, X, y, scoring=score, cv=10).mean()
    rows.append(row)

models_score = pd.DataFrame(rows).set_index("model")
print(models_score)

# Hiperparametre Optimizasyonu

lgbm_model = LGBMClassifier(random_state=46)
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]
             }


lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_, verbose=-1).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

print("LightGBM")
for score in ["roc_auc_ovr", "f1_macro", "precision_macro", "recall_macro", "accuracy"]:
    cvs = cross_val_score(final_model, X, y, scoring=score, cv=10).mean()
    print(score + " score:" + str(cvs))



################################################################
# Görev 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
################################################################

# feature importance
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMClassifier()
model.fit(X, y)

plot_importance(model, X)












































