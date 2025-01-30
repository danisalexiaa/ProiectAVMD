# app.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer, StandardScaler
from scipy.stats import kurtosis, skew

# Regresia liniara
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# 

# Regresie logistica 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# XGB
import xgboost as xgb

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Incarcare set de date
@st.cache
def load_data():
    return pd.read_csv('risk.csv')

df = load_data()
st.title('Preprocesarea, transformarea si analiza datelor referitoare la clienti si credite bancare')

# Afisare set de date
if st.checkbox('Afisare set de date'):
    st.write(df)

# Descrierea statistica a setului de date
# st.subheader('Vizualizare set de date')
# st.write(df.describe(include="all"))

# Verificarea datelor lipsa
st.subheader('Date lipsa')
missing_values = df.isna().sum()
st.write(missing_values)

# Heatmap pentru valorile lipsa
st.subheader('Heatmap - valori lipsa')
st.write('Acest heatmap evidentiaza locurile din setul de date unde exista valori lipsa')
cols = df.columns
colours = ['#000099', '#ffff00']
fig, ax = plt.subplots()
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours), ax=ax)
st.pyplot(fig)

# Inlocuirea valorilor lipsa
st.subheader('Inlocuirea valorilor lipsa')
df_filled = df.copy()

# Valori numerice
st.write('Valorile numerice lipsa vor fi inlocuite cu media tuturor valorilor de pe coloana respectiva')
numeric_cols = df.select_dtypes(include=[np.number]).columns.values
for col in numeric_cols:
    df_filled[col].fillna(df_filled[col].median(), inplace=True)
print(df_filled)

# Valori non-numerice
st.write('Valorile non-numerice lipsa vor fi inlocuite string-ul _NOT SPECIFIED_')
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.values
for col in non_numeric_cols:
    df_filled[col].fillna('_NOT SPECIFIED_', inplace=True)

# Afisare date dupa inlocuire valori lipsa
st.write(df_filled.isna().sum())

# st.subheader('Heatmap - valori lipsa (Dupa inlocuire)')
# cols = df_filled.columns
# fig, ax = plt.subplots()
# sns.heatmap(df_filled[cols].isnull(), cmap=sns.color_palette(colours), ax=ax)
# st.pyplot(fig)

#############################################################
#merge
# df1 = df[['id', 'Job']]
# print(df1)

#limita_durata = int(input("Introduceti limita pentru durata in luni a creditului: "))
# limita_durata = 12
# df2 = df[df['Duration in months'] < limita_durata]
# #print(df2)
# df3 = pd.merge(df1, df2, left_on=df1['id'].astype(int), right_on=df2['id'])
# df3['Purpose of the credit'].value_counts().plot.bar()


# plt.show()

# Hist1
st.subheader('Histograma -  Durata creditului in luni')
st.write("""Pentru preprocesarea datelor, se va folosi Label Encoder, ce are ca scop transformarea valorilor non-numerice in valori numerice. 
Astfel, dataframe-ul folosit va contine doar valori numerice. 
La final, se va realiza un grafic ce afiseaza frecventa de aparitie a fiecarei durate de credit masurata in luni.
Putem spune ca acele credite ce au durate mai lungi de timp au o frecventa de aparitie redusa, in comparatie
cu creditele ce au intre 10 si 15 luni.""")

fig, ax = plt.subplots()
df['Duration in months'].plot(kind='hist', ax=ax)
st.pyplot(fig)

df_transform = df[['Age in years', 'Duration in months', 'personal_status', 'Credit amount', 'Number of existing credits at this bank']]

# Label Encoding pentru datele non-numerice - acestea vor fi transformate in valori numerice
# st.subheader('Label Encoder')
label_encoder = LabelEncoder()
df_transform['personal_status'] = label_encoder.fit_transform(df_transform['personal_status'].astype(str))

# MinMaxScaler
st.subheader('Scalare date cu MinMax')
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_transform)
df_scaled = pd.DataFrame(df_scaled, columns=df_transform.columns)

st.write(df_scaled)
st.write("""Se doreste scalarea datelor pentru a le normaliza într-un interval [0, 1], ceea ce poate fi util pentru modelele de învățare automata, 
care beneficiază de date normalizate, ce au aceeasi scala. Distributia datelor scalate se poate urmari in graficul afisat.
Putem spune ca valorile extreme se afla in partea din dreapta a graficului, reprezentand valorile extreme superioare.
Se observa ca majoritatea creditelor au un numar scazut de luni ca durata.""")
# Plot 'Duration in months'
st.subheader('Histograma dupa scalare')
fig, ax = plt.subplots()
df_scaled['Duration in months'].plot(kind='hist', ax=ax)
st.pyplot(fig)

st.write("""Se doreste vizualizarea valorilor extreme in ceea ce priveste numarul de luni pe care s-a deschis un credit. 
Pentru acest lucru, am folosit un boxplot, unde se pot observa valorile extreme in partea de sus a graficului. 
Observam ca mediana este aproape de 20 (luni), iar mustatile graficului indica valorile ce nu fac parte din medie, insa sunt considerate normale. 
Putem spune ca datele sunt simetrice (mediana se afla in zona de mijloc a box-ului), insa exista outliers.""")

df.boxplot(column=['Duration in months'])

# Boxplot 'Duration in months'
st.subheader('Boxplot Duration in Months')
fig, ax = plt.subplots()
df.boxplot(column=['Duration in months'], ax=ax)
st.pyplot(fig)

# Skewness and Kurtosis
# st.subheader('Skewness si Kurtosis pentru variabila "Credit amount"')
# st.write(f"Skewness: {skew(df['Credit amount'])}")
# st.write(f"Kurtosis: {kurtosis(df['Credit amount'])}")

print(df.select_dtypes(include=[np.number]).kurtosis())

st.subheader("""Mai departe, am ales generarea unei scatter plot pentru analiza relatiei dintre variabilele Duration in months si Credit amount. 
Acest grafic ar putea raspunde la intrebarile: Persoanele ce isi iau credite pe o durata mai lunga tind sa isi ia un credit cu o valoare mai mare?/Exista credite foarte mari dar pe durate scurte?""")

# Scatter plots - corelatie
st.subheader('Scatter Plot: Duration vs Credit Amount')
fig, ax = plt.subplots()
df_transform.plot(x='Duration in months', y='Credit amount', kind='scatter', ax=ax)
st.pyplot(fig)

st.write(""" Se observa ca majoritatea creditelor au valori mici si nu au o durata foarte lunga, fiind concentrate in zona 10 luni - 2000 u.m.
Exista insa, si o valoare extrema, ce ne ajuta sa identificam un credit cu o valoare foarte mare, dar care are o durata de mai putin de 10 luni.""")

st.subheader('Scatter Plot: Number of Credits vs Age')
st.write("""Urmatorul scatter plot foloseste variabilele ce indica numarul de credite existente si varsta, pentru a identifica o posibila relatie intre aceste doua variabile.
Majoritatea persoanelor au un singur credit, indiferent de varsta. 
Putem spune ca persoanele in varsta detin mai putine credite fata de persoanele mai tinere.""")
fig, ax = plt.subplots()
df_transform.plot(x='Number of existing credits at this bank', y='Age in years', kind='scatter', ax=ax)
st.pyplot(fig)

# Preprocesare date pentru regresie liniară
st.subheader('Regresie Liniara: Age in years vs Credit amount')
if 'Duration in months' in df.columns and 'Credit amount' in df.columns:

    st.write("""Selectarea variabilei independente și variabilei dependente""")

    X = df[['Duration in months']]
    y = df['Credit amount']

    st.write("""Verificare și completare valori lipsa""")
    
    if X.isna().sum().sum() > 0 or y.isna().sum() > 0:
        st.write("Valorile lipsa sunt completate cu mediana")
        X.fillna(X.median(), inplace=True)
        y.fillna(y.median(), inplace=True)

    st.write("""Impartirea setului de date in seturi de antrenare și testare""")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write("""Construirea modelului de regresie liniara""")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    st.write("""Predictii""")
    y_pred = model.predict(X_test)
    
    st.write("""Evaluare model cu ajutorul MSE (Mean Standard Error) si R patrat""")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Eroare medie pătratica (MSE): {mse:.2f}")
    st.write("""Valoarea de 6830196.06 reprezinta eroarea medie patratica a modelului. 
    Este o masura a performantei, aratand cât de mult difera predictiile modelului de valorile reale. 
    Valorile prezise de model sunt destul de diferite fata de cele reale, pe baza acestui coeficient.""")
    st.write(f"Coeficientul de determinare (R²): {r2:.2f}")
    st.write("""R2 = 0.36 indica faptul ca modelul explica 36% din variatia valorilor pentru variabila dependentă (Credit amount) folosind variabila preditor (Duration in months). 
    Este un indicator al puterii explicative a modelului.""")
    
    # Afișarea coeficienților
    st.write(f"Coeficientul pentru 'Duration in months': {model.coef_[0]:.2f}")
    st.write("""Coeficientul 134.97 arata ca, pentru fiecare luna suplimentara de durata a creditului, valoarea creditului creste în medie, 
    cu 134.97 unitati monetare""")
    st.write(f"Interceptul modelului: {model.intercept_:.2f}")
    st.write("""Interceptul este valoarea prezisa a Credit amount atunci când Duration in months este 0. 
    Desi un credit cu durata de 0 luni nu exista, aceasta valoare servește ca punct de referinta pentru linia regresiei.""")
    
    # Vizualizarea predicțiilor
    st.subheader('Grafic: Valori Reale vs Predicții')
    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, label='Valori reale', color='blue', alpha=0.6)
    ax.scatter(X_test, y_pred, label='Predicții', color='red', alpha=0.6)
    ax.plot(X_test, model.predict(X_test), color='green', label='Regresie liniară')
    ax.set_xlabel('Duration in months')
    ax.set_ylabel('Credit amount')
    ax.legend()
    st.pyplot(fig)
else:
    st.write("Variabilele 'Duration in months' si 'Credit amount' nu sunt prezente in setul de date")


st.write("""Regresia logistica""")
st.write("""Creare variabila noua 'Risk', pe baza altor variabile asociate""")

df['Risk'] = 'Scăzut'  # Inițializăm toți clienții cu risc scăzut

# Condiții pentru a actualiza valoarea riscului
df.loc[(df['Credit history'] == 'critical/other existing credit') & (df['Credit amount'] > 5000), 'Risk'] = 'Ridicat'
df.loc[(df['Installment rate in percentage of disposable income'] > 2), 'Risk'] = 'Mediu'
df.loc[(df['Credit history'] == 'existing paid'), 'Risk'] = 'Scazut'

X = df[['Duration in months', 'Credit history', 'Credit amount', 
        'Installment rate in percentage of disposable income', 
        'Age in years', 'Number of existing credits at this bank']]
y = df['Risk']

X['Credit history'] = label_encoder.fit_transform(X['Credit history'])
y = label_encoder.fit_transform(y)

st.write("Am impartit datele inainte de preprocesare si am verificat sa nu existe valori lipsa pentru variabile.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit_transform doar pe antrenament
X_test_scaled = scaler.transform(X_test)  # transform pe setul de test, nu fit

st.write("Creare si antrenare model")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

st.write("Am folosit predict pentru predictia pe setul meu de date")
y_pred = model.predict(X_test_scaled)

st.write("Am evaluat modelul cu ajutorul acuratetii si a matricei de confuzie")
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

st.write("Rezultate:")
st.write(f"Precizia modelului: {accuracy * 100:.2f}%")
st.write("Precizia modelului este de 88%, ceea ce reprezinta o valoare buna pentru model. Acest lucru inseamna ca moelul a prezis corect 88% dintre valori")
st.write("Matricea de confuzie:")
st.write(conf_matrix)
st.write("""Pentru matricea de confuzie, se observa ca majoritatea valorilor au fost prezise corect. Ca exemplu:
Pe prima linie, corespondenta nivelului 'Scazut', 26 de predictii au fost corecte pentru riscul scazut, una gresita, etichetata ca 'Mediu', 3 gresite, etichetate ca 'Ridicat', si 0 gresite pentru categoria 'Necunoscut'""")


# Crearea și antrenarea modelului XGBoost
st.subheader('Antrenare Model XGBoost')

model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model_xgb.fit(X_train_scaled, y_train)

# Predicții pe setul de test
y_pred_xgb = model_xgb.predict(X_test_scaled)

# Evaluare model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)

# Afișarea rezultatului
st.write(f"Precizia modelului XGBoost: {accuracy_xgb * 100:.2f}%")
st.write("Matricea de confuzie model XGBoost:")
st.write(conf_matrix_xgb)

# Vizualizarea importanței caracteristicilor
st.subheader('Importanta caracteristicilor')
xgb.plot_importance(model_xgb)
st.pyplot()
