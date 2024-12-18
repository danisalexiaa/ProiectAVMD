# app.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer
from scipy.stats import kurtosis, skew

import streamlit as st

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
