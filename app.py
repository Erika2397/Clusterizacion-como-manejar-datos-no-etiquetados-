
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Tribus digitales", layout="wide")
st.title("🧠 Descubre tu Tribu Digital")
st.subheader("Segmentación de estilo de vida con Machine Learning")

@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/Erika2397/Clusterizacion-como-manejar-datos-no-etiquetados-/refs/heads/main/datos_mkt_traducido.csv"
    return pd.read_csv(url)

df = cargar_datos()

st.write("Vista previa del dataset:")
st.dataframe(df.head())

columnas_intereses = [
    'basquet', 'futbol_americano', 'futbol', 'softbol', 'voleibol', 'natacion',
    'animacion', 'beisbol', 'tenis', 'deportes', 'danza', 'banda',
    'marcha', 'musica', 'rock', 'cabello', 'vestido', 'compras', 'nuestra_marca', 
    'marca_de_la_competencia', 'bebidas'
]

k = st.slider("Número de clusters", 2, 10, 3)

df_intereses = df[columnas_intereses]
df_scaled = StandardScaler().fit_transform(df_intereses)
df_pca = PCA(n_components=2).fit_transform(df_scaled)
clusters = KMeans(n_clusters=k, random_state=42).fit_predict(df_scaled)

df['cluster'] = clusters

st.markdown("### Visualización de clusters (PCA)")
fig, ax = plt.subplots()
sns.scatterplot(x=df_pca[:,0], y=df_pca[:,1], hue=clusters, palette='Set2', ax=ax)
st.pyplot(fig)

st.markdown("### Perfil promedio por cluster")
resumen = df.groupby('cluster')[columnas_intereses].mean().T
st.dataframe(resumen.style.highlight_max(axis=1))
