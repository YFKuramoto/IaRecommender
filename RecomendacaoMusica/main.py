import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt

#caminho do arquivo
dataset_path = "C:/Users/Yuri/Documents/VSCode/dataset.csv" #substituir pelo seu caminho

# Carregar o dataset
music_data = pd.read_csv(dataset_path)

# Usar apenas 10.000 linhas para reduzir o tamanho da matriz
music_data_sample = music_data.sample(n=20000, random_state=42)

# Criar uma coluna combinando informações de gênero e artista
music_data_sample['combined_features'] = (
    music_data_sample['track_genre'].fillna('') + " " +  
    music_data_sample['artists'].fillna('')  
)
# Verificar a criação da coluna 'combined_features'
#print(music_data_sample[['track_genre', 'artists', 'combined_features']].head())

# Vetorizar usando TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(music_data_sample['combined_features'])

# Atribuir peso maior para os atributos textuais
alpha = 10
tfidf_matrix = tfidf_matrix * alpha

# Selecionar colunas numéricas
numerical_features = ['danceability', 'energy', 'valence', 'tempo', 'popularity']

#Preencher valores ausentes com 0
music_data_sample[numerical_features] = music_data_sample[numerical_features].fillna(0)

# Normalizar os valores entre 0 e 1
scaler = MinMaxScaler()
numerical_data = scaler.fit_transform(music_data_sample[numerical_features])

# Combine matriz TF-IDF e atributos numéricos
sparse_feature_matrix = hstack([tfidf_matrix, csr_matrix(numerical_data)])

# Reduzir a dimensionalidade com PCA
pca = PCA(n_components=100)  # Reduz para 100 dimensões
reduced_features = pca.fit_transform(sparse_feature_matrix.toarray())

# Calcular similaridade
cosine_sim = cosine_similarity(reduced_features)


def recommend_songs(song_name, music_data, cosine_sim):
    # Obter o índice da música
    try:
        idx = music_data[music_data['track_name'] == song_name].index[0]
    except IndexError:
        return f"A música '{song_name}' não foi encontrada no dataset."

    # Similaridades para a música selecionada
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar por similaridade, excluindo a música selecionada
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

    # Obter os índices das músicas mais similares
    song_indices = [i[0] for i in sim_scores[:10]]

    # Retornar os nomes das músicas recomendadas
    return music_data.iloc[song_indices][['track_name', 'artists']]

# Testar a função
song_name = "Californication"
recs = recommend_songs(song_name, music_data, cosine_sim)

if isinstance(recs, pd.DataFrame):
    rec_labels = recs['track_name'] + " - " + recs['artists']

    # Plotar o gráfico
    plt.barh(rec_labels, range(10), color='black')
    plt.xlabel("Relevância")
    plt.ylabel("Músicas e Artistas")
    plt.title(f"Recomendações para '{song_name}'")
    plt.show()
else:
    print(recs)
