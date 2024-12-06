import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt

# 1. Carregar o dataset
dataset_path = "C:/Users/Yuri/Documents/VSCode/RecomendacaoMusica/dataset.csv"
music_data = pd.read_csv(dataset_path)

# 2. Pré-processamento
music_data_sample = music_data.sample(n=30000, random_state=42)
music_data_sample.fillna({'track_genre': 'unknown', 'artists': 'unknown'}, inplace=True)
music_data_sample['combined_features'] = (
    music_data_sample['track_genre'] + " " + music_data_sample['artists']
)

# 3. TF-IDF e Pesos
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(music_data_sample['combined_features'])
alpha = 1  # Peso para os atributos textuais
tfidf_matrix = tfidf_matrix * alpha

# 4. Normalizar atributos numéricos
numerical_features = ['danceability', 'energy', 'valence', 'tempo', 'popularity']
scaler = MinMaxScaler()
numerical_data = scaler.fit_transform(music_data_sample[numerical_features].fillna(0))
beta = 1  # Peso para os atributos numéricos
numerical_data = numerical_data * beta

# 5. Combinar e Reduzir Dimensionalidade
sparse_feature_matrix = hstack([tfidf_matrix, csr_matrix(numerical_data)])
pca = PCA(n_components=100)
reduced_features = pca.fit_transform(sparse_feature_matrix.toarray())

# 6. Similaridade
cosine_sim = cosine_similarity(reduced_features)

# 7. Recomendação
def recommend_songs(song_name, music_data, cosine_sim):
    try:
        # Procura o índice da música na amostra, não no dataset completo
        idx = music_data[music_data['track_name'] == song_name].index[0]
    except IndexError:
        return pd.DataFrame(), []  # Retorna valores padrão se a música não for encontrada

    # Garante que o índice seja relativo à amostra
    idx = music_data.index.get_loc(idx)

    # Calcula similaridade e ordena
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Top 5 mais similares
    indices = [i[0] for i in sim_scores]

    # Recupera as músicas recomendadas
    recommendations = music_data.iloc[indices][['track_name', 'artists', 'track_genre']]
    return recommendations, sim_scores

# 8. Avaliar Métricas de Qualidade
def evaluate_recommendations(song_name, recommendations, sim_scores):
    avg_similarity = sum([score[1] for score in sim_scores]) / len(sim_scores)  # Média das similaridades
    genres = recommendations['track_name'].count()
    diversity = len(recommendations['track_genre'].unique())  # Diversidade dos gêneros
    return avg_similarity, diversity

# 9. Visualização das Recomendações
def plot_recommendations(song_name, recommendations):
    if recommendations.empty:
        print(f"Não há recomendações para a música '{song_name}'!")
        return
    rec_labels = recommendations['track_name'] + " - " + recommendations['artists']
    plt.barh(rec_labels, range(len(rec_labels)), color='skyblue')
    plt.xlabel("Similaridade")
    plt.ylabel("Músicas e Artistas")
    plt.title(f"Recomendações para '{song_name}'")
    plt.gca().invert_yaxis()
    plt.show()

# 10. Testar o Sistema
song_name = "Hallowed Be Thy Name"  # Substitua pelo nome de uma música válida no dataset
if song_name in music_data_sample['track_name'].values:
    recommendations, sim_scores = recommend_songs(song_name, music_data_sample, cosine_sim)

    if not recommendations.empty and sim_scores:
        # Calcular métricas de qualidade
        avg_similarity, diversity = evaluate_recommendations(song_name, recommendations, sim_scores)
        print(f"Métricas de Qualidade para '{song_name}':")
        print(f"- Similaridade Média: {avg_similarity:.4f}")
        print(f"- Diversidade de Gêneros: {diversity}")

        # Plotar recomendações
        plot_recommendations(song_name, recommendations)
    else:
        print(f"Nenhuma recomendação válida encontrada para '{song_name}'.")
else:
    print(f"A música '{song_name}' não foi encontrada na amostra.")

