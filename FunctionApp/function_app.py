import logging
import json
import os
import pickle
# import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient
from sklearn.metrics.pairwise import cosine_similarity
import azure.functions as func

# Créer l'application Functions (modèle v2)
app = func.FunctionApp()

# Paramètres du stockage
CONTAINER_NAME = os.getenv('BLOB_CONTAINER_NAME', 'data')
FILES = {
    'metadata': 'articles_metadata.csv',
    'embeddings': 'articles_embeddings_reduced.pickle',
    'user_clicks': 'user_clicks.pickle',
    'popular_articles': 'popular_articles.pickle'
}

# Chargement en cache des fichiers depuis Blob Storage
CACHE = {}

def load_blob_to_memory(filename):
    """Télécharge un blob et le charge en DataFrame ou objet Python."""
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=filename)
    data_bytes = blob_client.download_blob().readall()
    if filename.endswith('.csv'):
        from io import BytesIO
        return pd.read_csv(BytesIO(data_bytes))
    else:
        return pickle.loads(data_bytes)

# Chargement initial des fichiers
for key, fname in FILES.items():
    CACHE[key] = load_blob_to_memory(fname)

articles_df = CACHE['metadata']
embeddings_matrix = CACHE['embeddings']
user_clicks = CACHE['user_clicks']
popular_articles = CACHE['popular_articles']

def get_user_profile_last_click(user_id):
    """Retourne l’embedding du dernier article cliqué pour l’utilisateur."""
    clicks = user_clicks.get(int(user_id), [])
    if not clicks:
        return None
    last_idx = clicks[-1]
    # compatibilité DataFrame ou numpy array
    return embeddings_matrix.iloc[last_idx].values if hasattr(embeddings_matrix, 'iloc') else embeddings_matrix[last_idx]

def recommend_articles(user_vector, user_id, top_n=5):
    """Calcule les articles les plus similaires en excluant ceux déjà vus."""
    if user_vector is None:
        # Utilisateur inconnu : proposer les articles les plus populaires
        return [{'article_id': int(a), 'score': None} for a in popular_articles[:top_n]]
    
    sims = cosine_similarity([user_vector], embeddings_matrix.values if hasattr(embeddings_matrix, 'values') else embeddings_matrix)[0]
    seen = set(user_clicks.get(int(user_id), []))
    for idx in seen:
        sims[idx] = -1
    top_indices = sims.argsort()[::-1][:top_n]
    recs = []
    for idx in top_indices:
        art_id = int(articles_df.iloc[idx]['article_id'])
        recs.append({'article_id': art_id, 'score': float(sims[idx])})
    return recs

@app.function_name(name="recommend_last_click")
@app.route(route="recommend_last_click", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET","POST"])
def recommend_last_click(req: func.HttpRequest) -> func.HttpResponse:
    """Point d’entrée HTTP : calcule des recommandations last‑click."""
    logging.info("Requête de recommandation reçue.")
    user_id = req.params.get('user_id')
    if not user_id:
        try:
            body = req.get_json()
            user_id = body.get('user_id')
        except Exception:
            pass
    if not user_id:
        return func.HttpResponse("Paramètre user_id manquant.", status_code=400)
    vector = get_user_profile_last_click(user_id)
    recs = recommend_articles(vector, user_id)
    return func.HttpResponse(
        json.dumps({'user_id': int(user_id), 'recommendations': recs}, ensure_ascii=False),
        status_code=200,
        mimetype="application/json"
    )