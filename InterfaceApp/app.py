from __future__ import annotations

import json
import os
from typing import List, Dict, Any

import requests
import pandas as pd
import streamlit as st

# ==== Configuration ====
DEFAULT_ENDPOINT = (
    "https://func-rec-eme7ffg9chaydnhg.francecentral-01.azurewebsites.net/api/recommend_last_click"
)
RECOMMEND_ENDPOINT = os.getenv("RECOMMEND_FUNC_URL", DEFAULT_ENDPOINT)
TOP_N = 5
TIMEOUT_SEC = 10


@st.cache_data(show_spinner=False, ttl=60)
def call_recommendation_api(user_id: int, endpoint: str) -> List[Dict[str, Any]]:
    """Appelle l'API de recommandation Azure et renvoie TOP_N éléments {article_id, score}."""
    params = {"user_id": user_id}
    try:
        response = requests.get(endpoint, params=params, timeout=TIMEOUT_SEC)
    except requests.RequestException as exc:
        raise ValueError(f"Erreur réseau/API: {exc}") from exc

    if response.status_code != 200:
        preview = (response.text or "")[:200]
        raise ValueError(f"API {response.status_code}: {preview}")

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise ValueError("Réponse JSON invalide") from exc

    recs = payload.get("recommendations") or []

    # Normalisation minimaliste 
    df = pd.DataFrame(recs)
    if df.empty:
        return []

    # colonnes attendues
    if "article_id" not in df.columns or "score" not in df.columns:
        return []

    # typage & nettoyage
    df["article_id"] = pd.to_numeric(df["article_id"], errors="coerce").astype("Int64")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["article_id", "score"])

    # tri décroissant puis top N
    df = df.sort_values("score", ascending=False).head(TOP_N)

    # retour en list[dict]
    return df[["article_id", "score"]].to_dict(orient="records")


def main() -> None:
    """Point d'entrée de l'application Streamlit."""
    st.set_page_config(page_title="Recommandation de contenu", layout="centered")

    st.title("🔮 Recommandations d’articles")
    # st.caption("Interface de démonstration — Azure Functions • historique d’interactions")
    st.markdown( """ Cette application Streamlit utilise un service Azure Functions pour 
                fournir des recommandations d'articles à partir de votre historique d'interactions.
                Entrez votre identifiant utilisateur ci‑dessous pour découvrir de nouveaux contenus. """ 
                )
    st.subheader("Obtenir des recommandations")
    user_id = st.number_input(
        label="Identifiant utilisateur",
        min_value=0,
        value=0,
        step=1,
        help="Identifiant unique de l'utilisateur pour lequel générer des recommandations."
    )

    col_run, _ = st.columns([1, 2])
    with col_run:
        run = st.button("Obtenir les recommandations", type="primary")

    if run:
        with st.spinner("Consultation des recommandations…"):
            try:
                recs = call_recommendation_api(int(user_id), RECOMMEND_ENDPOINT)
            except ValueError as err:
                st.error(f"Impossible d'obtenir des recommandations : {err}")
                return

        if not recs:
            st.warning("Aucune recommandation disponible pour cet utilisateur.")
            return

        # Tableau lisible
        df_recs = pd.DataFrame(recs).rename(
            columns={"article_id": "ID article", "score": "Score"}
        )
        df_recs["Score"] = pd.to_numeric(df_recs["Score"], errors="coerce").round(6)
        st.dataframe(df_recs, use_container_width=True)

        st.success("Recommandations générées avec succès !")

    st.divider()
    st.markdown(
        """
        ### À propos
        Ce projet a été réalisé par *Said Arrazouaki* dans le cadre de la formation AI Engineer d’OpenClassrooms.
        """
    )


if __name__ == "__main__":
    main()
