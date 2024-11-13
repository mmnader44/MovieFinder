import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import seaborn as sns

# Charger le DataFrame
chemin = Path(__file__).parent
fichier_data = chemin / "df_final.csv"
df = pd.read_csv(fichier_data, sep=",", lineterminator="\n")

# Charger les composants sauvegardés
chemin = Path(__file__).parent
tfidf_matrix = load(chemin / "tfidf_matrix.joblib")
indices = load(chemin / "indices.joblib")
titles = load(chemin / "titles.joblib")

st.sidebar.markdown("# About this app:")

intro_text = """
Hi! \n
I'm NADER Mehdi-Michel, and this is my second python project for my Data analyst Bootcamp at Wild Code School.  \n
This app recommends nearby movies based on a user's choice of a movie .\n
You can chose the range of the recommandations \n
You have the choice to filter or not by "genre" to refine recommendations \n
I gathered movie data from IMDB and TMDB using pandas to clean and merge the datasets.\n
The project involves exploratory data analysis on datasets, visualized by graphs\n
If you're curious about the code and want to explore it, feel free to visit my Github account! [GitHub](https://github.com/mmnader44/MovieFinder)\n
"""
st.sidebar.markdown(intro_text)


# Interface utilisateur Streamlit
st.markdown(
    "<h1 style='color: white ; text-align: center;'>MOVIE FINDER</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h2 style='color: white ; text-align: center;'>Based on IMDB and TMDB database</h2>",
    unsafe_allow_html=True,
)

st.write("\n")
st.write("\n")

# le slider pour le choix des nearestneighbors repis plus tard dans le .fit()
x = st.select_slider("Number of films to recommend :", options=range(1, 21))
nb = x + 1

# tri colonne genre
all_genres = " ".join(df["genres"].str.replace(",", " ")).split()
choix_genre = list(set(all_genres))

# Widget multiselect pour filtrer les genres
genres_selectionnes = st.multiselect("Optionnal genre filter:", options=choix_genre)

# Vérifier si des genres ont été sélectionnés
if genres_selectionnes:
    # Obtenir les indices des films correspondant aux genres sélectionnés
    indices_genres = df[df["genres"].str.contains("|".join(genres_selectionnes))].index
    # Filtrer la matrice TF-IDF et les titres correspondants
    tfidf_matrix_filtre = tfidf_matrix[indices_genres]
    df_filtre = df.loc[indices_genres].reset_index(drop=True)
    titles_filtre = titles[indices_genres].reset_index(drop=True)
else:
    # Utiliser la matrice TF-IDF complète si aucun genre n'est sélectionné
    tfidf_matrix_filtre = tfidf_matrix
    df_filtre = df.reset_index(drop=True)
    titles_filtre = titles.reset_index(drop=True)

dico_films = {name: index for name, index in zip(titles, indices)}

# Onglets
tab1, tab2, tab3 = st.tabs(
    ["MOVIE FINDER", "BDD IMDB DATAVIZ", " BDD MOVIE FINDER DATAVIZ"]
)

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.write("Choose a film:")
        user_input = st.selectbox(
            "", titles_filtre.sort_values(ascending=True), index=None
        )

        # Vérifier si l'utilisateur a sélectionné un film
        film_selected = user_input is not None

        if st.button("Search:"):
            if not film_selected:
                st.error("Please select a film.")
            else:  # Si un film est sélectionné
                user_index = dico_films[user_input]

                # Vérifier si l'utilisateur a sélectionné des genres
                if genres_selectionnes:
                    # Recupérer user_index du DF filtré
                    user_index_filtre = df_filtre.index[
                        df_filtre["originalTitle"] == user_input
                    ][0]
                else:
                    user_index_filtre = user_index
            
                st.write("You chose:")
                choix = user_input
                st.image(
                    f"https://image.tmdb.org/t/p/original/{df.loc[user_index, 'poster_path']}"
                )
                st.write(f"Nom : **{choix}**")
                st.write(f"Genres : {df.loc[user_index, 'genres']}")
                st.write(f"Année : {df.loc[user_index, 'startYear']}")
                st.write(f"Director : {df.loc[user_index, 'directors']}")
                st.write(f"Note : {df.loc[user_index, 'averageRating']}")
                st.write(f"Synopsys : {df.loc[user_index, 'overview']}")
                st.write("\n")

                modelNN = NearestNeighbors(n_neighbors=nb)
                modelNN.fit(tfidf_matrix_filtre)
                _, indices = modelNN.kneighbors(tfidf_matrix_filtre[user_index_filtre])

                with col2:
                    st.header("Recommended films:")
                    for index in indices[0]:
                        if index != user_index_filtre:
                            recommendation = df_filtre.loc[index]
                            st.image(
                                f"https://image.tmdb.org/t/p/original/{recommendation['poster_path']}"
                            )
                            st.write(f"Nom : **{recommendation['originalTitle']}**")
                            st.write(f"Genres : {recommendation['genres']}")
                            st.write(f"Année : {recommendation['startYear']}")
                            st.write(f"Director : {recommendation['directors']}")
                            st.write(f"Note : {recommendation['averageRating']}")
                            st.write(f"Synopsys : {recommendation['overview']}")
                            st.write("\n")

with tab2:

    st.header("BDD IMDB DATAVIZ")
    st.divider()
    graph_1 = str(chemin / "graph_1.png")
    graph_2 = str(chemin / "graph_2.png")
    graph_3 = str(chemin / "graph_3.png")
    graph_4 = str(chemin / "graph_4.png")
    st.image(graph_1)
    st.divider()
    st.image(graph_2)
    st.divider()
    st.image(graph_3)
    st.divider()
    st.image(graph_4)

with tab3:

    st.header("BDD Movie Finder DATAVIZ")
    st.divider()

    # Distribution des années de sortie des films
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_filtre['startYear'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution des années de sortie des films')
    ax.set_xlabel('Année de sortie')
    ax.set_ylabel('Nombre de films')
    st.pyplot(fig)

    # Distribution des notes moyennes des films
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_filtre['averageRating'], bins=20, kde=True, ax=ax)
    ax.set_title('Distribution des notes moyennes des films')
    ax.set_xlabel('Note moyenne')
    ax.set_ylabel('Nombre de films')
    st.pyplot(fig)

    # Top des genres les plus populaires
    fig, ax = plt.subplots(figsize=(10, 6))
    df_filtre['genres'].str.split(',').explode().value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Top des genres les plus populaires')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Nombre de films')
    st.pyplot(fig)

    # Corrélation entre la durée et la note moyenne des films
    fig, ax = plt.subplots(figsize=(12, 8))

    # Créer un dataframe pour les genres en les explosant (oui j'aurais du le faire dans le dataframe déjà, au moins pour les split (',') :) )
    genre_df = df_filtre.assign(genre=df_filtre['genres'].str.split(',')).explode('genre')

    # graph
    sns.boxplot(data=genre_df, x='genre', y='averageRating', ax=ax)
    ax.set_title('Distribution des notes moyennes par genre')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Note moyenne')
    ax.tick_params(axis='x', rotation=45)  # Rotation des étiquettes de l'axe Xpour la lisibilité
    st.pyplot(fig)
