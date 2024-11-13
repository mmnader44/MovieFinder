# Movie Finder 🎬

## Description
**Movie Finder** est une application de recommandation de films développée en Python avec Streamlit. Basée sur les bases de données IMDB et TMDB, elle utilise le machine learning pour suggérer des films similaires à ceux que vous aimez. Ce projet a été réalisé dans le cadre de mon programme de Data Analyst à la Wild Code School.

## Fonctionnalités
- Recommandation de films similaires en fonction d’un film choisi par l’utilisateur.
- Option pour ajuster le nombre de recommandations.
- Filtrage par genre pour affiner les recommandations.
- Visualisation de données et analyses de la base IMDB et TMDB avec des graphiques.

## Installation
Pour exécuter ce projet en local, suivez les étapes ci-dessous :

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/mmnader44/MovieFinder.git
   ```

2. Accédez au dossier du projet :
   ```bash
   cd MovieFinder
   ```

3. Installez les dépendances nécessaires :
   ```bash
   pip install -r requirements.txt
   ```

4. Placez les fichiers de données (`df_final.csv`, `tfidf_matrix.joblib`, `indices.joblib`, `titles.joblib`) dans le répertoire du projet.

5. Lancez l’application Streamlit :
   ```bash
   streamlit run app.py
   ```

## Utilisation
1. Sélectionnez un film depuis la liste déroulante dans l'application.
2. Ajustez le nombre de recommandations grâce au slider.
3. Utilisez le filtre de genres pour limiter les recommandations à un type de film spécifique.
4. Explorez les autres onglets pour visualiser les données de la base IMDB et les analyses.

## Aperçu de l'application

https://moviefinder-wcs.streamlit.app/

## Techniques de Machine Learning
Le modèle de recommandation utilise **Nearest Neighbors** pour trouver les films similaires en fonction de la matrice TF-IDF, qui analyse la similitude des descriptions de films. 
Les données sont nettoyées et transformées avec Pandas pour garantir la qualité des recommandations.

## Données
Les données proviennent de deux sources :
- **IMDB** : Informations de base sur les films, y compris le genre, la note et l'année de sortie.
- **TMDB** : Affiches et résumés des films pour enrichir l'expérience utilisateur.

## Auteur
Créé par **NADER Mehdi-Michel** dans le cadre de mon programme de formation Data Analyst à la Wild Code School Nantes