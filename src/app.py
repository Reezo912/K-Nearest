import streamlit as st
import pandas as pd
import pickle
from explore import df, X_vectorized


model = pickle.load(open("../models/Nearest_neighbour.pkl", "rb"))

st.title('Bienvenido al recomendador de peliculas!')
st.markdown('Este recomendador de peliculas utiliza un algoritmo de Nearest Neighbours')
st.subheader('Introduce el nombre de la pelicula en Ingles')

# st.text_input('Aqui escribes tu pelicula', placeholder='Avatar')


lista_peliculas = df['title']

pelicula_seleccionada = st.selectbox("Nombre de tu pelicula",(lista_peliculas),)

def get_movie_recommendations(movie_title, dataset, model, x):
            # Obtener el índice de la película
            movie_index = dataset[dataset["title"] == movie_title].index[0]
            
            # Encontrar los Nearest Neighbours
            distances, indices = model.kneighbors(x[movie_index], n_neighbors=6)
            
            # Hago la lista con los vecinos encontrados, excluyo el primer titulo en el output
            similar_movies = [(dataset["title"].iloc[i], distances[0][j]) for j, i in enumerate(indices[0])]

            return similar_movies[1:]


if st.button("Predict"):
    # Suponiendo que 'df' es tu DataFrame original y 'X_vectorized' la matriz TF-IDF
    recommendations = get_movie_recommendations(pelicula_seleccionada, df, model, X_vectorized)

    if recommendations:
        st.write(f"**Film recommendations for:** '{pelicula_seleccionada}'")
        for movie, distance in recommendations:
            st.write(f"- {movie}")
    else:
        st.write("I didn't find that title! Check spelling!")