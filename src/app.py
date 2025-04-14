import streamlit as st
import pandas as pd
import pickle
from explore import df, X_vectorized


model = pickle.load(open("../models/Nearest_neighbour.pkl", "rb"))

st.title('Bienvenido al recomendador de peliculas!')
st.page_link('https://github.com/Reezo912', label='Mi Github')


st.markdown('Este recomendador de peliculas utiliza un algoritmo de Nearest Neighbours, agrupando automaticamente las peliculas que tienen cosas en comun.')

#st.subheader('Introduce el nombre de la película en inglés')


lista_peliculas = df['title']

pelicula_seleccionada = st.selectbox("Nombre de tu pelicula en inglés:",(lista_peliculas), index=None, placeholder='Escribe aqui...')

def get_movie_recommendations(movie_title, dataset, model, x):
            # Obtener el índice de la película
            movie_index = dataset[dataset["title"] == movie_title].index[0]
            
            # Encontrar los Nearest Neighbours
            distances, indices = model.kneighbors(x[movie_index], n_neighbors=6)
            
            # Hago la lista con los vecinos encontrados, excluyo el primer titulo en el output
            similar_movies = [(dataset["title"].iloc[i], distances[0][j]) for j, i in enumerate(indices[0])]

            return similar_movies[1:]


if st.button("Prediccion!"):
    # Suponiendo que 'df' es tu DataFrame original y 'X_vectorized' la matriz TF-IDF
    recommendations = get_movie_recommendations(pelicula_seleccionada, df, model, X_vectorized)

    if recommendations:
        st.write(f"**Las recomendaciones para** '{pelicula_seleccionada}' son:")
        for movie, distance in recommendations:
            st.write(f"- {movie}")
            
        st.write('Si te han gustado las peliculas, por favor pon nota para que sepa el comportamiento del modelo. Gracias!')
        sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
        selected = st.feedback("thumbs")
        if selected is not None:
            st.markdown(f"You selected: {sentiment_mapping[selected]}")


    else:
        st.write("No he encontrado ninguna pelicula con ese titulo, revisa que este todo bien escrito!")






