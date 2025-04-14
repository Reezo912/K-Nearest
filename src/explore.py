#!/usr/bin/env python
# coding: utf-8

# # Recomendador de peliculas

# In[37]:


import pandas as pd
import numpy as np
import sqlite3



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[38]:


#url_movies = 'https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv'

#url_credits = 'https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_credits.csv'


# In[39]:


df_movies = pd.read_csv('../data/raw/movies.csv')

df_credits = pd.read_csv('../data/raw/credits.csv')

#df_movies.to_csv('../data/raw/movies.csv')
#df_credits.to_csv('../data/raw/credits.csv')


# In[40]:


df_movies.sample(5)


# In[41]:


df_credits.sample(5)


# Voy a comprobar si mis titulos son unicos, las instrucciones indican que deberia hacer un join a traves de la columna titulo, pero si hay algun titulo repetido, esto borraria y solo dejaria uno.

# In[42]:


df_movies.shape


# In[43]:


df_movies['title'].nunique()


# In[44]:


repeated_titles = df_credits['title'][df_credits['title'].duplicated()].unique()
print(repeated_titles)


# Veo que si que tengo duplicados, voy a comprobar las filas a ver si son duplicados completos

# In[45]:


resultado = df_movies[df_movies['title'] == 'Batman']
resultado


# In[46]:


resultado = df_movies[df_movies['title'] == 'Out of the Blue']
resultado


# In[47]:


resultado = df_movies[df_movies['title'] == 'The Host']
resultado


# Puedo observar que son completamente diferentes, por lo tanto unir por titulo no seria factible, ya que estos repetidos se eliminarian

# Voy a comprobar duplicados en la columna ID, en caso de que no los haya sera la que utilice para unir mis dos datasets

# In[48]:


print(df_movies['id'].nunique())
print(df_credits['movie_id'].nunique())


# Puedo observar que no hay duplicados en los ids de cada uno de mis datasets, intentare hacer el join a traves de ellos, en caso de obtener mas filas de las que tengo originalmente, sera porque no coinciden entre si

# In[49]:


# Creo mi database y la conexion a ella
conn = sqlite3.connect('../data/processed/movie_database.db')

# Creo el objeto cursor que interactuara con ella
cursor = conn.cursor()

# Creo mis tablas
df_movies.to_sql('tabla_movies', conn, if_exists='replace', index=False)
df_credits.to_sql('tabla_credits', conn, if_exists='replace', index=False)


# In[50]:


query = """
CREATE TABLE IF NOT EXISTS tabla_resultados AS 
SELECT tabla_credits.movie_id, tabla_movies.title, tabla_movies.overview, 
       tabla_movies.genres, tabla_movies.keywords, tabla_credits.cast, tabla_credits.crew
FROM tabla_movies
INNER JOIN tabla_credits
ON tabla_movies.id = tabla_credits.movie_id;
"""


cursor.execute(query)
conn.commit()


# In[51]:


df = pd.read_sql('select * from tabla_resultados', conn)

conn.close()


# In[52]:


df.sample(5)


# In[53]:


df.shape


# Con esto he conseguido unir las dos bases de datos sin eliminar las filas repetidas.

# nearest neighbours, tf-idf

# In[54]:


import json

df = df.assign(genres=lambda x: x['genres'].apply(lambda y: [item['name'] for item in json.loads(y)] if pd.notna(y) else None))
df = df.assign(keywords=lambda x: x['keywords'].apply(lambda y: [item['name'] for item in json.loads(y)] if pd.notna(y) else None))
df = df.assign(cast=lambda x: x['cast'].apply(lambda y: [item['name'] for item in json.loads(y)][:3] if pd.notna(y) else None))

# columna crew
df = df.assign(crew=lambda x: x['crew'].apply(lambda y: next((item['name'] for item in json.loads(y) if item.get('job') == 'Director'), None) if pd.notna(y) else None))

df = df.assign(overview=lambda x: x['overview'].apply(lambda y: [str(y)] if pd.notna(y) else []))


# In[55]:


df.head()


# In[56]:


df.head(5)


# In[57]:


from funciones import remove_spaces

# Borro los espacios en las columnas
df['genres'] = df['genres'].apply(remove_spaces)
df['cast'] = df['cast'].apply(remove_spaces)
df['crew'] = df['crew'].apply(remove_spaces)
df['keywords'] = df['keywords'].apply(remove_spaces)


# In[58]:


df.head(5)


# In[59]:


df["genres"] = df["genres"].apply(lambda x: [str(genre) for genre in x])
df["keywords"] = df["keywords"].apply(lambda x: [str(keyword) for keyword in x])
df["cast"] = df["cast"].apply(lambda x: [str(actor) for actor in x])

# Convierto mi columna 'crew' en una lista para poder hacer el .join
df["crew"] = df["crew"].apply(lambda x: [x] if pd.notna(x) and isinstance(x, str) else [])


# In[60]:


df.head(5)


# In[61]:


df["tags"] = df["overview"] + df["genres"] + df["keywords"] + df["cast"] + df["crew"]
df["tags"] = df["tags"].apply(lambda x: ",".join(x).replace(",", " "))


# In[62]:


df.drop(columns=["genres", "keywords", "cast", "crew", "overview"], inplace=True)

df.iloc[0].tags


# In[63]:


df.head()


# ## Vectorizado

# In[64]:


from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
import pickle


stop_words = get_stop_words('en')

x = df['tags']

tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
X_vectorized = tfidf_vectorizer.fit_transform(x)

with open('../models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(X_vectorized, f)


# In[65]:


from sklearn.neighbors import NearestNeighbors

# cosine: mide similitud del angulo entre dos vectores, análisis de textos y datos de alta dimensión.
nn_model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute', n_jobs=-1)

nn_model.fit(X_vectorized)

with open('../models/Nearest_neighbour.pkl', 'wb') as f:
    pickle.dump(nn_model, f)


# In[ ]:

if __name__ == '__main__':
        
    input_title = input('Write the title of the movie: ')

    def get_movie_recommendations(movie_title, dataset, model, x):
        # Obtener el índice de la película
        movie_index = dataset[dataset["title"] == movie_title].index[0]
        
        # Encontrar los Nearest Neighbours
        distances, indices = model.kneighbors(x[movie_index], n_neighbors=6)
        
        # Hago la lista con los vecinos encontrados, excluyo el primer titulo en el output
        similar_movies = [(dataset["title"].iloc[i], distances[0][j]) for j, i in enumerate(indices[0])]

        return similar_movies[1:]

    # Suponiendo que 'df' es tu DataFrame original y 'X_vectorized' la matriz TF-IDF
    recommendations = get_movie_recommendations(input_title, df, nn_model, X_vectorized)

    if recommendations:
        print("Film recommendations for '{}'".format(input_title))
        for movie, distance in recommendations:
            print("- Film: {}".format(movie))
    else:
        print("I didnt find that title! Check spelling!")

