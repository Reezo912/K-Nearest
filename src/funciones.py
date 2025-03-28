import pandas as pd
import numpy as np
import unicodedata
import re
import string
from collections import Counter


def get_unique_characters_count(df, column_name="review"):
    # Unimos todo el texto de la columna en una sola cadena
    all_text = ''.join(df[column_name].dropna().astype(str))
    # Obtenemos un contador con la frecuencia de cada carácter
    char_counts = Counter(all_text)
    # Convertimos el contador en un diccionario ordenado (por ejemplo, alfabéticamente)
    sorted_char_counts = dict(sorted(char_counts.items()))
    return sorted_char_counts


def clean_text_for_words(text, keep_numbers=False):
    """
    Limpia el texto para análisis a nivel de palabras:
      - Convierte a minúsculas.
      - Elimina dígitos y cualquier carácter que no sea una letra (a-z) o espacio, si keep_numbers es False.
        Si keep_numbers es True, permite dígitos.
      - Elimina espacios extra.
    
    Args:
        text (str): Texto a limpiar.
        keep_numbers (bool): Indica si se deben mantener los números en el texto. Por defecto, False.
    
    Returns:
        str: Texto limpio.
    """
    text = text.lower()
    if keep_numbers:
        # Permitir letras (a-z), dígitos (0-9) y espacios
        pattern = r'[^a-z0-9\s]'
    else:
        # Permitir solo letras (a-z) y espacios
        pattern = r'[^a-z\s]'
    text = re.sub(pattern, '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_reviews_for_word_analysis(df, column="review", keep_numbers=False):
    """
    Recibe un DataFrame y limpia la columna indicada para análisis a nivel de palabras,
    eliminando caracteres no deseados según el parámetro keep_numbers.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene la columna de reviews.
        column (str): Nombre de la columna a limpiar (por defecto "review").
        keep_numbers (bool): Indica si se deben mantener los números en el texto.
        
    Returns:
        pd.DataFrame: DataFrame con la columna limpia para análisis.
    """
    df[column] = df[column].astype(str).apply(lambda x: clean_text_for_words(x, keep_numbers=keep_numbers))
    return df