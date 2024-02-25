import mysql.connector
from openai import OpenAI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast
from dotenv import dotenv_values

db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'root',
    'database': 'product_recommendations_embedding'
}
config = dotenv_values(".env")
OPENAI_EMBEDDINGS_KEY = config['API_KEY']
EMBEDDING_MODEL = "text-embedding-3-small"


def get_all_products():
    products = []
    try:
        # Connect to MySQL database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # Execute SQL query to select all products
        cursor.execute("SELECT UniqueID,Title,Description,Price FROM allembeds")

        # Fetch all products from the cursor
        products = cursor.fetchall()

        # Close cursor and connection
        cursor.close()
        conn.close()

        return products
    except Exception as e:
        print("Error fetching products:", e)
        return None


def get_product_by_id(product_id):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    product = None
    # Execute SQL query to select all products
    cursor.execute("SELECT UniqueID,Title,Description,Price FROM allembeds WHERE UniqueID = '{}'".format(product_id))
    product = cursor.fetchone()
    product_title = product['Title']
    return product


def generate_single_embedding(text, openai_client, model):
    """
        Function to generate embedding from a given piece of text
        First check, if there is an embedding for this product title,
        If yes, return it, otherwise create it and store for the respective
        product.
    """
    embeddings = None
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    sql = "Select Embedding from allembeds where Title = '{}'".format(text)
    cursor.execute(sql)
    product_embedding = cursor.fetchone()

    try:
        embeddings = ast.literal_eval(product_embedding['Embedding'])
    except ValueError as v:
        embeddings = None

    if embeddings is None:
        response = openai_client.embeddings.create(
            input=text,
            model=model
        )

        embeddings = response.data[0].embedding
    return embeddings


def load_embeddings_from_csv(file_path):
    df = pd.read_csv(file_path)
    embeddings = df['Embedding'].apply(ast.literal_eval)
    return np.array(embeddings.tolist())


# Function to compute similarity between text and embeddings
def compute_similarity(query_embedding, embeddings):
    similarities = cosine_similarity(query_embedding, embeddings)
    return similarities[0]


# Function to get top 5 similar records
def top_5_similar_records(similarities, df_records, input_title, n=5):
    similar_records = []

    input_title = input_title.lower()
    top_indices = similarities.argsort()[-(n + 1):][::-1]  # Include an extra item to account for input title
    similar_titles = df_records.iloc[top_indices]['Title'].str.lower().tolist()
    similar_titles_ids = df_records.iloc[top_indices]['UniqueID'].tolist()

    for idx in top_indices:
        if df_records.iloc[idx]['Title'].lower() != input_title:
            title = df_records.iloc[idx]['Title']
            ids = df_records.iloc[idx]['UniqueID']
            similarity = similarities[idx]
            similar_records.append((title, ids, similarity))
    return similar_records


def fetch_similar_products(product_title):
    client = OpenAI(api_key=OPENAI_EMBEDDINGS_KEY)
    # Step 1 : Fetch All embeddings from file
    extracted_embeddings = load_embeddings_from_csv('allembeds.csv')

    # Step 2: Get embedding of the item
    emb_query = generate_single_embedding(product_title, client, EMBEDDING_MODEL)

    # Step 3: Transform embeddings for computing similarity
    text_embedding_array = np.array(emb_query)
    text_embedding_reshaped = text_embedding_array.reshape(1, -1)
    similarities = compute_similarity(text_embedding_reshaped, extracted_embeddings)

    df = query_to_df()
    # Step 4: Print Similarities:
    top5 = top_5_similar_records(similarities, df, product_title)
    return top5


def query_to_df():
    products = get_all_products()
    df = pd.DataFrame(products, columns=['UniqueID', 'Title', 'Description', 'Price'])
    return df
