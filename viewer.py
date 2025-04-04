import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd 
import streamlit as st
import argparse

parser = argparse.ArgumentParser(description='Set the Chroma DB path to view collections')
parser.add_argument('db')

pd.set_option('display.max_columns', 4)

def view_collections(dir):

    
    st.markdown("### DB Path: %s" % dir)

    client = chromadb.PersistentClient(path="./chroma_db")
    collectins = client.list_collections()
    # print the first collection
    print("Collections:", collectins)
    collection = client.get_collection(name="resumes")
    # print(collection.get(include=['embeddings', 'documents', 'metadatas']))
    data = collection.get(include=["ids", "embeddings", "metadatas"])
    # print the first data from the collection
    # print("Data:", data)
    # print("Collection Name:", collection.name)

    st.header("Collections")

    ids = data['ids']
    embeddings = data["embeddings"]
    # print first 1 embedding
    print("Embeddings:", embeddings[0])
    metadata = data["metadatas"]
    df = pd.DataFrame.from_dict(data)
    st.dataframe(df)

    # # for collection in client.list_collections():
    #     print(collection.get())
    #     data = collection.get()
    #     # print the data from the collection
    #     print("Collection Name:", collection.name)
    #     print("IDs:", data['ids'])
    #     print("Embeddings:", data["embeddings"])
    #     print("Metadata:", data["metadatas"])
    #     print("Documents:", data["documents"])

    #     ids = data['ids']
    #     embeddings = data["embeddings"]
    #     metadata = data["metadatas"]
    #     documents = data["documents"]

    #     df = pd.DataFrame.from_dict(data)
    #     st.markdown("### Collection: **%s**" % collection.name)
    #     st.dataframe(df)

if __name__ == "__main__":
    try:
        args = parser.parse_args()
        print("Opening database: %s" % args.db)
        view_collections(args.db)
    except:
        pass

