from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("f0d9f7f3-b513-4bfa-bbaa-e803494d8e01")
PINECONE_API_ENV = os.environ.get("gcp-starter")

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
        api_key=os.environ.get("f0d9f7f3-b513-4bfa-bbaa-e803494d8e01")
    )

    # Now do stuff
if 'my_index' not in pc.list_indexes().names():
        pc.create_index(
            name='my_index', 
            dimension=1536, 
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )




index_name="medical-chatbot"

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
