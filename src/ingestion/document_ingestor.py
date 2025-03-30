import os
from tqdm import tqdm
from openai import OpenAI
from loguru import logger

client = OpenAI()

ROOT = "./data/storage"
VECTOR_DB_NAME = "DataPops"

# Create vector store (if you run multiple times, it will create multiple vectordbs)
logger.info(f"Creating vector store {VECTOR_DB_NAME}")
vector_store = client.vector_stores.create(  # Create vector store
    name=VECTOR_DB_NAME,
)

FILE_METADATA = {}
logger.info(f"Listing files in {ROOT}")
for l1_level in os.listdir(ROOT):
    for l2_level in os.listdir(os.path.join(ROOT, l1_level)):
        for file in os.listdir(os.path.join(ROOT, l1_level, l2_level)):
            FILE_METADATA[file] = {
                "pop": l1_level,
                "category": l2_level,
                "path": os.path.join(ROOT, l1_level, l2_level, file),
            }

# list currents vector DBS
# vector_stores = client.vector_stores.list()
# for vector_store in vector_stores:

# Upload Data
logger.info(f"Uploading files to vector store {VECTOR_DB_NAME}")
for file_name, metadata in tqdm(FILE_METADATA.items(), desc="Uploading files"):

    # Upload file
    response = client.vector_stores.files.upload_and_poll(  # Upload file
        vector_store_id=vector_store.id,
        file=open(metadata["path"], "rb"),
    )
    uploaded_file_id = response.id

    # Update file metadata
    response = client.vector_stores.files.update(
        vector_store_id=vector_store.id,
        file_id=uploaded_file_id,
        attributes={
            "pop": metadata["pop"],
            "category": metadata["category"],
        },
    )

# List uploaded files
logger.info(f"Listing uploaded files in vector store {VECTOR_DB_NAME}")
for file in client.vector_stores.files.list(vector_store_id=vector_store.id):
    print(file.model_dump())
