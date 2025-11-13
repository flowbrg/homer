###############################################################################
#DEPRECATED: This module is deprecated and will be removed in future versions.#
#It is kept for the clear and update_database methods, which will be re used. #
###############################################################################


# src/agents/database_agent.py
#from pathlib import Path

#import faiss
#from langchain_community.docstore.in_memory import InMemoryDocstore
#from langchain_community.vectorstores import FAISS

#from parser.parser import simple_parser
#from utils.utils import load_embeddings
#from schemas.configuration import AppConfig
#from langchain_core.vectorstores import VectorStore

#def _init_vectorstore(embeddings, index, vectorstore_path: Path) -> None:
#  vector_store = FAISS(
#    embedding_function=embeddings,
#    index=index,
#    docstore=InMemoryDocstore(),
#    index_to_docstore_id={},
#  )
#  vector_store.save_local(vectorstore_path)


#def load_vectorstore(config: AppConfig) -> VectorStore:
#  vectorstore_path = Path(config.user_data.user_data_path) / "faiss_index"
#  embeddings = load_embeddings(config.models.embeddings)
#  index = faiss.IndexHNSWFlat(
#    len(embeddings.embed_query("hello world")),
#    32
#  )

#  if not vectorstore_path.is_dir():
#    _init_vectorstore(config, index, vectorstore_path)
  
#  vectorstore = FAISS.load_local(
#    vectorstore_path,
#    embeddings,
#    allow_dangerous_deserialization=True
#  )

#  return vectorstore


#def clear(config: dict):
#  db_name = config["db_name"]
#  conn = connections.connect(uri=config["uri"])
#  existing_databases = db.list_database()
#  assert db_name in existing_databases, f'There are no database with the name {db_name}'

  # Use the database context
#  db.using_database(db_name)

  # Drop all collections in the database
#  collections = utility.list_collections()
#  for collection_name in collections:
#    collection = Collection(name=collection_name)
#    collection.drop()
#    print(f"Collection '{collection_name}' has been dropped.")

#  db.drop_database(db_name)
    
#  db.create_database(db_name)
#  print(f"Database '{db_name}' reset successfully.")

#def update_database(config: dict):
#  kwargs={}
#  if "disable_clear_temp" in config["args"]:
#    kwargs["disable_clear_temp"]=True
#  if "ignore_image_processing" in config["args"]:
#    kwargs["ignore_image_processing"]=True
#  if "ignore_pdf_preprocessing" in config["args"]:
#    kwargs["ignore_pdf_preprocessing"]=True
#  documents=preprocess(data_path=config["data_path"], **kwargs)
#  vectorstore=load_vectorstore(config=config)
#  vectorstore.add_documents(documents)