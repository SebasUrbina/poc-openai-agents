from enum import Enum
from agents import Agent, FileSearchTool
from pydantic import BaseModel, Field
from .intent_agent import Categories

PROMPT = (
    "Eres un experto en redes de entregar informacion factual al usuario en "
    "base al contexto que recibes."
)

VECTOR_STORE_ID = "vs_67e96e2e97888191be383a9f1349129a"


# Agente que busca en la base de datos de documentos y entrega la información al usuario.
def create_file_search_agent(query: str, nemonico: str, doc_category: Categories):
    return Agent(
        name="File searcher",
        instructions="Eres un experto en redes de entregar informacion factual al usuario en base al contexto que recibes.",
        tools=[
            FileSearchTool(
                max_num_results=3,
                vector_store_ids=[VECTOR_STORE_ID],
                include_search_results=True,
                filters={
                    "type": "and",
                    "filters": [
                        {"type": "eq", "key": "category", "value": doc_category},
                        {"type": "eq", "key": "pop", "value": nemonico},
                    ],
                },
            )
        ],
    )
