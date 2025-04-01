from enum import Enum
from agents import Agent
from pydantic import BaseModel, Field

PROMPT = (
    "Eres un experto en identificar la intención del usuario y devuelve un "
    "objeto con la query, el nemonico del POP/Antena/Sitio y la categoría de documento."
)


class Categories(str, Enum):
    energia = "energia"
    revisiones = "revisiones"


class AgentOutput(BaseModel):
    query: str
    nemonico: str | None = Field(
        description="Nemonico del POP/Antena/Sitio. Ejemplo: SA001, FN633, etc..."
    )
    doc_category: Categories | None


# Agente que extrae la intención del usuario y devuelve un objeto con la query, el nemonico del POP/Antena/Sitio y la categoría de documento.
intent_agent = Agent(
    name="Intent agent",
    instructions=PROMPT,
    output_type=AgentOutput,
)
