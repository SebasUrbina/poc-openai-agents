# example from: https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/routing.py
import asyncio
import uuid
from enum import Enum
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import (
    Agent,
    FileSearchTool,
    RawResponsesStreamEvent,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    trace,
    InputGuardrail,
    input_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
)
from pydantic import BaseModel, Field
from loguru import logger
import json


class Categories(str, Enum):
    energia = "energia"
    revisiones = "revisiones"


class AgentOutput(BaseModel):
    query: str
    nemonico: str | None = Field(
        description="Nemonico del POP/Antena/Sitio. Ejemplo: SA001, FN633, etc..."
    )
    doc_category: Categories | None


class ConsultaOutput(BaseModel):
    is_valid_query: bool
    reasoning: str


guardrail_agent = Agent(
    name="Agente de validación",
    instructions="Verifica si la consulta del usuario es válida y está relacionada con antenas de telecomunicaciones y entrega el nemonico de un POP. Por ejemplo: AT001",
    output_type=ConsultaOutput,
)


@input_guardrail
async def consulta_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input_data: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """This is an input guardrail function, which happens to call an agent to check if the input is valid and related to antennas."""
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(ConsultaOutput)

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_valid_query,
    )


intent_agent = Agent(
    name="Intent agent",
    instructions="Eres un experto en identificar la intención del usuario y devuelve un objeto con la query, el nemonico del POP/Antena/Sitio y la categoría de documento.",
    output_type=AgentOutput,
    input_guardrails=[consulta_guardrail],
)


def create_file_search_agent(query: str, nemonico: str, doc_category: Categories):
    return Agent(
        name="File searcher",
        instructions="Eres un experto en redes de entregar informacion factual al usuario en base al contexto que recibes.",
        tools=[
            FileSearchTool(
                max_num_results=3,
                vector_store_ids=["vs_67e96e2e97888191be383a9f1349129a"],
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


# read this: https://github.com/openai/openai-agents-python/blob/main/examples/handoffs/message_filter_streaming.py
async def main():

    logger.info("Starting NetworkGPT")
    conversation_id = str(uuid.uuid4().hex)
    inputs: list[TResponseInputItem] = []

    while True:
        with trace("NetworkGPT", group_id=conversation_id):
            try:
                user_input = input("Hola, ¿en qué puedo ayudarte hoy?: ")
                inputs.append({"content": user_input, "role": "user"})
                logger.info(f"Inputs: {json.dumps(inputs, indent=2)}")
                logger.info("Running intent agent")
                result = await Runner.run(intent_agent, inputs)
                logger.info(f"Intent result: {result.final_output}")

                query = result.final_output.query
                nemonico = result.final_output.nemonico
                doc_category = result.final_output.doc_category

                if nemonico and doc_category:
                    file_search_agent = create_file_search_agent(
                        query, nemonico, doc_category.value
                    )
                    logger.info("Running file search agent")
                    result = await Runner.run(file_search_agent, inputs)
                    inputs = result.to_input_list()
                    logger.info(f"File search agent result: {result.final_output}")
                else:
                    logger.warning("No intent result found")
                    msg = "Lo siento, no puedo identificar el POP/Antena/Sitio. Por favor, intenta reformular tu consulta."
                    logger.warning(msg)
                    inputs.append({"content": msg, "role": "assistant"})

            except InputGuardrailTripwireTriggered:
                logger.error(f"Triggered guardrail")
                msg = "Lo siento, hubo un error. Por favor, intenta reformular tu consulta."
                logger.warning(msg)
                inputs.append({"content": msg, "role": "assistant"})


if __name__ == "__main__":
    asyncio.run(main())
