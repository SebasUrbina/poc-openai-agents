# example from: https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/routing.py
import asyncio
import uuid
from enum import Enum
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import (
    Agent,
    FileSearchTool,
    RawResponsesStreamEvent,
    Runner,
    TResponseInputItem,
    trace,
    InputGuardrail,
    GuardrailFunctionOutput,
)
from pydantic import BaseModel, Field
from loguru import logger


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


async def consulta_guardrail(ctx, agent, input_data):
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
    input_guardrails=[
        InputGuardrail(guardrail_function=consulta_guardrail),
    ],
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

    msg = input("Hola, ¿en qué puedo ayudarte hoy?: ")

    agent = intent_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        with trace("NetworkGPT", group_id=conversation_id):
            try:
                logger.info("Running intent agent")
                intent_result = await Runner.run(agent, inputs)
                logger.info(f"Intent result: {intent_result.final_output}")

                query = intent_result.final_output.query
                nemonico = intent_result.final_output.nemonico
                doc_category = intent_result.final_output.doc_category

                # TODO: Create the agent workflow
                if nemonico and doc_category:
                    file_search_agent = create_file_search_agent(
                        query, nemonico, doc_category.value
                    )
                    file_search_agent_result = await Runner.run(
                        file_search_agent, inputs
                    )
                    logger.info(
                        f"File search agent result: {file_search_agent_result.final_output}"
                    )
                else:
                    logger.warning("No intent result found")
                    user_msg = input("Ingresa un mensaje: ")
                    inputs.append({"content": user_msg, "role": "user"})
                    continue

            except Exception as e:
                logger.error(f"Error durante la ejecución: {str(e)}")
                print(
                    "Lo siento, hubo un error. Por favor, intenta reformular tu consulta."
                )
                user_msg = input("Ingresa un mensaje: ")
                inputs.append({"content": user_msg, "role": "user"})
                continue

        inputs = file_search_agent_result.to_input_list()
        user_msg = input("Enter a message: ")
        inputs.append({"content": user_msg, "role": "user"})


if __name__ == "__main__":
    asyncio.run(main())
