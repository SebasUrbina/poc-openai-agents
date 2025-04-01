# example from: https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/routing.py
import asyncio
import uuid
from agents import (
    Runner,
    TResponseInputItem,
    trace,
)
from loguru import logger
from src.agents.intent_agent import intent_agent
from src.agents.file_search_agent import create_file_search_agent
import json


# read this: https://github.com/openai/openai-agents-python/blob/main/examples/handoffs/message_filter_streaming.py
async def main():
    logger.info("Starting NetworkGPT")
    conversation_id = str(uuid.uuid4().hex)

    msg = input("Hi, how can I help you today?: ")

    # Historial de la conversacion
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        # Each conversation turn is a single trace. Normally, each input from the user would be an
        # API request to your app, and you can wrap the request in a trace()
        with trace("NetworkGPT", group_id=conversation_id):
            # TODO: Use streaming
            logger.info(f"Inputs: {json.dumps(inputs, indent=2)}")
            logger.info("Running intent agent")
            intent_result = await Runner.run(intent_agent, inputs)
            logger.info(f"Intent result: {intent_result.final_output}")

            query = intent_result.final_output.query
            nemonico = intent_result.final_output.nemonico
            doc_category = intent_result.final_output.doc_category

            # TODO: Create the agent workflow
            if nemonico and doc_category:
                file_search_agent = create_file_search_agent(
                    query, nemonico, doc_category.value
                )
                file_search_agent_result = await Runner.run(file_search_agent, inputs)
                logger.info(
                    f"File search agent result: {file_search_agent_result.final_output}"
                )
            else:
                logger.warning("No intent result found")
                user_msg = input("Enter a message: ")
                inputs.append({"content": user_msg, "role": "user"})
                continue

        inputs = file_search_agent_result.to_input_list()
        user_msg = input("Enter a message: ")
        inputs.append({"content": user_msg, "role": "user"})


if __name__ == "__main__":
    asyncio.run(main())
