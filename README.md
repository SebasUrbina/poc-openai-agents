# Poc OpenAI Agents

## Description

This is a proof of concept for using OpenAI agents to answer questions about a set of documents with metadata filters.

### Use Case

You have multiple documents for categories of first and second level. For example. Your documents are about of antennas of telecomunications. Each antenna has a first level category (the site of the antenna) and a second level category (the type of information you know about the antenna).

You want to answer questions about antennas. 

For example. If you have a set of documents like:

```
/data
    /storage
        /SA001
            /energia
                consumo_energetico_sa001.txt
            /revisiones
                historial_revisiones_sa001.txt
        /FN633
            /energia
                consumo_energetico_fn633.txt
            /revisiones
                historial_revisiones_fn633.txt
        /MA300
            /energia
                consumo_energetico_ma300.txt
            /revisiones
                historial_revisiones_ma300.txt
```

The agent identify the user intention and return a structured output with the following fields:

- query: The query of the user
- site: The site of the antenna (SA001,FN663 or MA300)
- category: The category of the information (energia or revisiones)

The a second agent is used to search the information in the vector database using the site and category to filter the results and improve the accuracy of the answer.

## Setup

You need to have installed `uv` to run the project.

```bash
uv sync
```

## Usage

To start chatting you have to run the main.py file.

```bash
python main.py
```
