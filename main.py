import streamlit as st
import cohere
import numpy as np
import os

def rerank_example(co):
    docs = ['Carson City is the capital city of the American state of Nevada.',
            'The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.',
            'Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.',
            'Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.']

    model_name = 'rerank-english-v2.0'
    query = 'What is the capital of the United States?'
    response = co.rerank(
        model=model_name,
        query=query,
        documents=docs,
        top_n=3,
    )
    print(response)


def chat_example(co):
    chat_history = [
        {"role": "USER", "message": "Who discovered gravity?"},
        {"role": "CHATBOT", "message": "The man who is widely credited with discovering gravity is Sir Isaac Newton"}
    ]
    msg = "What year was he born?"

    # Test chat
    response = co.chat(
        chat_history=chat_history,
        message=msg,
        # perform web search before answering the question. You can also use your own custom connector.
        connectors=[{"id": "web-search"}]
    )
    print("Chat result: ", response)

def embed_example(co):
    response = co.embed(
        texts=['hello', 'goodbye'],
        model='embed-english-v3.0',
        input_type='classification'
    )
    print("Embedding: ", response)


def summarize_example(co):
    text = (
        "Ice cream is a sweetened frozen food typically eaten as a snack or dessert. "
        "It may be made from milk or cream and is flavoured with a sweetener, "
        "either sugar or an alternative, and a spice, such as cocoa or vanilla, "
        "or with fruit such as strawberries or peaches. "
        "It can also be made by whisking a flavored cream base and liquid nitrogen together. "
        "Food coloring is sometimes added, in addition to stabilizers. "
        "The mixture is cooled below the freezing point of water and stirred to incorporate air spaces "
        "and to prevent detectable ice crystals from forming. The result is a smooth, "
        "semi-solid foam that is solid at very low temperatures (below 2 °C or 35 °F). "
        "It becomes more malleable as its temperature increases.\n\n"
        "The meaning of the name \"ice cream\" varies from one country to another. "
        "In some countries, such as the United States, \"ice cream\" applies only to a specific variety, "
        "and most governments regulate the commercial use of the various terms according to the "
        "relative quantities of the main ingredients, notably the amount of cream. "
        "Products that do not meet the criteria to be called ice cream are sometimes labelled "
        "\"frozen dairy dessert\" instead. In other countries, such as Italy and Argentina, "
        "one word is used fo\r all variants. Analogues made from dairy alternatives, "
        "such as goat's or sheep's milk, or milk substitutes "
        "(e.g., soy, cashew, coconut, almond milk or tofu), are available for those who are "
        "lactose intolerant, allergic to dairy protein or vegan."
    )

    print("Summary of text: ", co.summarize(text=text))


if __name__ == "__main__":
    api_key = None  # Type your API key
    co = cohere.Client(api_key)  # Set up the Cohere client

    chat_example(co)
    rerank_example(co)
    embed_example(co)
    summarize_example(co)
