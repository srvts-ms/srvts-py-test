import streamlit as sl
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "3f2bdd4be228b12cd5fc37f1c43a72cf03f3d8d1f35bad33663159f3e028c481"

llm = OpenAI(temperature=0.9)
sl.title("AI Recipe Generator")

selected_cuisine = sl.sidebar.selectbox(
    "Pick Cuisine",
    (
        "Karnataka",
        "South Indian",
        "North Indian",
        "Andra",
        "Hyderabad",
        "Kerala",
        "Tamilnadu",
        "Gujarati",
    ),
)


def generate_dish_name(cuisine):
    # Chain 1 --> Dish Name
    prompt_template_name = PromptTemplate(
        input_variables=["cuisine"],
        template="""
        Suggest me an iconic dish for a {cuisine} restaurant.
        I just want a dish name. 
        """,
    )

    chain_item_1 = LLMChain(
        llm=llm, prompt=prompt_template_name, output_key="dish_name"
    )

    # Chain 2 --> Ingredients
    prompt_template_ingredients = PromptTemplate(
        input_variables=["dish_name"],
        template="Now, give me the list of ingredients required for preparing {dish_name} in a CSV format.",
    )

    chain_item_2 = LLMChain(
        llm=llm, prompt=prompt_template_ingredients, output_key="ingredients"
    )

    # print(chain_item_1.run("Karnataka"))
    query_chain = SimpleSequentialChain(chains=[chain_item_1, chain_item_2])

    # Chain 3 --> Recipe
    prompt_template_recipe = PromptTemplate(
        input_variables=["dish_name", "ingredients"],
        template="Now, give me the recipe for the suggested dish {dish_name} "
        "and the suggested ingredients {ingredients}",
    )

    chain_item_3 = LLMChain(llm=llm, prompt=prompt_template_recipe, output_key="recipe")

    query_chain_seq = SequentialChain(
        chains=[chain_item_1, chain_item_2, chain_item_3],
        input_variables=["cuisine"],
        output_variables=["dish_name", "ingredients", "recipe"],
    )

    response_simple_seq = query_chain_seq(cuisine)

    return response_simple_seq


if selected_cuisine:
    response = generate_dish_name(selected_cuisine)
    sl.header(response["dish_name"])
    ingredients_list = response["ingredients"].split(",")
    recipe = response["recipe"].strip()

    # for ingredient in ingredients_list:
    #     sl.write("-", ingredient.strip())

    sl.write(recipe)