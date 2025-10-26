"""iBot Cog for Discord Bot"""

from qdrant_client import QdrantClient
from dotenv import load_dotenv  # For loading API key from a .env file
# import google.generativeai as genai
from langchain_qdrant import Qdrant  # Qdrant Vector Store Wrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import pandas as pd
import json
from tqdm.auto import tqdm
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import nextcord
from datetime import datetime
from nextcord.ext import commands
from typing import List
from helpers.utils import load_config
import ollama
from tqdm.auto import tqdm

tqdm.pandas(desc="Processing DataFrame")
config = load_config()

OLLAMA_MODEL_NAME = 'llama3.2:latest'
CLEANING_PATTERN = r'[^a-zA-Z0-9]'

LLM_PERSONA = '''
You are a helpful assistant that helps user to find recipes from the Indian Food Dataset.
Your pesona is polite, friendly and helpful.
You are not a chef, you are a recipe search assistant.

While Trying to sovle the customer's query, you can use the following information:
 - You can ask clarifying questions to understand the user's needs better.
 - You can use the Indian Food Dataset to find recipes that match the user's request.
 - You can provide additional information about the recipes, such as preparation time, ingredients, and instructions.
 - You can suggest alternatives or modifications to the recipes based on the user's preferences.

You are not allowed to:
 - Provide medical or dietary advice.
 - Make assumptions about the user's dietary restrictions or preferences without asking.
 - Provide personal opinions or recommendations that are not based on the dataset.
 - Provide recipes that are not in the Indian Food Dataset.
 - Provide recipes that are not relevant to the user's query.

> RESPOND ONLY IN ENGLISH, DO NOT RESPOND IN ANY OTHER LANGUAGE.
You are not allowed to use any other language than English.
'''

OBJECTIVE_PROMPT = '''
Your objective is to help the user find recipes from the Indian Food Dataset.
Your hidden side task is also to suggest, cross sell and/upsell dishes.
'''

PROMPT = '''
Your persona:
{llm_persona}

Your objective:
{objective_prompt}

User Query: {user_message}

Chat History:
{chat_history}

Searched Context from Database:
{context}
'''

RAG_PROMPT = '''
User Query: {user_message}
Chat History:
{chat_history}

'''


columns = ['TranslatedRecipeName', 'TranslatedIngredients',
           'PrepTimeInMins', 'CookTimeInMins', 'TotalTimeInMins', 'Servings',
           'Cuisine', 'Course', 'Diet', 'TranslatedInstructions', 'URL',
           'ComplexityLevel', 'MainIngredient']

doc_columns = ['score', 'page_content',]

ollama_client = ollama.Client(host='http://localhost:11434',)
model_768 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/LaBSE",
)

model_384 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

model_64 = HuggingFaceEmbeddings(
    model_name="ClovenDoug/tiny_64_all-MiniLM-L6-v2",
)


def convert_to_doc(row):
    """
    Convert a row of the DataFrame to a Document object."""
    doc = Document(
        page_content=f'''
# Recipe Name: {row['TranslatedRecipeName']}
> URL: {row['URL']}

## Ingredients:

{row['TranslatedIngredients']}

## Instructions:

{row['TranslatedInstructions']}
''',
        metadata={
            'TranslatedRecipeName': row['TranslatedRecipeName'],
            'PrepTimeInMins': row['PrepTimeInMins'],
            'CookTimeInMins': row['CookTimeInMins'],
            'TotalTimeInMins': row['TotalTimeInMins'],
            'Servings': row['Servings'],
            'Cuisine': row['Cuisine'],
            'Course': row['Course'],
            'Diet': row['Diet'],
            'ComplexityLevel': row['ComplexityLevel'],
            'MainIngredient': row['MainIngredient'],
        }
    )

    return doc


def generate_metadata(search_query, o_client: ollama.Client):
    """
    Generate metadata filter dictionary for the search query."""
    meta_prompt = f'''
    Given below the user request for queries, create metadata filter dictionary
    for the search.

    user query: {search_query}

    > provide only and only a simple phrase for the user query, do not add any
    > other information or context.
    > this output will be used to filter the recipes.

    available metadata:
    - 'Cuisine': string: ['Indian', 'Kerala Recipes', 'Oriya Recipes',
        'Chinese', 'Konkan', 'Chettinad', 'Mexican', 'Kashmiri',
        'South Indian Recipes', 'North Indian Recipes', 'Andhra',
        'Gujarati Recipes', 'Continental',]
    - 'Diet': string: ['Vegetarian', 'High Protein Vegetarian',
        'Non Vegeterian', 'Eggetarian', 'Diabetic Friendly',
        'Gluten Free', 'Sugar Free Diet', 'No Onion No Garlic (Sattvic)',
        'Vegan', 'High Protein Non Vegetarian',]
    - 'ComplexityLevel': string: ['Medium', 'Hard']

    We can do exact match only.

    respond with a valid json dictionary, do not add any other information or
    context.
    EXPECTED OUTPUT FORMAT: JSON
    '''

    resp = o_client.generate(
        model=OLLAMA_MODEL_NAME,
        prompt=meta_prompt,
        options={
            'temperature': 0.0,
            'max_tokens': 1000,
            # 'stop_sequences': ['```json', '```']
        }
    ).response

    try:
        metadata = json.loads(resp.split(
            'json')[1].strip().split('```')[0].strip())
    except:
        metadata = json.loads(resp.strip().replace("'", '"'))

    return metadata


def rewrite_query(search_query, o_client):
    """
    Rewrite the query to a more search-friendly term."""
    prompt = f'''
    Given below the user request for queries regarding Indian food recipes,
    rephrase and expand the query to a more search friendly term.

    user query: {search_query}

    > provide only and only a simple phrase for the user query, do not add any other information or context.
    > this output will be used to search the database for recipes.
    '''
    resp = o_client.generate(
        model=OLLAMA_MODEL_NAME,
        prompt=prompt,
    ).response
    print(resp)
    return resp


def break_query(search_query, o_client):
    """
    Break down the query into multiple subqueries for better search results."""
    subquery_prompt = f'''
    Given below the user request for queries, break down the query into multiple subqueries.
    user query: {search_query}
    > provide only and only a simple phrase for the user query, do not add any other information or context.
    > this output will be used to search the database for recipes.

    > respond with a valid json array of strings, do not add any other information or context.
    '''

    resp = o_client.generate(
        model=OLLAMA_MODEL_NAME,
        prompt=subquery_prompt,).response
    try:
        subqueries = json.loads(resp.split(
            'json')[1].strip().split('```')[0].strip())
    except:
        subqueries = json.loads(resp.strip().replace("'", '"'))

    return subqueries


def rerank_results(
        search_query,
        searched_df,
        reranking_model
):
    """
    Rerank the results based on the reranking model."""
    if searched_df.empty:
        return searched_df
    new_doc_embeddings = np.array(
        reranking_model.embed_documents(searched_df.page_content)
    )

    query_embedding = np.array(
        reranking_model.embed_query(search_query)
    )

    similarity_scores = cosine_similarity(
        query_embedding.reshape(1, -1),
        new_doc_embeddings
    )
    searched_df['rerank_score'] = similarity_scores[0].tolist()
    return searched_df


def search(
        search_query,
        o_client,
        vector_store,
        reranking_model,
        n_results=10,
        similarity_threshold=0.1,
        flag_rewrite_query=True,
        flag_ai_metadata=True,
        flag_break_query=True,
        flag_rerank_results=True,
):
    """
    Search for the given query in the vector store and return the top n results.
    """
    metadata = {}  # Empty metadata
    subqueries = [search_query]

    if flag_rewrite_query:
        search_query = rewrite_query(search_query, o_client)

    if flag_ai_metadata:
        metadata = generate_metadata(search_query, o_client)
        print(metadata)

    if flag_break_query:
        subqueries = break_query(search_query, o_client)

    ret_docs = []

    for subquery in subqueries:
        ret_docs += vector_store.similarity_search_with_score(
            subquery,
            k=n_results,
            score_threshold=similarity_threshold,
            filter=metadata
        )

    searched_df = pd.DataFrame(
        [
            {
                'score': score,
                **doc.metadata,
                'page_content': doc.page_content,
            } for doc, score in ret_docs
        ],
        columns=doc_columns+columns
    )

    searched_df = searched_df.groupby(
        'TranslatedRecipeName').first().reset_index()
    searched_df['rerank_score'] = searched_df['score']

    if flag_rerank_results:
        searched_df = rerank_results(
            search_query,
            searched_df=searched_df,
            reranking_model=reranking_model,
        ).sort_values(
            'rerank_score',
            ascending=False,
        )

    return searched_df.head(n_results).round(2)[
        [
            'TranslatedRecipeName',
            'page_content',
            'PrepTimeInMins',
            'CookTimeInMins',
            'TotalTimeInMins',
            'Servings',
            'Cuisine',
            'Diet',
            'ComplexityLevel',
            'MainIngredient',
            'score',
            'rerank_score'
        ]
    ]


def as_cards(df):
    """Convert a DataFrame to a list of markdown strings for Discord cards.
    """
    return df.apply(lambda x: x.to_markdown(), axis=1).to_list()


class GenAIBot(commands.Cog):
    """A simple Discord bot cog that captures all messages and provides a
    slash command."""

    def __init__(
        self,
        bot: commands.Bot
    ) -> None:
        super().__init__()
        self.bot = bot
        self._chat_history = {}

        df = pd.read_csv(
            '../IndianFoodDataset.csv',
        ).set_index('Srno')[columns]

        data = df[:].progress_apply(convert_to_doc, axis=1)
        self.vector_store_unchunked = Qdrant.from_documents(
            data,
            model_384,
            collection_name="indian-food-metadata",
            location=':memory:',
            # url="http://localhost:6333",
        )

        # self.vector_store_unchunked = Qdrant(
        #     client=QdrantClient(url='http://localhost:6333'),
        #     collection_name="indian-food-metadata",
        #     embeddings=model_384,
        # )

    @commands.Cog.listener()
    async def on_message(
        self,
        message: nextcord.Message
    ):
        """Capturing All messages"""
        print(message)

        if message.author == self.bot.user or message.author.bot:
            return

    @nextcord.slash_command(
        guild_ids=[config['guild_id']],
        description="Execute Command")
    async def raginator(
            self,
            interaction: nextcord.Interaction,
            user_message: str
    ):
        """A slash command to start ragging."""
        await interaction.response.defer()
        print(interaction.user)
        print(user_message)

        if interaction.user.id not in self._chat_history:
            self._chat_history[interaction.user.id] = []

        chat_messages = self._chat_history[interaction.user.id]

        chat_messages.append(
            {'role': 'user', 'content': user_message}
        )

        chat_history = '\n'.join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in chat_messages
            ]
        )
        user_messages = '\n'.join(
            [
                message['content']
                for message in chat_messages if
                message['role'] == 'user'
            ])
        print(user_messages)
        results = search(
            user_messages,
            # RAG_PROMPT.format(
            #     user_message=user_message,
            #     chat_history=chat_messages,
            # ),
            o_client=ollama_client,
            vector_store=self.vector_store_unchunked,
            reranking_model=model_768,
            n_results=5,
            similarity_threshold=0.1,
            flag_rewrite_query=True,
            flag_ai_metadata=False,
            flag_break_query=True,
            flag_rerank_results=True,
        )

        context = '\n---\n'.join(as_cards(results))

        llm_response = ollama_client.generate(
            model=OLLAMA_MODEL_NAME,
            prompt=PROMPT.format(
                llm_persona=LLM_PERSONA,
                objective_prompt=OBJECTIVE_PROMPT,
                user_message=user_message,
                chat_history=chat_history,
                context=context,
            ),
            stream=False,
        ).response

        chat_messages.append({
            'role': 'assistant',
            'content': llm_response,
        })

        await interaction.followup.send(
            content=llm_response,
            delete_after=300
        )


def setup(bot):
    """Setup function to add the cog to the bot."""
    bot.add_cog(GenAIBot(bot))
