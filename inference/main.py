import os
import json
import argparse
import gradio as gr

from openai import api_key
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms import openai
from elasticsearch import AsyncElasticsearch

api_key = os.getenv("OPENAI_API_KEY")
class AI:
    def __init__(self, es_client_args, es_store_args, server_port, model="gpt-3.5-turbo-0125", temperature=0.4):
        """
        Initialize the AI Assistant with the specified model and Elasticsearch configuration.

        :param es_client_args: Arguments for configuring the Elasticsearch client.
        :param es_store_args: Arguments for configuring the Elasticsearch store.
        :param server_port: The OpenAI model to be used.
        :param model: The OpenAI model to be used.
        :param temperature: The temperature setting for the LLM.
        """
        llm = openai.OpenAI(
            temperature=float(temperature),
            model=model,
        )
        self.server_port = server_port
        self.es_client = AsyncElasticsearch(**es_client_args)
        self.es_vector_store = ElasticsearchStore(
            es_client=self.es_client, **es_store_args)

        Settings.llm = llm
        self.vector_store_index = VectorStoreIndex.from_vector_store(
            self.es_vector_store)
        self.query_engine = self.vector_store_index.as_chat_engine()

    def predict(self, input_text, _):
        response = self.query_engine.chat(input_text)
        return str(response)

    def launch_interface(self, description="AI Assistant powered by OpenAI and Elasticsearch", share=True):
        """
        Launch the Gradio chat interface.

        :param description: Description of the interface.
        :param share: Whether to generate a shareable link.
        """
        interface = gr.ChatInterface(
            self.predict,
            description=description
        )
        interface.launch(share=False, server_port=self.server_port)


def main(args):
    ai = AI(
        es_client_args=json.loads(args.es_client_args),
        es_store_args=json.loads(args.es_store_args),
        model=args.model,
        temperature=args.temperature,
        server_port=args.server_port
    )
    ai.launch_interface()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the AI Assistant with specified Elasticsearch and OpenAI configuration."
    )

    parser.add_argument(
        "--es_client_args",
        type=str,
        help="Arguments for configuring the Elasticsearch client.",
        required=True
    )

    parser.add_argument(
        "--es_store_args",
        type=str,
        help="Arguments for configuring the Elasticsearch store.",
        required=True
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-0125",
        help="The OpenAI model to be used."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="The temperature setting for the language model."
    )

    parser.add_argument(
        "--server_port",
        type=int,
        default=8080,
        help="Gradio Server Port."
    )

    args = parser.parse_args()

    main(args)
