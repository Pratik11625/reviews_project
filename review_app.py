<<<<<<< HEAD
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from enum import Enum
import json
import time

# Define Sentiment Enum
class Sentiment(str, Enum):
    positive = "POSITIVE"
    negative = "NEGATIVE"
    neutral = "NEUTRAL"

# Define the compressed review model
class CompressedReview(BaseModel):
    sentiment: Sentiment = Field(description="Gives sentiment value of the review")
    theme: list = Field(description="Themes expressed in the reviews")

# Initialize the LLM model
llm = ChatOllama(model="qwen2.5:7b")

# JSON Output Parser
parser_json = JsonOutputParser(pydantic_object=CompressedReview)

# Define the Prompt Template
prompt = PromptTemplate(
    template="Give sentiment and themes for the given input review.\n{format_instructions}\nReview: {input_review}",
    input_variables=["input_review"],
    partial_variables={"format_instructions": parser_json.get_format_instructions()},
)

# Streamlit UI
st.title("Review Sentiment & Theme Analyzer")

review_text = st.text_area("Enter a review:")

def review_txt_file(results_data, txt_file="review.txt"):
        with open(txt_file, 'a', encoding='utf-8') as file:  # Use 'a' to append to the file instead of 'w'
            file.write("\n\n--- New Search Results ---\n")
            file.write(str(results_data) + "\n")

review_txt_file(review_text)

def response_json_file(results_data, txt_file="response_json.txt"):
        with open(txt_file, 'a', encoding='utf-8') as file:  # Use 'a' to append to the file instead of 'w'
            file.write("\n\n--- New Search Results ---\n")
            file.write(str(results_data) + "\n")

if st.button("Analyze Review"):
    with st.spinner("Wait for it..."):
        time.sleep(5)
        if review_text:
            formatted_prompt = prompt.invoke({'input_review': review_text})
            response = llm.invoke(formatted_prompt)
            compressed_review = parser_json.invoke(response)
            
            # Display the output
            st.subheader("Analysis Result:")
            # st.json(json.loads(compressed_review))
            # st.success(response.content)
            st.write(response.content)
            # st.json(response)

            response_json_file(response.content)

        else:
            st.error("Please enter a review before analyzing.")
=======
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from enum import Enum
import json
import time

# Define Sentiment Enum
class Sentiment(str, Enum):
    positive = "POSITIVE"
    negative = "NEGATIVE"
    neutral = "NEUTRAL"

# Define the compressed review model
class CompressedReview(BaseModel):
    sentiment: Sentiment = Field(description="Gives sentiment value of the review")
    theme: list = Field(description="Themes expressed in the reviews")

# Initialize the LLM model
llm = ChatOllama(model="qwen2.5:7b")

# JSON Output Parser
parser_json = JsonOutputParser(pydantic_object=CompressedReview)

# Define the Prompt Template
prompt = PromptTemplate(
    template="Give sentiment and themes for the given input review.\n{format_instructions}\nReview: {input_review}",
    input_variables=["input_review"],
    partial_variables={"format_instructions": parser_json.get_format_instructions()},
)

# Streamlit UI
st.title("Review Sentiment & Theme Analyzer")

review_text = st.text_area("Enter a review:")

if st.button("Analyze Review"):
    with st.spinner("Wait for it..."):
        time.sleep(5)
        if review_text:
            formatted_prompt = prompt.invoke({'input_review': review_text})
            response = llm.invoke(formatted_prompt)
            compressed_review = parser_json.invoke(response)
            
            # Display the output
            st.subheader("Analysis Result:")
            # st.json(json.loads(compressed_review))
            # st.success(response.content)
            st.write(response.content)
            # st.json(response)

        else:
            st.error("Please enter a review before analyzing.")
>>>>>>> 61a6151014dda399bc0bb2469592a6f16a5e0b27
