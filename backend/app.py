#app.py
import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

DB_FAISS_PATH = 'vectorstores/db_faiss'
MODEL_PATH = 'TheBloke/Llama-2-7B-Chat-GGML'
QA_CHAIN_PATH = 'qa_chain.pkl'

app = Flask(__name__)
CORS(app)  # Enabling CORS for all routes

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If and only if you don't know the answer atleast little bit also then, just say that you don't know, don't try to make up answer.Generally do not use she or he, use you.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

class QueryModel(BaseModel):
    query: str

initialized = False
qa_chain = None

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def retrieval_qa_chain(llm, prompt, db):
    print("Initializing QA Chain")
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 1}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def initialize():
    global initialized, qa_chain
    if not initialized:
        print("Initializing System")
        start_time = time.time()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        llm = CTransformers(
            model=MODEL_PATH,
            model_type="llama",
            max_new_tokens=1056,
            temperature=0.5
        )
        qa_prompt = set_custom_prompt()
        qa_chain = retrieval_qa_chain(llm, qa_prompt, db)
        initialized = True
        end_time = time.time()
        print(f"Initialization completed in {end_time - start_time:.2f} seconds")


@app.route("/query", methods=["POST"])
def query_endpoint():
    global initialized, qa_chain
    if not initialized:
        initialize()

    try:
        data = request.json
        query_model = QueryModel(**data)
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": e.errors()}), 400

    print(f"Received query: {query_model.query}")
    start_time = time.time()
    response = qa_chain.invoke({'query': query_model.query})
    end_time = time.time()
    print(f"Response generation took {end_time - start_time:.2f} seconds")
    
    return jsonify({"response": response['result']})

if __name__ == '__main__':
    initialize()  # here Ensuring the system is initialized before handling requests
    app.run(debug=False)  # Running in debug mode for development
