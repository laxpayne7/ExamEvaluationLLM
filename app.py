import streamlit as st

from langchain_openai import AzureChatOpenAI, AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

import os

from dotenv import load_dotenv
load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY")
)

## set up Streamlit 
st.title("Auto Exam Evaluation using OpenAI")
st.write("Please upload the relevant books of the Exam. It's recommended to upload multiple authors.")


endpoint=os.getenv("AZURE_CHAT_OPENAI_ENDPOINT")
key=os.getenv("OPENAI_API_KEY")
llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    model_name="gpt-4o",
    azure_endpoint=endpoint,
    api_version="2023-03-15-preview",
    openai_api_key=key,
)


uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)

if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

    # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents) 

        index_name = os.getenv("PINECONE_INDEX") # put in the name of your pinecone index here

        pc = Pinecone(
        api_key = os.getenv("PINECONE_API_KEY"),
        )

        pc.list_indexes().names() # to check if my index exsist
        index = pc.Index(index_name)
        index.describe_index_stats()

        vectorstore = PineconeVectorStore(pinecone_api_key = os.getenv("PINECONE_API_KEY"), index_name= index_name, embedding=embeddings, index= index)

        vectorstore.add_texts(texts=[t.page_content for t in splits])
        retriever = vectorstore.as_retriever()

        true_answer_prompt = """
        You are a teacher. You will answer the following question. The points that should be present in your answer is present in the context provided. You must strictly follow it. You should not give the answer outside the context. However, You have to elaborate the points with your own understanding. You have to answer the question pointwise.

        Question: {question} 
        Context: {context} 
        Answer:
        """

        true_answer_generator = ChatPromptTemplate.from_messages([
        ("human",true_answer_prompt),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | true_answer_generator
            | llm
            | StrOutputParser()
        )

        question = st.text_input("Enter the question along with the Q No. and the Marks:")
        if question:
            true_answer = rag_chain.invoke(question)
            st.write("Assistant:", true_answer)

            student_answer = st.text_input("Enter the Student's Answer: ")
            if student_answer:
                answer_evaluation_prompt = f"""
                You are a teacher who will evaluate and grade an exam. You are given a question and the student's answer. You have to evaluate the answer with respect to the true answer and give marks as per the relevance to the true answer.

                Question: {question} 
                Student's answer: {student_answer} 
                True Answer: {true_answer}
                Answer:
                """
                evaluation=llm.invoke(answer_evaluation_prompt)
                st.write("Student's Answer Evaluation:", evaluation.content)
