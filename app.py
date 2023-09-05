import os
from typing import Any
from dotenv import load_dotenv
import gradio as gr
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains.question_answering import load_qa_chain 
from langchain.chains import ConversationalRetrievalChain,RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter

import fitz
from PIL import Image

import chromadb
from langchain.vectorstores import FAISS
import re
import uuid 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

FAISS_INDEX_PATH = "faiss_index"
handler = StreamingStdOutCallbackHandler()

load_dotenv(".env")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


enable_box = gr.Textbox.update(value = None, placeholder = 'Upload your OpenAI API key',interactive = True)
disable_box = gr.Textbox.update(value = 'OpenAI API key is Set', interactive = False)

def set_apikey(api_key: str):
        app.OPENAI_API_KEY = api_key        
        return disable_box
    
def enable_api_box():
        return enable_box

def add_text(history, text: str):
    if not text:
         raise gr.Error('enter text')
    history = history + [(text,'')] 
    return history

class my_app:
    def __init__(self, OPENAI_API_KEY: str = None ) -> None:
        self.OPENAI_API_KEY: str = OPENAI_API_KEY
        self.chain = None
        self.chat_history: list = []
        self.N: int = 0
        self.count: int = 0

    def __call__(self, file: str) -> Any:
        if self.count==0:
            self.chain = self.build_chain(file)
            self.count+=1
        return self.chain
    
    def chroma_client(self):
        #create a chroma client
        client = chromadb.Client()
        #create a collecyion
        collection = client.get_or_create_collection(name="my-collection")
        return client
    
    def process_file(self,file: str):

        # loader = PyPDFLoader(file.name)
        # documents = loader.load()  
        
        # pattern = r"/([^/]+)$"
        # match = re.search(pattern, file.name)
        # file_name = match.group(1)

        # print("Splitting all documents")

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000,
        #     chunk_overlap=100,
        #     length_function=len,
        # )
        # split_docs = text_splitter.split_documents(documents)
        # return split_docs, file_name


        reader = PdfReader(file.name)
        raw_text = ''

        for i, page in enumerate(reader.pages):
             text =page.extract_text()
             if text:
                raw_text+=text  


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        split_text = text_splitter.split_text(raw_text)
        return split_text, file.name


    def build_chain(self, file: str):
        documents, file_name = self.process_file(file)
        #Load embeddings model
        embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY) 

        # pdfsearch = Chroma.from_documents(documents, embeddings, collection_name= file_name,)
        pdfsearch = FAISS.from_texts(documents, embedding=embeddings)
        pdfsearch.save_local(FAISS_INDEX_PATH)
        
        retriever = pdfsearch.as_retriever()        
        # retriever.search_kwargs['distance_metric'] = 'cos'
        retriever.search_kwargs['fetch_k'] = 100
        # retriever.search_kwargs['maximal_marginal_relevance'] = True
        retriever.search_kwargs['k'] = 100

        # chain = ConversationalRetrievalChain.from_llm(
        #         ChatOpenAI(temperature=0.0, openai_api_key=self.OPENAI_API_KEY), 
        #         retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
        #         return_source_documents=True,max_tokens_limit=1000)


        # chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", 
        #                                                     retriever=retriever,                                                            
        #                                                     return_source_documents=True,verbose=True
        #                                                     ,max_tokens_limit=1000
        #                                                     )


        # model = ChatOpenAI(model_name="gpt-3.5-turbo",
        #                 streaming=True,
        #                 callbacks=[handler],
        #                 verbose=True,
        #                 max_tokens=1000,)  # switch to 'gpt-4'
        
        # chain = ConversationalRetrievalChain.from_llm(model, retriever=retriever,max_tokens_limit=1000)
        
        chain = load_qa_chain(OpenAI(),chain_type="stuff")
        return chain,pdfsearch
    

def get_response(history, query, file): 
        if not file:
            raise gr.Error(message='Upload a PDF')    
        chain,pdfsearch = app(file)

        # docs = pdfsearch.similarity_search_with_score(query)
        
        # for doc, score in docs:
        #         print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

        docs = pdfsearch.similarity_search(query)
        

        result = chain({"input_documents": docs,"question": query, 'chat_history':app.chat_history},return_only_outputs=True)
        app.chat_history += [(query, result["output_text"])]
        # app.N = list(result['source_documents'][0])[1][1]['page']
        for char in result['output_text']:
           history[-1][-1] += char
           yield history,''

def render_file(file):
        doc = fitz.open(file.name)
        page = doc[app.N]
        #Render the page as a PNG image with a resolution of 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image

def render_first(file):
        doc = fitz.open(file.name)
        page = doc[0]
        #Render the page as a PNG image with a resolution of 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image,[]

app = my_app()
with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=0.8):
                api_key = gr.Textbox(placeholder='Enter OpenAI API key', show_label=False, interactive=True).style(container=False)
            with gr.Column(scale=0.2):
                change_api_key = gr.Button('Change Key')
        with gr.Row():           
            chatbot = gr.Chatbot(value=[], elem_id='chatbot').style(height=650)
            show_img = gr.Image(label='Upload PDF', tool='select' ).style(height=680)
    with gr.Row():
        with gr.Column(scale=0.60):
            txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter",
                    ).style(container=False)
        with gr.Column(scale=0.20):
            submit_btn = gr.Button('submit')
        with gr.Column(scale=0.20):
            btn = gr.UploadButton("📁 upload a PDF", file_types=[".pdf"]).style()
        
    api_key.submit(
            fn=set_apikey, 
            inputs=[api_key], 
            outputs=[api_key,])
    change_api_key.click(
            fn= enable_api_box,
            outputs=[api_key])
    btn.upload(
            fn=render_first, 
            inputs=[btn], 
            outputs=[show_img,chatbot],)
    
    submit_btn.click(
            fn=add_text, 
            inputs=[chatbot,txt], 
            outputs=[chatbot, ], 
            queue=False).success(
            fn=get_response,
            inputs = [chatbot, txt, btn],
            outputs = [chatbot,txt]).success(
            fn=render_file,
            inputs = [btn], 
            outputs=[show_img]
    )

    
demo.queue()
demo.launch()  
