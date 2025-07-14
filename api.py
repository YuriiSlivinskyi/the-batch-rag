from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from build_rag import ask, get_complete_answer, load_pretrained, get_input_files, system_template
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    response: str
    context: str
    image_base64: Optional[str] = None

def load_artifacts():
    clip_embedding, text_embeddings, llm = load_pretrained()
    Settings.llm = llm
    Settings.embed_model = text_embeddings
    Settings.image_embed_model = clip_embedding
    storage_context = StorageContext.from_defaults(persist_dir=".index")
    index = load_index_from_storage(storage_context,
                                    llm=llm,
                                    embed_model=text_embeddings,
                                    image_embed_model=clip_embedding)
    vec_retr = index.as_retriever(
        similarity_top_k=5,
        image_similarity_top_k=3,
        vector_store_query_mode='mmr',
        vector_store_kwargs={"mmr_threshold": 0.8}
    )
    input_files = get_input_files('/.data')
    documents = SimpleDirectoryReader(input_files=input_files).load_data()
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    bm25_retr = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=5
    )
    hybrid_retr = QueryFusionRetriever(
        retrievers=[vec_retr, bm25_retr],
        similarity_top_k=5,
        mode="reciprocal_rerank"
    )
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=5
    )
    chat_tpl = ChatPromptTemplate(
        message_templates=[
            ChatMessage(role="system", content=system_template),
            ChatMessage(role="system", content="Context:\n{context_str}"),
            ChatMessage(role="user", content="{query_str}")
        ]
    )
    response_synth = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=chat_tpl
    )
    adv_engine = RetrieverQueryEngine(
        retriever=hybrid_retr,
        node_postprocessors=[reranker],
        response_synthesizer=response_synth,
    )
    return adv_engine

engine = load_artifacts()

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        rsp, ctx = ask(request.question, engine)
        rag_rsp_str, ctx_str, fig = get_complete_answer(rsp, ctx)
        img_b64 = None
        if fig:
            buf = BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return AnswerResponse(response=rag_rsp_str, context=ctx_str, image_base64=img_b64)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 