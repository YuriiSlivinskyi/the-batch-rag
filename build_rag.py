import os

import matplotlib.pyplot as plt
import nest_asyncio
from PIL import Image
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    generate_question_context_pairs,
)
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.llms import ImageBlock
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.retrievers.bm25 import BM25Retriever

system_template = """
"You are an expert AI assistant specializing in artificial intelligence and machine learning, dedicated to providing insightful and comprehensive answers to user queries. 
Your primary source of information is the provided context, which consists of articles and potentially images from DeepLearning.AI's 'The Batch' newsletter.

Instructions:
Strictly adhere to the provided context: Do not use any prior internal knowledge, external sources, or make assumptions. All information in your answer must be directly supported by the provided text and images.
Provide complete and detailed answers: Strive for thoroughness. Explain concepts, summarize findings, and elaborate on details as presented in the context. Avoid short, superficial, or overly general responses. Imagine you are explaining the topic to someone who needs a clear and comprehensive understanding based solely on the provided materials.
If images are provided in the context, carefully analyze them and incorporate relevant visual information into your textual answer where appropriate to enrich the explanation.
Maintain a helpful and informative tone.
If, after careful review, you determine that the provided context does not contain the information necessary to answer the user's question, or if the context is entirely irrelevant to the query, respond with: "I cannot assist you with this question as the provided context does not contain relevant information."
Your goal is to deliver highly accurate, detailed, and contextually grounded answers that reflect the depth and breadth of information available in 'The Batch' articles provided."
"""


def get_complete_answer(rsp):
    rag_rsp_str = rsp.response
    ctx_str = ''
    for node in rsp.metadata['text_nodes']:
        ctx_piece = f"""Article name: {node.metadata['file_name'].split('.')[0]}
            Relevant content: {node.text}

            """
        ctx_str += ctx_piece

    relevant_images = []
    img_nodes = []
    for node in rsp.metadata["image_nodes"]:
        relevant_images.append(node.metadata['file_path'])
        img_nodes.append(node)

    fig = None
    axs = None
    if len(relevant_images) > 0:
        if len(relevant_images) == 1:
            img = Image.open(relevant_images[0])
            fig = plt.imshow(img)
            fig = fig.get_figure()
            ax = fig.get_axes()[0]
            ax.axis('off')
        if len(relevant_images) > 1:
            if len(relevant_images) <= 4:
                fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            if len(relevant_images) > 4:
                fig, axs = plt.subplots(3, 3, figsize=(14, 16))

            axs = axs.flatten()
            for id, path in enumerate(relevant_images):
                img = Image.open(path)
                axs[id].imshow(img)
                axs[id].axis('off')
                axs[id].set_title(img_nodes[id].metadata['file_name'].split('.')[0])

                if id == 8:
                    break

            for ax in axs:
                ax.axis('off')

    return rag_rsp_str, ctx_str, fig


def load_pretrained():
    load_dotenv()
    clip_embedding = ClipEmbedding(model_name="ViT-B/32")
    text_embeddings = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-large"
    )

    llm = GoogleGenAI(
        model="models/gemini-2.5-flash-preview-05-20"
    )
    return clip_embedding, text_embeddings, llm


def get_input_files(directory_path):
    all_files = []
    if not os.path.isdir(directory_path):
        return all_files

    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            full_filepath = os.path.join(root, filename)
            all_files.append(full_filepath)
    return all_files


def ask(question: str, engine):
    rsp = engine.query(question)
    context = rsp.source_nodes

    text_ctx = []
    text_nodes = []
    img_ctx = []
    img_nodes = []

    for node in context:
        if 'text' in node.metadata['file_type']:
            if node.score >= .5:
                text_ctx.append(node.text)
                text_nodes.append(node)
        if 'image' in node.metadata['file_type']:
            if node.score is not None:
                img_ctx.append(ImageBlock(path=node.metadata['file_path']))
                img_nodes.append(node)

    rsp.metadata['text_nodes'] = text_nodes
    rsp.metadata['image_nodes'] = img_nodes

    return rsp, context


def generate_custom_qa_dataset(documents, llm, num_questions=15):
    from llama_index.core.evaluation.dataset_generation import DatasetGenerator
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    generator = DatasetGenerator(nodes, llm=llm, num_questions_per_chunk=5)
    qa_dataset = generator.generate_dataset_from_nodes(num=num_questions)
    pairs = []
    for idx, (q, _) in enumerate(qa_dataset.qr_pairs):
        node = nodes[idx % len(nodes)]
        file_name = node.metadata.get('file_name', 'unknown')
        pairs.append((q, file_name, node.text))
    return pairs


def main():
    clip_embedding, text_embeddings, llm = load_pretrained()

    Settings.llm = llm
    Settings.embed_model = text_embeddings
    Settings.image_embed_model = clip_embedding

    input_files = get_input_files('.data')
    text_files = [f for f in input_files if f.endswith('.txt')]
    documents = SimpleDirectoryReader(input_files=text_files).load_data()

    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    index = MultiModalVectorStoreIndex(nodes,
                                       embed_model=text_embeddings,
                                       image_embed_model=clip_embedding)

    index.storage_context.persist(persist_dir=".index")

    vec_retr = index.as_retriever(
        similarity_top_k=5,
        image_similarity_top_k=3,
        vector_store_query_mode='mmr',
        vector_store_kwargs={"mmr_threshold": 0.8}
    )

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

    nest_asyncio.apply()

    qa_pairs = generate_custom_qa_dataset(documents, llm, num_questions=15)
    faithfulness_evaluator = FaithfulnessEvaluator()
    relevancy_evaluator = RelevancyEvaluator()
    answer_relevancy_evaluator = AnswerRelevancyEvaluator(llm=llm)
    context_relevancy_evaluator = ContextRelevancyEvaluator(llm=llm)
    correctness_evaluator = CorrectnessEvaluator(llm=llm)
    semantic_similarity_evaluator = SemanticSimilarityEvaluator(embed_model=text_embeddings)
    print("\nEvaluating...")
    for idx, (question, file_name, reference_text) in enumerate(qa_pairs):
        print(f"\n=== QA Pair {idx+1} ===")
        print(f"Question: {question}")
        print(f"File: {file_name}")
        rsp, _ = ask(question, adv_engine)
        answer = rsp.response
        faith = faithfulness_evaluator.evaluate_response(response=rsp, query=question)
        print(f"Faithfulness: score={faith.score}, passing={faith.passing}")
        rel = relevancy_evaluator.evaluate_response(response=rsp, query=question)
        print(f"Relevancy: score={rel.score}, passing={rel.passing}")
        ans_rel = answer_relevancy_evaluator.evaluate_response(query=question, response=answer)
        print(f"AnswerRelevancy: score={ans_rel.score}, feedback={ans_rel.feedback}")
        ctx_rel = context_relevancy_evaluator.evaluate_response(query=question, contexts=[reference_text])
        print(f"ContextRelevancy: score={ctx_rel.score}, feedback={ctx_rel.feedback}")
        corr = correctness_evaluator.evaluate_response(query=question, response=answer, reference=reference_text)
        print(f"Correctness: score={corr.score}, feedback={corr.feedback}")
        sem = semantic_similarity_evaluator.evaluate_response(response=answer, reference=reference_text)
        print(f"SemanticSimilarity: score={sem.score}, passing={sem.passing}")


if __name__ == '__main__':
    main()
