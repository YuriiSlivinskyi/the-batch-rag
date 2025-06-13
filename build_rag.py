import os

import matplotlib.pyplot as plt
import nest_asyncio
from PIL import Image
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.base.response.schema import Response
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation import RelevancyEvaluator
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.llms import ChatMessage, ImageBlock
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, ImageNode
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

system_template = """
You are asistant providing answers and information to users. Use information only from provided context which includes text and possibly images, do not use any internal knowledge.
Give complete and detailed answers.
If context isn't relevant or if you can't assist user answer using "I can not assist you with this question."
"""


def get_complete_answer(rsp, ctx):
    rag_rsp_str = rsp.response
    ctx_str = ''
    for node in ctx:
        if isinstance(node.node, TextNode):
            ctx_piece = f"""Article name: {node.metadata['file_name'].split('.')[0]}
            Relevant content: {node.text}

            """
            ctx_str += ctx_piece

    relevant_images = []
    img_nodes = []
    for node in ctx:
        if isinstance(node.node, ImageNode):
            relevant_images.append(node.metadata['file_path'])
            img_nodes.append(node)

    fig = None
    if len(relevant_images) > 0:
        if len(relevant_images) == 1:
            img = Image.open(relevant_images[0])
            fig = plt.imshow(img)
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


def ask(question: str, vec_retr, llm):
    raw_context = vec_retr.retrieve(question)
    context = []
    for node in raw_context:
        if node.score >= .7:
            context.append(node)
    text_ctx = []
    img_ctx = []
    img_nodes = []

    for node in context:
        if 'text' in node.metadata['file_type']:
            text_ctx.append(node.text)
        if 'image' in node.metadata['file_type']:
            img_ctx.append(ImageBlock(path=node.metadata['file_path']))
            img_nodes.append(node)

    text_ctx_str = '\n'.join(text_ctx)

    sys_msg = ChatMessage(role='system', content=system_template)
    ctx_msg = ChatMessage(role="system", content=f"Context:\n{text_ctx_str}")
    if img_ctx:
        for img in img_ctx:
            ctx_msg.blocks.append(img)
    usr_msg = ChatMessage(role="user", content=f"{question}")
    answer = llm.chat(messages=[sys_msg, ctx_msg, usr_msg])
    response = Response(
        response=str('\n'.join([block.text for block in answer.message.blocks])),
        source_nodes=context,
        metadata={
            "text_nodes": text_ctx,
            "image_nodes": img_nodes,
        },
    )
    return response, context


def main():
    clip_embedding, text_embeddings, llm = load_pretrained()

    Settings.llm = llm
    Settings.embed_model = text_embeddings
    Settings.image_embed_model = clip_embedding

    input_files = get_input_files('.data')

    documents = SimpleDirectoryReader(input_files=input_files).load_data()

    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    index = MultiModalVectorStoreIndex(nodes,
                                       embed_model=text_embeddings,
                                       image_embed_model=clip_embedding,
                                       similarity_top_k=10)

    index.storage_context.persist(persist_dir=".index")

    vec_retr = index.as_retriever(
        similarity_tok_k=10,
        image_similarity_top_k=3,
        mmr=True,
        mmr_diversity_bias=0.5,
    )

    nest_asyncio.apply()

    eval_llm = GoogleGenAI(
        model="gemini-2.0-flash-lite"
    )

    qa_dataset = generate_question_context_pairs(
        [documents[0]],
        llm=eval_llm,
        num_questions_per_chunk=2
    )

    test_query = "Provide information about how deepfakes can hurt people"
    print('Using testing query to evaluate perfomance, QA dataset created, but not used to preserve token LLM quota')

    faithfulness_evaluator = FaithfulnessEvaluator()
    relevancy_eval = RelevancyEvaluator()

    def measure_faitfulness(query, vec_retr, llm):
        r, c = ask(query, vec_retr, llm)
        eval_result = faithfulness_evaluator.evaluate_response(response=r, query=query)
        return eval_result

    def measure_relevancy(query, vec_retr, llm):
        r, c = ask(query, vec_retr, llm)
        eval_result = relevancy_eval.evaluate_response(response=r, query=query)
        return eval_result

    print('Testing query:\n', test_query)
    rsp, ctx = ask(test_query, vec_retr, llm)
    rsp_str, ctx_str, _ = get_complete_answer(rsp, ctx)

    print("RAG's response:\n", rsp_str)
    print("Text context:\n", ctx_str)

    faitfulness = measure_faitfulness(test_query, vec_retr, llm)
    print("Faithfulness:\t", faitfulness.score, faitfulness.passing)

    relevancy = measure_relevancy(test_query, vec_retr, llm)
    print("Relevancy:\t", relevancy.score, relevancy.passing)
    return


if __name__ == '__main__':
    main()
