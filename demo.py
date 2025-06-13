from io import BytesIO

import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage

from build_rag import ask, get_complete_answer, load_pretrained

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
        similarity_top_k=10,
        image_similarity_top_k=3,
        vector_store_query_mode='mmr',
        vector_store_kwargs={"mmr_threshold": 0.8}
    )

    return index, vec_retr, llm

def main():

    index, vec_retr, llm = load_artifacts()

    def process_question(question):
        rsp, ctx = ask(question, vec_retr, llm)
        rag_rsp_str, ctx_str, fig = get_complete_answer(rsp, ctx)

        if fig:
            buf = BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)

        if fig is None:
            img = None
        return rag_rsp_str, ctx_str, img


    with gr.Blocks() as demo:
        gr.Markdown("## Enter your question")
        inp = gr.Textbox(label="Your Question")
        btn = gr.Button("Ask")

        with gr.Row():
            response = gr.Textbox(label="Response")
            context = gr.Textbox(label="Context")
            image = gr.Image(label="Context images")

        btn.click(process_question, inputs=inp, outputs=[response, context, image])

    demo.launch(debug=True)


if __name__ == '__main__':
    main()