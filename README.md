## Initialization
1. Clone repository using 
``git clone https://github.com/YuriiSlivinskyi/the-batch-rag.git``
2. Initialize virtual environment using \
``python -m venv .venv``
3. Install all dependencies using 
``pip install -r requirements.txt``
4. Create .env file with **GOOGLE_API_KEY** value

## Data Scrapping

All the data is saved in *.data* directory and sorted by article categories
To run the scraping use
``python scrapper.py``

## Building index

The index is stored in *.index* folder in default llamaindex format.
\
After building the RAG evaluation of faithfulness and relevance is made. 
\
To run indexing use ``python build_rag.py`` \
**Note**: evaluation dataset is created only on 1 document to preserve token quota.

## Demo

Demo is build using Gradio. It creates a web server to which you can connect from your browser.
To start the demo use
``python demo.py``. Link to Web UI will be shared in terminal.