## Initialization
1. Clone repository using 
``git clone https://github.com/YuriiSlivinskyi/the-batch-rag.git``
2. Initialize virtual environment using \
``python -m venv .venv``
3. Install all dependencies using 
``pip install -r requirements.txt``

## Data Scrapping

All the data is saved in *.data* directory and sorted by article categories
To run the scraping use
``python scrapper.py``

## Building index

The index is stored in *.index* folder in default llamaindex format.
\
To run indexing use ``python build_rag.py``

## Demo

Demo is build using Gradio. It creates a web server to which you can connect from your browser.
To start the demo use
``python demo.py``. Link to Web UI will be shared in terminal.