"""Global options for analysis
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Set


# sys.path.append("..")
# Hardware options
N_CORES: int = 32  # max number of CPU cores to use
RAM_CORENLP: str = "16G"  # max RAM allocated for parsing using CoreNLP; increase to speed up parsing
PARSE_CHUNK_SIZE: int = 1000  # number of lines in the input file to process using CoreNLP at once. # Increase on workstations with larger RAM (e.g. to 1000 if RAM is 64G)

# Directory locations
os.environ[
    "CORENLP_HOME"
] = "/Models/stanford-corenlp-full-2018-10-05"

# location of the CoreNLP models; use / to separate folders
DATA_FOLDER: str = "Data/"
MODEL_FOLDER: str = "Models/"  # will be created if it does not exist
OUTPUT_FOLDER: str = "Outputs/"  # will be created if it does not exist; !!! WARNING: existing files will be removed !!!
UTILS_FOLDER: str = "Utils/"

# Parsing and analysis options
STOPWORDS: Set[str] = set(
    Path("Data/", "StopWords_Generic.txt").read_text().lower().split()
)  # Costume to this or other stopwords dictionary if necessary. Set of stopwords from https://https://sraf.nd.edu/textual-analysis/stopwords/ with Stopwords_Generic
PHRASE_THRESHOLD: float = 0.55  # threshold of the phrase module (smaller -> more phrases)
PHRASE_MIN_COUNT: int = 10  # min number of times a bi-gram needs to appear in the corpus to be considered as a phrase
W2V_DIM: int = 300  # dimension of word2vec vectors
W2V_WINDOW: int = 5  # window size in word2vec
W2V_ITER: int = 20  # number of iterations in word2vec
W2V_SKIP: bool = False
LDA_TOPIC_NUMBER: int = 5  # number of topics to be extracted by the lda model
N_WORDS_DIM: int = 500  # max number of words in each dimension of the dictionary
DICT_RESTRICT_VOCAB = None # change to a fraction number (e.g. 0.2) to restrict the dictionary vocab in the top 20% of most frequent vocab

# Inputs for constructing the expanded dictionary
DIMS: List[str] = ["Urgency"]
SEED_WORDS: Dict[str, List[str]] = {
    "Urgency": [
        "immediate", "quick_sale", "motivated"
        # expanded:
"quick_sale", "motivated", "immediate", "drastically_reduce", "relocation", "slash", "drastic_price_reduction",
"motivated_seller", "seller_motivated", "short_sale_foreclosure", "foreclosure", "aggressively", "relocate"
"aggressive", "short_sale_reo", "quickly", "quick_closing", "bank_foreclosure", "act_fast", "repo", "price_reduce"
"quick_response"
    ]
}

# Create directories if not exist
# Path(DATA_FOLDER, "processed", "parsed").mkdir(parents=True, exist_ok=True)
# Path(DATA_FOLDER, "processed", "unigram").mkdir(parents=True, exist_ok=True)
# Path(DATA_FOLDER, "processed", "bigram").mkdir(parents=True, exist_ok=True)
# Path(DATA_FOLDER, "processed", "trigram").mkdir(parents=True, exist_ok=True)
# Path(MODEL_FOLDER, "phrases").mkdir(parents=True, exist_ok=True)
# Path(MODEL_FOLDER, "phrases").mkdir(parents=True, exist_ok=True)
# Path(MODEL_FOLDER, "w2v").mkdir(parents=True, exist_ok=True)
# Path(OUTPUT_FOLDER, "dict").mkdir(parents=True, exist_ok=True)
# Path(OUTPUT_FOLDER, "scores").mkdir(parents=True, exist_ok=True)
# Path(OUTPUT_FOLDER, "scores", "temp").mkdir(parents=True, exist_ok=True)
# Path(OUTPUT_FOLDER, "scores", "word_contributions").mkdir(parents=True, exist_ok=True)
# Path(UTILS_FOLDER).mkdir(parents=True, exist_ok=True)

