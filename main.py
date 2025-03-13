import datetime
import functools
import logging
import sys
from pathlib import Path
import clean
import parse
from creat_dictionary import creat_dict
import global_options
from Utils import train_models_untils, multiple_word_detect, file_process
from score import run_scoring_pipeline

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#%%
# Initial clean for following parsing work
clean.clean_file(
    input_path= Path(
global_options.DATA_FOLDER, "Input", "documents.txt"
    ),
    output_path= Path(
global_options.DATA_FOLDER, "Processed", "cleaned", "documents.txt"
    ),
    to_lower=True,
    remove_punc=True,
)
#%%
# Parsing
parse.parse_document(
        input_path=Path(
            global_options.DATA_FOLDER, "Processed", "cleaned", "documents.txt"
        ),
        input_id=file_process.file_to_list(Path(
            global_options.DATA_FOLDER, "Input", "document_ids.txt"
        )),
        output_path=Path(
            global_options.DATA_FOLDER, "processed", "parsed", "documents.txt"
        ),
        output_id=Path(
            global_options.DATA_FOLDER, "processed", "parsed", "document_sent_ids.txt"
        ),
        lemma=True
    )
#%%
# Final clean(e.g. remove punctuation and ner/pos tags)
clean.clean_file(
    input_path= Path(
global_options.DATA_FOLDER, "processed", "parsed", "documents.txt"
    ),
    output_path= Path(
global_options.DATA_FOLDER, "Processed", "unigram", "documents.txt"
    ),
    to_lower=True,
    remove_num=True,
    remove_punc=True,
    remove_stop=True,
    remove_single=True,
    custom_stop=None,
    language='english'
)

#%%
# train and apply a phrase model to detect 2-word phrases ----------------
multiple_word_detect.train_bigram_model(
    input_path=Path(
        global_options.DATA_FOLDER, "Processed", "unigram", "documents.txt"
    ),
    model_path=Path(
        global_options.MODEL_FOLDER, "phrases", "bigram.mod"
    ),
)

multiple_word_detect.file_bigramer(
    input_path=Path(
        global_options.DATA_FOLDER, "Processed", "unigram", "documents.txt"
    ),
    output_path=Path(
        global_options.DATA_FOLDER, "Processed", "bigram", "documents.txt"
    ),
    model_path=Path(
        global_options.MODEL_FOLDER, "phrases", "bigram.mod"
    ),
    scoring="npmi_scorer",
    threshold=global_options.PHRASE_THRESHOLD,
)

# train and apply a phrase model to detect 3-word phrases ----------------
multiple_word_detect.train_bigram_model(
    input_path=Path(
        global_options.DATA_FOLDER, "Processed", "bigram", "documents.txt"
    ),
    model_path=Path(
        global_options.MODEL_FOLDER, "phrases", "trigram.mod"
    ),
)

multiple_word_detect.file_bigramer(
    input_path=Path(
        global_options.DATA_FOLDER, "Processed", "bigram", "documents.txt"
    ),
    output_path=Path(
        global_options.DATA_FOLDER, "Processed", "trigram", "documents.txt"
    ),
    model_path=Path(global_options.MODEL_FOLDER, "phrases", "trigram.mod"),
    scoring="npmi_scorer",
    threshold=global_options.PHRASE_THRESHOLD,
)
#%%
# train the word2vec model ----------------
print(datetime.datetime.now())
print("Training w2v model...")
train_models_untils.train_w2v_model(
    input_path=Path(
        global_options.DATA_FOLDER, "processed", "trigram", "documents.txt"
    ),
    model_path=Path(global_options.MODEL_FOLDER, "w2v", "w2v.mod"),
    vector_size=global_options.W2V_DIM,
    window=global_options.W2V_WINDOW,
    workers=global_options.N_CORES,
    epochs=global_options.W2V_ITER,
    sg=global_options.W2V_SKIP
)
#%%
# expand the seed words to dictionary using trained w2v model
print(datetime.datetime.now())
print("Expanding dictionary...")
creat_dict(
    input_path=Path(global_options.MODEL_FOLDER, "w2v", "w2v.mod"),
    output_path = Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv")
)
#%%
# scoring the remarks based on term frequency(or TFIDF, WFIDF...)
print(datetime.datetime.now())
print("Scoring samples...")
run_scoring_pipeline(
    dict_path=Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv"),
    corpus_path=Path(global_options.DATA_FOLDER, "processed", "trigram", "documents.txt"),
    id_path=Path(global_options.DATA_FOLDER, "processed", "parsed", "document_sent_ids.txt"),
    methods=['TF']
)