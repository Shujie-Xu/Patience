import itertools
import os
import pickle
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

import pandas as pd
from tqdm import tqdm as tqdm

import global_options
from Utils import dictionary, file_process


# @TODO: The scoring functions are not memory friendly. The entire processed corpus needs to fit in the RAM. Rewrite a memory friendly version.


def construct_doc_level_corpus(sent_corpus_file, sent_id_file):
    """Construct document level corpus from sentence level corpus and write to disk.
    Dump "corpus_doc_level.pickle" and "doc_ids.pickle" to Path(global_options.OUTPUT_FOLDER, "scores", "temp"). 

    Arguments:
        sent_corpus_file {str or Path} -- The sentence corpus after parsing and cleaning, each line is a sentence
        sent_id_file {str or Path} -- The sentence ID file, each line correspond to a line in the sent_co(docID_sentenceID)

    Returns:
        [str], [str], int -- a tuple of a list of documents, a list of document IDs, and the number of documents
    """
    print("Constructing doc level corpus")
    # sentence level corpus
    sent_corpus = file_process.file_to_list(sent_corpus_file)
    sent_IDs = file_process.file_to_list(sent_id_file)
    assert len(sent_IDs) == len(sent_corpus)
    # doc id for each sentence
    doc_ids = [x.split("_")[0] for x in sent_IDs]
    # concat all text from the same doc
    id_doc_dict = defaultdict(lambda: "")
    for i, id in enumerate(doc_ids):
        id_doc_dict[id] += " " + sent_corpus[i]
    # create doc level corpus
    corpus = list(id_doc_dict.values())
    doc_ids = list(id_doc_dict.keys())
    assert len(corpus) == len(doc_ids)
    with open(
            Path(global_options.OUTPUT_FOLDER, "scores", "temp", "corpus_doc_level.pickle"),
            "wb",
    ) as out_f:
        pickle.dump(corpus, out_f)
    with open(
            Path(global_options.OUTPUT_FOLDER, "scores", "temp", "doc_ids.pickle"), "wb"
    ) as out_f:
        pickle.dump(doc_ids, out_f)
    N_doc = len(corpus)
    return corpus, doc_ids, N_doc


def calculate_df(corpus):
    """Calculate and dump a document-freq dict for all the words.

    Arguments:
        corpus {[str]} -- a list of documents

    Returns:
        {dict[str: int]} -- document freq for each word
    """
    print("Calculating document frequencies.")
    # document frequency
    df_dict = defaultdict(int)
    for doc in tqdm(corpus):
        doc_splited = doc.split()
        words_in_doc = set(doc_splited)
        for word in words_in_doc:
            df_dict[word] += 1
    # save df dict
    with open(
            Path(global_options.OUTPUT_FOLDER, "scores", "temp", "doc_freq.pickle"), "wb"
    ) as f:
        pickle.dump(df_dict, f)
    return df_dict


def load_doc_level_corpus():
    """load the corpus constructed by construct_doc_level_corpus()

    Returns:
        [str], [str], int -- a tuple of a list of documents, a list of document IDs, and the number of documents
    """
    print("Loading document level corpus.")
    with open(
            Path(global_options.OUTPUT_FOLDER, "scores", "temp", "corpus_doc_level.pickle"),
            "rb",
    ) as in_f:
        corpus = pickle.load(in_f)
    with open(
            Path(global_options.OUTPUT_FOLDER, "scores", "temp", "doc_ids.pickle"), "rb"
    ) as in_f:
        doc_ids = pickle.load(in_f)
    assert len(corpus) == len(doc_ids)
    N_doc = len(corpus)
    return corpus, doc_ids, N_doc


def score_tf(documents, doc_ids, expanded_dict):
    """
    Score documents using term freq. 
    """
    print("Scoring using Term-freq (tf).")
    score = dictionary.score_tf(
        documents=documents,
        document_ids=doc_ids,
        expanded_words=expanded_dict,
        n_core=global_options.N_CORES,
    )
    score.to_csv(
        Path(global_options.OUTPUT_FOLDER, "scores", "scores_TF.csv"), index=False
    )


def score_tf_idf(documents, doc_ids, N_doc, method, expanded_dict, **kwargs):
    """Score documents using tf-idf and its variations

    Arguments:
        documents {[str]} -- list of documents
        doc_ids {[str]} -- list of document IDs
        N_doc {int} -- number of documents
        method {str} -- 
            TFIDF: conventional tf-idf 
            WFIDF: use wf-idf log(1+count) instead of tf in the numerator
            TFIDF/WFIDF+SIMWEIGHT: using additional word weights given by the word_weights dict
        expanded_dict {dict[str, set(str)]} -- expanded dictionary
    """
    if method == "TF":
        print("Scoring TF.")
        score_tf(documents, doc_ids, expanded_dict)
    else:
        print("Scoring TF-IDF.")
        # load document freq
        df_dict = pd.read_pickle(
            Path(global_options.OUTPUT_FOLDER, "scores", "temp", "doc_freq.pickle")
        )
        # score tf-idf
        score, contribution = dictionary.score_tf_idf(
            documents=documents,
            document_ids=doc_ids,
            expanded_words=expanded_dict,
            df_dict=df_dict,
            N_doc=N_doc,
            method=method,
            **kwargs
        )
        # save the document level scores (without dividing by doc length)
        score.to_csv(
            str(
                Path(
                    global_options.OUTPUT_FOLDER,
                    "scores",
                    "scores_{}.csv".format(method),
                )
            ),
            index=False,
        )
        # save word contributions
        pd.DataFrame.from_dict(contribution, orient="index").to_csv(
            Path(
                global_options.OUTPUT_FOLDER,
                "scores",
                "word_contributions",
                "word_contribution_{}.csv".format(method),
            )
        )


def run_scoring_pipeline(
    dict_path,
    corpus_path,
    id_path,
    methods,
    **kwargs
):
    """
    Runs the full scoring pipeline:
      1) Reads the dictionary (and optional similarity weights).
      2) Constructs a document-level corpus from sentence-level corpus.
      3) Calculates document frequency (df).
      4) Scores documents via TF or TF-IDF-based methods.

    Parameters
    ----------
    dict_path : str or Path
        Path to the expanded dictionary CSV.
    corpus_path : str or Path
        Path to the processed sentences corpus file (.txt).
    id_path : str or Path
        Path to the file containing sentence IDs corresponding to the corpus.
    methods : list of str
        A list of methods to run. E.g. ["TF", "TFIDF", "WFIDF", ...].
    **kwargs : dict
        Any additional arguments you want passed to the 'score_tf_idf' function.
        For instance, you can include:
           normalize=False,
           word_weights=...,
           etc.
    """

    # 1. Read the dictionary
    dict, all_dict_words = dictionary.read_dict_from_csv(dict_path)
    # (Optional) words weighted by similarity rank
    word_sim_weights = dictionary.compute_word_sim_weights(dict_path)

    # 2. Create document-level corpus
    corpus, doc_ids, N_doc = construct_doc_level_corpus(
        sent_corpus_file=corpus_path,
        sent_id_file=id_path
    )

    # 3. Calculate document frequency
    _ = calculate_df(corpus)  # returns df_dict, but you can store it if needed

    # 4. Score each requested method
    for method in methods:
        score_tf_idf(
            documents=corpus,
            doc_ids=doc_ids,
            N_doc=N_doc,
            method=method,
            expanded_dict=dict,
            word_weights=word_sim_weights,
            **kwargs
        )


if __name__ == "__main__":
    run_scoring_pipeline(
        dict_path=Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv"),
        corpus_path=Path(global_options.DATA_FOLDER, "processed", "trigram", "documents.txt"),
        id_path=Path(global_options.DATA_FOLDER, "processed", "parsed", "document_sent_ids.txt"),
        methods=['TF']
    )