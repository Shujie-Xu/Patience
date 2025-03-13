import gensim
from gensim.corpora import Dictionary
from pathlib import Path
import os


def train_w2v_model(input_path, model_path, *args, **kwargs):
    """ Train a word2vec model using the LineSentence file in input_path,
    save the model to model_path.count

    Arguments:
        input_path {str} -- Corpus for training, each line is a sentence
        model_path {str} -- Where to save the model?
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    corpus_confcall = gensim.models.word2vec.PathLineSentences(
        str(input_path), max_sentence_length=10000000
    )
    model = gensim.models.Word2Vec(corpus_confcall, *args, **kwargs)
    model.save(str(model_path))


def train_lda_model(input_path, model_path, *args, **kwargs):
    """
    Train an LDA model using the corpus at input_path and save it to model_path.

    Arguments:
        input_path {str or Path} -- Corpus for training, each line is a sentence
        model_path {str or Path} -- Where to save the model?
        num_topics {int} -- Number of topics to be extracted by the LDA model.

    Additional arguments (*args and **kwargs) can be passed to the LdaModel constructor.
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    documents = gensim.models.word2vec.PathLineSentences(
        str(input_path), max_sentence_length=10000000
    )
    dictionary = Dictionary(documents)
    corpus_politic = [dictionary.doc2bow(doc) for doc in documents]

    lda_model = gensim.models.ldamodel.LdaModel(
        corpus_politic, id2word=dictionary, *args, **kwargs
    )

    lda_model.save(str(model_path))
    return lda_model


def tf_idf_keywords(folder_path, document_path, num_keywords=500, *args, **kwargs):
    """
    Train a TF-IDF model using documents in a specified folder and then extract keywords
    from a specified document.

    Arguments:
        folder_path {str or Path} -- Path to the folder containing documents for training.
        document_path {str or Path} -- Path to the document for keyword extraction.
        num_keywords {int} -- Number of top keywords to extract.
    """
    # Read and preprocess documents from the folder
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(Path(folder_path) / filename, 'r', encoding='utf-8') as file:
                documents.append(file.read().split())

    # Create a dictionary and corpus for the TF-IDF model
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Train the TF-IDF model
    tfidf = gensim.models.TfidfModel(
        corpus, *args, **kwargs
    )

    # Read the specified document
    with open(document_path, 'r', encoding='utf-8') as file:
        document = file.read().split()

    # Convert the document to the bag-of-words format
    doc_bow = dictionary.doc2bow(document)

    # Apply the TF-IDF model to the document
    doc_tfidf = tfidf[doc_bow]

    # Sort the words by their TF-IDF score
    sorted_keywords = sorted(doc_tfidf, key=lambda kv: kv[1], reverse=True)

    # Extract the top N keywords
    keywords = [(dictionary[word_id], score) for word_id, score in sorted_keywords[:num_keywords]]
    return keywords