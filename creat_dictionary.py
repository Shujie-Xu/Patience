import global_options
import gensim
from Utils import dictionary
from pathlib import Path


def creat_dict(input_path, output_path):
    """
    Load a pre-trained Word2Vec model, expand and process the dictionary, and save the result as a CSV file.

    Parameters:
        input_path: The path to the Word2Vec model file.
        output_path: The path to save the expanded dictionary CSV file.
    """
    # Load the Word2Vec model
    model = gensim.models.Word2Vec.load(str(input_path))

    # Print the vocabulary size of the model
    vocab_number = len(model.wv.key_to_index)
    print("Vocab size in the Word2Vec model: {}".format(vocab_number))

    # Expand the dictionary based on the provided seed words
    expanded_words = dictionary.expand_words_dimension_mean(
        word2vec_model=model,
        seed_words=global_options.SEED_WORDS,
        restrict=global_options.DICT_RESTRICT_VOCAB,
        n=global_options.N_WORDS_DIM,
    )
    print("Dictionary created.")

    # Deduplicate keywords to ensure each word is assigned to only one dimension
    expanded_words = dictionary.deduplicate_keywords(
        word2vec_model=model,
        expanded_words=expanded_words,
        seed_words=global_options.SEED_WORDS,
    )
    print("Dictionary deduplicated.")

    # Rank the words within each dimension based on similarity to the seed words
    expanded_words = dictionary.rank_by_sim(
        expanded_words, global_options.SEED_WORDS, model
    )

    # Save the expanded dictionary to a CSV file
    dictionary.write_dict_to_csv(
        dict=expanded_words,
        file_name=output_path,
    )
    print("Dictionary saved at {}".format(output_path))


if __name__ == "__main__":
    # Example usage: using paths from global_options or customize them here
    input_path = Path(global_options.MODEL_FOLDER, "w2v", "w2v.mod")
    output_path = Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv")
    creat_dict(input_path, output_path)
