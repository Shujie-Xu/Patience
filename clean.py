"""
Module: clean.py
Description: Main file that integrates the TextCleaner from utils/clean.py to process an entire corpus.
"""

import global_options
from pathlib import Path
from Utils import file_process
from Utils.text_cleaning import TextCleaner


def clean_file(input_path, output_path, **kwargs):
    """
    Clean the entire corpus (output from CoreNLP) line by line and write the cleaned text to an output file.

    Args:
        input_path (str or Path): Input corpus file, each line is a sentence.
        output_path (str or Path): Output corpus file.
        **kwargs: Additional keyword arguments to configure the TextCleaner.
                  For example:
                    - to_lower (bool): Convert text to lower case. Default True.
                    - remove_num (bool): Remove numbers. Default True.
                    - remove_punc (bool): Remove punctuation. Default True.
                    - remove_stop (bool): Remove stopwords. Default True.
                    - remove_single (bool): Remove single-character tokens. Default True.
                    - custom_stop (set, optional): Custom stopword set.
                    - language (str): Language for stopwords. Default 'english'.
                  New parameters added in TextCleaner will be automatically accepted.
    """
    # Initialize the TextCleaner with the desired cleaning options using **kwargs.
    a_text_cleaner = TextCleaner(**kwargs)

    # Count the number of lines in the input file to generate fake IDs.
    total_lines = file_process.line_counter(input_path)
    input_file_ids = [str(i) for i in range(total_lines)]

    # Process the input file in large chunks using the cleaning function.
    file_process.process_large_file(
        input_file=input_path,
        output_file=output_path,
        input_file_ids=input_file_ids,  # Fake IDs, as they are not needed for this function.
        output_index_file=None,
        function_name=lambda line, _id: (a_text_cleaner.clean(line), _id),
        chunk_size=20000,
    )


if __name__ == "__main__":
    # Example usage: modify the paths and parameters as needed.
    input_corpus = Path(
        global_options.DATA_FOLDER, "Input", "documents.txt"
    )
    output_corpus = Path(
        global_options.DATA_FOLDER, "Processed", "cleaned", "documents.txt"
    )

    # Call clean_file with desired cleaning options.
    clean_file(input_corpus, output_corpus,
               to_lower=True,
               remove_num=True,
               remove_punc=True,
               remove_stop=True,
               remove_single=True,
               custom_stop=None,
               language='english')
