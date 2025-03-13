import datetime
import itertools
import os
import sys
from multiprocessing import Pool, freeze_support
from pathlib import Path
import json

import pandas as pd
from tqdm import tqdm


def line_counter(a_file):
    """Count the number of lines in a text file

    Arguments:
        a_file {str or Path} -- input text file

    Returns:
        int -- number of lines in the file
    """
    n_lines = 0
    with open(a_file, "rb") as f:
        n_lines = sum(1 for _ in f)
    return n_lines


def file_to_list(a_file):
    """Read a text file to a list, each line is an element

    Arguments:
        a_file {str or path} -- path to the file

    Returns:
        [str] -- list of lines in the input file, can be empty
    """
    file_content = []
    with open(a_file, "rb") as f:
        for l in f:
            file_content.append(l.decode(encoding="utf-8").strip())
    return file_content


def list_to_file(lst, a_file, validate=True):
    """Write a list to a file, each element in a line
    The strings needs to have no line break "\n" or they will be removed

    Keyword Arguments:
        validate {bool} -- check if number of lines in the file
            equals to the length of the list (default: {True})
    """
    with open(a_file, "w", 8192000, encoding="utf-8", newline="\n") as f:
        for e in lst:
            e = str(e).replace("\n", " ").replace("\r", " ")
            f.write("{}\n".format(e))
    if validate:
        assert line_counter(a_file) == len(lst)


def read_large_file(a_file, block_size=10000):
    """A generator to read text files into blocks
    Usage:
    for block in read_large_file(filename):
        do_something(block)

    Arguments:
        a_file {str or path} -- path to the file

    Keyword Arguments:
        block_size {int} -- [number of lines in a block] (default: {10000})
    """
    block = []
    with open(a_file) as file_handler:
        for line in file_handler:
            block.append(line)
            if len(block) == block_size:
                yield block
                block = []
    # yield the last block
    if block:
        yield block


def merge_fields_and_contents(field_file, content_file, output_field_file, output_content_file):
    """
    Merge contents based on field names from two text files: one containing field names and the other containing corresponding contents.
    Similar fields (like A_1, A_2, etc.) are merged into a single field (like A), and the contents for these fields are concatenated.

    Arguments:
    field_file {str} -- Path to the text file containing field names.
    content_file {str} -- Path to the text file containing corresponding contents.
    output_field_file {str} -- Path to the output file to write merged field names.
    output_content_file {str} -- Path to the output file to write merged contents.

    The function reads field names and contents, merges contents for similar field names (removing any numerical suffixes from field names),
    and writes the unique field names and their concatenated contents to the specified output files.
    """
    with open(field_file, 'r', encoding='utf-8') as f_fields, open(content_file, 'r', encoding='utf-8') as f_contents:
        fields = f_fields.readlines()
        contents = f_contents.readlines()

    if len(fields) != len(contents):
        raise ValueError("Field and content files do not have the same number of lines")

    merged_contents = {}
    for field, content in tqdm(zip(fields, contents)):
        field_key = ''.join(filter(lambda x: not x.isdigit(), field.strip()))

        if field_key in merged_contents:
            merged_contents[field_key] += content.strip() + ' '
        else:
            merged_contents[field_key] = content.strip()

    with open(output_field_file, 'w', encoding='utf-8') as f_output_fields, open(output_content_file, 'w', encoding='utf-8') as f_output_contents:
        for field, content in merged_contents.items():
            f_output_fields.write(field + '\n')
            f_output_contents.write(content + '\n')


def process_large_file(
    input_file,
    output_file,
    input_file_ids,
    output_index_file,
    function_name,
    chunk_size=100,
    start_index=None,
):
    """ A helper function that transforms an input file + a list of IDs of each line (documents + document_IDs) to two output files (processed documents + processed document IDs) by calling function_name on chunks of the input files. Each document can be decomposed into multiple processed documents (e.g. sentences).
    Supports parallel with Pool.

    Arguments:
        input_file {str or Path} -- path to a text file, each line is a document
        output_file {str or Path} -- processed line sentence file (remove if exists)
        input_file_ids {str]} -- a list of input line ids
        output_index_file {str or Path} -- path to the index file of the output
        function_name {callable} -- A function that processes a list of strings, list of ids and return a list of processed strings and ids.
        chunk_size {int} -- number of lines to process each time, increasing the default may increase performance
        start_index {int} -- line number to start from (index starts with 0)

    Writes:
        Write the output_file and output_index_file
    """
    try:
        if start_index is None:
            # if start from the first line, remove existing output file
            # else append to existing output file
            os.remove(str(output_file))
            if output_index_file is not None:
                os.remove(str(output_index_file))
    except OSError:
        pass
    assert line_counter(input_file) == len(
        input_file_ids
    ), "Make sure the input file has the same number of rows as the input ID file. "

    with open(input_file, newline="\n", encoding="utf-8", errors="ignore") as f_in:
        line_i = 0
        # jump to index
        if start_index is not None:
            # start at start_index line
            for _ in range(start_index):
                next(f_in)
            input_file_ids = input_file_ids[start_index:]
            line_i = start_index
        for next_n_lines, next_n_line_ids in zip(
            itertools.zip_longest(*[f_in] * chunk_size),
            itertools.zip_longest(*[iter(input_file_ids)] * chunk_size),
        ):
            line_i += chunk_size
            print(datetime.datetime.now())
            print(f"Processing line: {line_i}.")
            next_n_lines = list(filter(lambda x: x is not None, next_n_lines))
            next_n_line_ids = list(filter(lambda x: x is not None, next_n_line_ids))
            output_lines = []
            output_line_ids = []
            # Use parse_parallel.py to speed things up
            for output_line, output_line_id in map(
                function_name, next_n_lines, next_n_line_ids
            ):
                output_lines.append(output_line)
                output_line_ids.append(output_line_id)
            output_lines = "\n".join(output_lines) + "\n"
            output_line_ids = "\n".join(output_line_ids) + "\n"
            with open(output_file, "a", newline="\n", encoding="utf-8") as f_out:
                f_out.write(output_lines)
            if output_index_file is not None:
                with open(output_index_file, "a", newline="\n") as f_out:
                    f_out.write(output_line_ids)

