from pathlib import Path
import global_options
from Utils.parser import SpacyParser
from Utils import file_process


def parse_document(input_path, input_id, output_path, output_id, gpu=False, **kwargs):
    parser = SpacyParser(use_gpu=gpu)
    def parse_line(line, line_id):
        """Parse each line and return a tuple of sentences, sentence_IDs,

        Arguments:
            line {str} -- a document
            line_id {str} -- the document ID

        Returns:
            str, str -- processed document with each sentence in a line,
                        sentence IDs with each in its own line: lineID_0 lineID_1 ...
        """
        try:
            sent = parser.sentence_split(line, **kwargs)
        except Exception as e:
            print(e)
            print("Exception in line: {}".format(line_id))

        sentences = [item[1].strip("\n") for item in sent if item[1].strip()]
        sentence_ids = [f"{line_id}_{item[0]}" for item in sent]

        processed_sentences = "\n".join(sentences)
        processed_sentence_ids = "\n".join(sentence_ids)

        return processed_sentences, processed_sentence_ids

    file_process.process_large_file(
        input_file=input_path,
        input_file_ids=input_id,
        output_file=output_path,
        output_index_file=output_id,
        function_name=parse_line,
        chunk_size=global_options.PARSE_CHUNK_SIZE
    )

if __name__ == "__main__":
    parse_document(
        input_path=Path(
            global_options.DATA_FOLDER, "Input", "documents.txt"
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