import spacy


class SpacyParser:
    def __init__(self, model="en_core_web_sm", use_gpu=False):
        """
        Initialize by loading the specified spaCy language model (default is English).
        """
        if use_gpu:
            spacy.require_gpu()
        self.nlp = spacy.load(model)

    def sentence_split(self, text, lemma=False, ner=False, pos=False):
        """
        Split the input text into sentences and process each sentence.

        :param text: Input text.
        :param lemma: If True, perform lemmatization on the text.
        :param ner: If True, apply NER formatting.
        :param pos: If True, append POS tags.
        :return: A list of tuples (sentence_id, processed_sentence).
        """
        # Optionally apply lemmatization to the entire text first.
        if lemma:
            text = self.lemma(text)

        doc = self.nlp(text)
        sentences = []
        for idx, sent in enumerate(doc.sents):
            sentence_text = sent.text
            # If both NER and POS are requested, use the combined function.
            if ner and pos:
                sentence_text = self.ner_pos(sentence_text)
            elif ner:
                sentence_text = self.ner(sentence_text)
            elif pos:
                sentence_text = self.pos(sentence_text)
            sentences.append((idx, sentence_text))
        return sentences

    def lemma(self, sentence):
        """
        Perform lemmatization on the sentence.

        :param sentence: Input sentence.
        :return: Sentence with each token replaced by its lemma.
        """
        doc = self.nlp(sentence)
        lemmatized = " ".join(token.lemma_ for token in doc)
        return lemmatized

    def ner(self, sentence):
        """
        Process the sentence for named entities.
        Merge tokens belonging to an entity with underscores and prepend with a tag
        formatted as [NER:ENTITY_TYPE] (using spaCy's recognized entity type directly).

        :param sentence: Input sentence.
        :return: Processed sentence with formatted named entities.
        """
        doc = self.nlp(sentence)
        tokens = []
        i = 0
        while i < len(doc):
            token = doc[i]
            if token.ent_iob_ == 'B':
                ent_tokens = [token.text]
                ent_type = token.ent_type_
                i += 1
                # Collect subsequent tokens inside the same entity.
                while i < len(doc) and doc[i].ent_iob_ == 'I':
                    ent_tokens.append(doc[i].text)
                    i += 1
                merged_ent = "_".join(ent_tokens)
                tokens.append(f"[NER:{ent_type}]{merged_ent}")
            else:
                tokens.append(token.text)
                i += 1
        return " ".join(tokens)

    def pos(self, sentence):
        """
        Append part-of-speech (POS) tags to each token in the sentence.
        Each token is formatted as 'word[POS:TAG]'.

        :param sentence: Input sentence.
        :return: Processed sentence with POS tags appended.
        """
        doc = self.nlp(sentence)
        tokens = [f"{token.text}[POS:{token.pos_}]" for token in doc]
        return " ".join(tokens)

    def ner_pos(self, sentence):
        """
        Combined NER and POS processing.
        For tokens that are part of a named entity, merge them and format as:
          [NER:ENTITY_TYPE]merged_entity (without an extra POS tag).
        For tokens not part of any entity, append the POS tag as: word[POS:TAG].

        :param sentence: Input sentence.
        :return: Processed sentence with combined NER formatting and POS tagging.
        """
        doc = self.nlp(sentence)
        tokens = []
        i = 0
        while i < len(doc):
            token = doc[i]
            if token.ent_iob_ == 'B':
                ent_tokens = [token.text]
                ent_type = token.ent_type_
                i += 1
                while i < len(doc) and doc[i].ent_iob_ == 'I':
                    ent_tokens.append(doc[i].text)
                    i += 1
                merged_ent = "_".join(ent_tokens)
                tokens.append(f"[NER:{ent_type}]{merged_ent}")
            else:
                tokens.append(f"{token.text}[POS:{token.pos_}]")
                i += 1
        return " ".join(tokens)


# Example usage:
if __name__ == "__main__":

    parser = SpacyParser()

    text = ("When I was a child in Ohio, I always wanted to go to Stanford University. "
            "But I had to go along with my parents.")
    text = "I wanted to sell the house immediately"
    text = "consulting"
    text = "consulting tomorrow"
    for sentence, idx in parser.sentence_split(text, lemma=True):
        print(sentence, idx)
    print(parser.sentence_split(text, lemma=True))

