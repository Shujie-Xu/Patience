"""
Module: utils/clean.py
Description: Provides text cleaning functions encapsulated in the TextCleaner class.
"""

import re
from nltk.corpus import stopwords


class TextCleaner:
    """
    A class to encapsulate various text cleaning operations.

    This class allows you to configure cleaning options such as lower-casing,
    number removal, punctuation removal, stopword removal (with support for a
    custom stopword set), extra whitespace removal, removal of single-character tokens,
    and removal of extra repeated punctuation.
    """

    def __init__(self, to_lower=False, remove_num=False, remove_punc=False, remove_extra_punc=False,
                 remove_stop=False, remove_single=False, custom_stop=None, language='english',
                 ):
        """
        Initialize the TextCleaner with the specified cleaning options.

        Args:
            to_lower (bool): Whether to convert text to lower case.
            remove_num (bool): Whether to remove numbers from the text.
            remove_punc (bool): Whether to remove punctuation.
            remove_extra_punc (bool): Whether to remove extra repeated punctuation.
                                       This normalizes sequences of punctuation to a single instance.
            remove_stop (bool): Whether to remove stopwords.
            remove_single (bool): Whether to remove single-character tokens.
            custom_stop (set, optional): A custom set of stopwords. If not provided,
                                          the NLTK stopword set for the given language is used.
            language (str): The language to use for NLTK stopwords (default is 'english').
        """
        self.to_lower = to_lower
        self.remove_num = remove_num
        self.remove_punc = remove_punc
        self.remove_extra_punc = remove_extra_punc
        self.remove_stop = remove_stop
        self.remove_single = remove_single
        self.language = language


        # Load stopwords from the custom set or from NLTK if stopword removal is enabled.
        if remove_stop:
            if custom_stop is not None:
                self.stops = set(custom_stop)
            else:
                try:
                    self.stops = set(stopwords.words(language))
                except LookupError:
                    import nltk
                    nltk.download('stopwords')
                    self.stops = set(stopwords.words(language))
        else:
            self.stops = set()

    def _to_lower(self, text):
        """
        Convert text to lower case.

        Args:
            text (str): The input text.

        Returns:
            str: Text in lower case.
        """
        return text.lower()

    def _remove_numbers(self, text):
        """
        Remove numeric characters from the text.

        Args:
            text (str): The input text.

        Returns:
            str: Text with numbers removed.
        """
        return re.sub(r'\d+', '', text)

    def _remove_punctuation(self, text):
        """
        Remove punctuation from the text.

        Args:
            text (str): The input text.

        Returns:
            str: Text with punctuation removed.
        """
        return re.sub(r'[^\w\s]', ' ', text)

    def _remove_extra_punctuation(self, text):
        """
        Normalize repeated punctuation in the text.

        This function replaces sequences of repeated punctuation characters with a single instance.
        For example, "!!!" becomes "!".

        Args:
            text (str): The input text.

        Returns:
            str: Text with repeated punctuation normalized.
        """
        return re.sub(r'([^\w\s])\1+', r'\1', text)

    def _remove_extra_whitespace(self, text):
        """
        Remove extra whitespace characters from the text.

        Args:
            text (str): The input text.

        Returns:
            str: Text with extra whitespace removed.
        """
        return re.sub(r'\s+', ' ', text).strip()

    def _remove_stopwords(self, text):
        """
        Remove stopwords from the text using the loaded stopword set.

        Args:
            text (str): The input text.

        Returns:
            str: Text with stopwords removed.
        """
        return " ".join(word for word in text.split() if word not in self.stops | {"-lrb-", "-rrb-", "-lsb-", "-rsb-", "'s"})

    def _remove_single(self, text):
        """
        Remove single-character tokens from the text.

        Args:
            text (str): The input text.

        Returns:
            str: Text with single-character tokens removed.
        """
        tokens = text.split()
        tokens = [word for word in tokens if len(word) > 1]
        return " ".join(tokens)

    def clean(self, text):
        """
        Clean the input text by applying the configured cleaning operations.

        Args:
            text (str): The input text.

        Returns:
            str: The cleaned text.
        """
        if self.to_lower:
            text = self._to_lower(text)
        if self.remove_num:
            text = self._remove_numbers(text)
        if self.remove_extra_punc:
            text = self._remove_extra_punctuation(text)
        if self.remove_punc:
            text = self._remove_punctuation(text)
        text = self._remove_extra_whitespace(text)
        if self.remove_stop:
            text = self._remove_stopwords(text)
        if self.remove_single:
            text = self._remove_single(text)
        return text
