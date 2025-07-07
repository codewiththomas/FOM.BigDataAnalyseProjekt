from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    """
    Anbstrakte Basis-Klasse für Tokenizer.
    """

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenisiert den gegebenen Text in eine Liste von Tokens.

        Args:
            text (str): Der zu tokenisierende Text.

        Returns:
            list[str]: Eine Liste von Tokens.
        """
        pass

    @abstractmethod
    def detokenize(self, tokens: list[str]) -> str:
        """
        Setzt eine Liste von Tokens in den ursprünglichen Text zurück.

        Args:
            tokens (list[str]): Die zu detokenisierenden Tokens.

        Returns:
            str: Der rekonstruierte Text.
        """
        pass
