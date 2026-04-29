"""Simple JSON-backed internationalization helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class I18n:
    DEFAULT_LANGUAGE = "uk"
    SUPPORTED_LANGUAGES = ("uk", "en")

    def __init__(self, language: str = DEFAULT_LANGUAGE) -> None:
        self._base_path = Path(__file__).resolve().parent / "locales"
        self._cache: dict[str, dict[str, str]] = {}
        self._language = self.DEFAULT_LANGUAGE
        self._translations: dict[str, str] = {}
        self.set_language(language)

    @property
    def language(self) -> str:
        return self._language

    def set_language(self, language: str) -> None:
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        self._translations = self._load(language)
        self._language = language

    def t(self, key: str, **params: Any) -> str:
        text = self._translations.get(key, key)
        if not params:
            return text
        try:
            return text.format(**params)
        except (IndexError, KeyError, ValueError):
            return text

    def _load(self, language: str) -> dict[str, str]:
        cached = self._cache.get(language)
        if cached is not None:
            return cached

        path = self._base_path / f"{language}.json"
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self._cache[language] = data
        return data