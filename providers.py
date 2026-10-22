"""Small provider adapters used by the session-scoped RAG pipeline."""
from __future__ import annotations

from google import genai
from google.genai import types
from openai import OpenAI


class OpenAIProvider:
    """Use OpenAI for multilingual vectors and grounded answer generation."""

    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key, timeout=90, max_retries=2)

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        """Keep the API's index order when embedding a batch of passages."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small", input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def generate(self, prompt: str, instructions: str) -> str:
        """Ask the Responses API for one cited answer."""
        response = self.client.responses.create(
            model="gpt-6-luna", instructions=instructions, input=prompt,
            max_output_tokens=900, store=False, reasoning={"effort": "none"})
        return (response.output_text or "").strip()


class GeminiProvider:
    """Use Gemini's free-tier capable embedding and text models."""

    def __init__(self, api_key: str) -> None:
        self.client = genai.Client(api_key=api_key)

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        """Request one vector per Content and format queries for document QA."""
        if kind == "query":
            prepared = [f"task: question answering | query: {text}" for text in texts]
        else:
            prepared = [f"title: none | text: {text}" for text in texts]
        contents = [types.Content(parts=[types.Part.from_text(text=text)])
                    for text in prepared]
        response = self.client.models.embed_content(
            model="gemini-embedding-2", contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=1536))
        return [embedding.values for embedding in (response.embeddings or [])]

    def generate(self, prompt: str, instructions: str) -> str:
        """Generate a grounded answer in the user's language."""
        response = self.client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=instructions, max_output_tokens=900))
        return (response.text or "").strip()


def create_provider(name: str, api_key: str) -> OpenAIProvider | GeminiProvider:
    """Build only the selected provider, so a second key is never required."""
    if name == "Gemini":
        return GeminiProvider(api_key)
    if name == "OpenAI":
        return OpenAIProvider(api_key)
    raise ValueError("Unsupported AI provider.")
