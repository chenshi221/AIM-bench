try:
    from clip.simple_tokenizer import SimpleTokenizer
except ImportError as exc:
    raise ImportError(
        "OpenAI CLIP is required for LongCLIP tokenization. Install dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc
