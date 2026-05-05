__all__ = ["load_gemma4_language_model", "load_gemma4_text_model"]


def __getattr__(name):
    if name in __all__:
        from gemma4.loaders.text import load_gemma4_language_model, load_gemma4_text_model

        loaders = {
            "load_gemma4_language_model": load_gemma4_language_model,
            "load_gemma4_text_model": load_gemma4_text_model,
        }
        return loaders[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
