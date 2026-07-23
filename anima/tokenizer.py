from pathlib import Path

from transformers import T5TokenizerFast, Qwen2TokenizerFast


class AnimaTokenizer:
    def __init__(self, root_dir):
        root = Path(root_dir) / "anima" / "model"
        self.qwen_tokenizer = Qwen2TokenizerFast.from_pretrained(
            (root / "tokenizer" / "Qwen2.5-0.5B-Instruct").as_posix(), local_files_only=True
        )
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(
            (root / "tokenizer" / "t5-v1_1-xxl").as_posix(), local_files_only=True
        )
        if self.qwen_tokenizer.pad_token_id is None:
            self.qwen_tokenizer.pad_token_id = 151643

    def tokenize(self, text):
        qwen_out = self.qwen_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        t5_out = self.t5_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        return {
            "qwen_input_ids": qwen_out.input_ids,
            "qwen_attention_mask": qwen_out.attention_mask,
            "t5_input_ids": t5_out.input_ids,
            "t5_attention_mask": t5_out.attention_mask,
        }
