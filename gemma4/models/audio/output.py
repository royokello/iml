import torch

class Gemma4AudioModelOutput(BaseModelOutputWithPooling):
    r"""
    attention_mask (`torch.BoolTensor`, *optional*):
        A torch.BoolTensor of shape `(batch_size, num_frames)`. True for valid positions, False for padding.
    """

    attention_mask: torch.BoolTensor | None = None
