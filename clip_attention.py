"""Scoped CLIP attention dispatch shared by generation and the encoder probe."""

from contextlib import contextmanager


@contextmanager
def attention_profile(profile, torch):
    if profile == "native":
        yield
        return
    if profile != "math":
        raise ValueError(f"Unknown attention profile: {profile}")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        # MHA's native fast path can bypass the SDPA backend selection.
        torch.backends.mha.set_fastpath_enabled(False)
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
