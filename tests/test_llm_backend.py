from rca_framework.llm.backend import VLLMBackend


class RecordingTokenizer:
    chat_template = "{{ bos_token }}<｜User｜>{{ content }}<｜Assistant｜>"
    bos_token = "<bos>"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.calls.append((messages, tokenize, add_generation_prompt))
        return "canonical-chat-template"


def test_vllm_render_prefers_tokenizer_chat_template_for_deepseek_markers():
    tokenizer = RecordingTokenizer()
    backend = VLLMBackend(model_path="unused", _tokenizer=tokenizer)
    assert backend._render("diagnose") == "canonical-chat-template"
    assert tokenizer.calls == [
        ([{"role": "user", "content": "diagnose"}], False, True)
    ]


class BrokenTemplateTokenizer(RecordingTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        raise ValueError("template unavailable")


def test_vllm_render_uses_manual_deepseek_fallback_only_after_template_failure():
    backend = VLLMBackend(model_path="unused", _tokenizer=BrokenTemplateTokenizer())
    assert backend._render("diagnose") == "<bos><｜User｜>diagnose<｜Assistant｜>"
