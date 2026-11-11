"""
Ways of running the vision-language model. All backends take a list of
Requests and return one raw text answer per request, in the same order.

- "transformers": loads the model with Hugging Face transformers on a
  CUDA GPU, Apple MPS or CPU. The schema is only described in the prompt,
  so answers are checked afterwards (parsing.py).
- "server": sends requests to an OpenAI-compatible server, e.g. vLLM on a
  GPU machine or LM Studio on a Mac. The server enforces the JSON schema
  (structured output), so invalid categories/styles can't be generated.
- "dry-run": no model; returns a placeholder answer built from the schema.
  For testing the rest of the pipeline.

The heavy libraries (torch, transformers, openai) are imported inside the
backend that needs them, so the other backends work without them.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from PIL import Image

from src.data_generation.image_to_text import config

BACKENDS = ("transformers", "server", "dry-run")


@dataclass
class Request:
    """One item to describe."""

    furniture_id: str
    image_path: str
    prompt: str
    schema: dict


class TransformersBackend:
    """
    Runs the model locally with transformers.

    Memory: in bf16 a model needs roughly 2 bytes per parameter, so the 4B
    model needs about 9 GB, more than an 8 GB Mac has. Use a GPU (e.g.
    Colab/Kaggle/KTH), a smaller model for testing, or the server backend
    with a quantised model.
    """

    enforces_schema = False

    def __init__(self, model_name=config.DEFAULT_MODEL, device=None, max_new_tokens=config.MAX_NEW_TOKENS):
        """
        Args:
            model_name: Hugging Face model ID.
            device: "cuda", "mps" or "cpu". Picked automatically if None.
            max_new_tokens: Upper limit on the answer length.
        """
        import torch
        from transformers import AutoProcessor

        # The Qwen3.5 model card loads the model with AutoModelForMultimodalLM;
        # older transformers versions only have AutoModelForImageTextToText.
        try:
            from transformers import AutoModelForMultimodalLM as AutoModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as AutoModelClass

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_name)
        # Batched generation needs the padding on the left, so every answer
        # starts right after its own prompt.
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelClass.from_pretrained(model_name, dtype="auto").to(device).eval()
        print(f"Loaded {model_name} on {device}")

    def generate(self, requests, sampling=None):
        """
        Args:
            requests: List of Request.
            sampling: None for greedy decoding, or a dict with temperature,
                top_p and top_k.

        Returns:
            List of raw answer strings.
        """
        texts = []
        images = []
        for request in requests:
            messages = [{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": request.prompt}],
            }]
            # Rendering the template to text first is the documented way to
            # pass enable_thinking. Qwen3.5 thinks by default, which is slow
            # and not needed for describing an image.
            texts.append(self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False,
            ))
            images.append(Image.open(request.image_path).convert("RGB"))

        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        generate_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        if sampling:
            generate_kwargs.update(do_sample=True, **sampling)

        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)


class ServerBackend:
    """
    Sends requests to an OpenAI-compatible server, several at a time (the
    server batches them on the GPU).

    Examples of starting a server:
        vLLM (GPU):  vllm serve Qwen/Qwen3.5-4B --port 8000
        LM Studio (Mac): load a quantised Qwen3.5 model and start the local
            server (default http://localhost:1234/v1).
    """

    enforces_schema = True

    def __init__(self, model_name=config.DEFAULT_MODEL, base_url="http://localhost:8000/v1", api_key="EMPTY",
                 workers=8, use_schema=True, max_new_tokens=config.MAX_NEW_TOKENS):
        """
        Args:
            model_name: Model name as the server knows it.
            base_url: Server URL, ending in /v1.
            api_key: Any string for local servers.
            workers: How many requests are in flight at once.
            use_schema: Ask the server to enforce the JSON schema. Turn off
                if the server rejects response_format.
            max_new_tokens: Upper limit on the answer length.
        """
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.workers = workers
        self.use_schema = use_schema
        self.enforces_schema = use_schema
        self.max_new_tokens = max_new_tokens

    def _generate_one(self, request, sampling):
        with open(request.image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")

        kwargs = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": request.prompt},
                ],
            }],
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
            # Non-standard options go in extra_body; servers ignore the ones
            # they don't know.
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if sampling:
            kwargs["temperature"] = sampling["temperature"]
            kwargs["top_p"] = sampling["top_p"]
            kwargs["extra_body"]["top_k"] = sampling["top_k"]
        if self.use_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "item_description", "schema": request.schema, "strict": True},
            }

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as error:  # one failed request shouldn't stop the run
            return f"<request failed: {error}>"

    def generate(self, requests, sampling=None):
        """Same as TransformersBackend.generate."""
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda r: self._generate_one(r, sampling), requests))


class DryRunBackend:
    """Returns a placeholder answer that fits the schema. No model needed."""

    enforces_schema = True

    def generate(self, requests, sampling=None):
        """Same as TransformersBackend.generate."""
        import json

        answers = []
        for request in requests:
            answer = {}
            for field, spec in request.schema["properties"].items():
                if "enum" in spec:
                    answer[field] = spec["enum"][0]
                elif spec["type"] == "array":
                    answer[field] = [spec["items"]["enum"][0]]
                elif spec["type"] == "boolean":
                    answer[field] = False
                else:
                    answer[field] = f"placeholder {field}"
            answers.append(json.dumps(answer))
        return answers


def make_backend(name, **kwargs):
    """
    Create a backend by name.

    Args:
        name: One of BACKENDS.
        **kwargs: Passed to the backend's constructor (unused ones are
            dropped, so run_captioning.py can pass all its settings).

    Returns:
        A backend object with a generate(requests, sampling=None) method.
    """
    if name == "transformers":
        keys = ("model_name", "device", "max_new_tokens")
        return TransformersBackend(**{k: v for k, v in kwargs.items() if k in keys and v is not None})
    if name == "server":
        keys = ("model_name", "base_url", "api_key", "workers", "use_schema", "max_new_tokens")
        return ServerBackend(**{k: v for k, v in kwargs.items() if k in keys and v is not None})
    if name == "dry-run":
        return DryRunBackend()
    raise ValueError(f"Unknown backend {name!r}; use one of {BACKENDS}")
