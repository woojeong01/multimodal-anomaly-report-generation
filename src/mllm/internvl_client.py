"""InternVL client for MMAD evaluation - HuggingFace transformers."""
from __future__ import annotations

import contextlib
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from .base import BaseLLMClient, INSTRUCTION, INSTRUCTION_WITH_AD, format_ad_info

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _patch_linspace_for_meta():
    """Workaround: InternVL calls .item() on torch.linspace() during __init__.

    When transformers initializes with meta tensors, linspace creates meta
    tensors where .item() fails. This forces linspace to use CPU.
    """
    _orig = torch.linspace

    def _safe_linspace(*args, **kwargs):
        kwargs.setdefault("device", "cpu")
        return _orig(*args, **kwargs)

    torch.linspace = _safe_linspace
    try:
        yield
    finally:
        torch.linspace = _orig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    """Build image transform for InternVL."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find closest aspect ratio for dynamic preprocessing."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamic preprocessing for InternVL2."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image_file: str, input_size: int = 448, max_num: int = 12):
    """Load and preprocess image for InternVL."""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVLClient(BaseLLMClient):
    """InternVL client using HuggingFace transformers.

    Supported models:
    - OpenGVLab/InternVL2-{1,2,4,8}B
    - OpenGVLab/InternVL2_5-{1,2,4,8}B
    - OpenGVLab/InternVL3_5-8B
    """

    NUM_LAYERS = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32,
        'InternVL2-8B': 32, 'InternVL2-26B': 48, 'InternVL2-40B': 60,
        'InternVL2-Llama3-76B': 80,
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 32,
        'InternVL2_5-8B': 32,
        'InternVL3_5-8B': 32,
    }

    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL2-8B",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 128,
        num_gpus: int = 1,
        max_patches: int = 1,  # Keep low for memory efficiency
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.device = device
        self.torch_dtype_str = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.num_gpus = num_gpus
        self.max_patches = max_patches

        self._model = None
        self._tokenizer = None

    def _get_torch_dtype(self):
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype_str, torch.bfloat16)

    def _split_model(self, model_name: str):
        """Create device map for multi-GPU inference."""
        device_map = {}
        world_size = torch.cuda.device_count()

        # Find matching key
        num_layers = None
        for key, val in self.NUM_LAYERS.items():
            if key in model_name:
                num_layers = val
                break

        if num_layers is None:
            num_layers = 32  # default

        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for _ in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1

        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map

    def _load_model(self):
        """Lazy load model and tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        # Loading InternVL model

        model_name = self.model_path.split('/')[-1]
        torch_dtype = self._get_torch_dtype()

        torch.set_grad_enabled(False)

        # InternVL custom code calls .item() during __init__,
        # which fails with meta tensors when low_cpu_mem_usage=True
        load_kwargs = dict(
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )

        if self.num_gpus > 1:
            load_kwargs["device_map"] = self._split_model(model_name)

        with _patch_linspace_for_meta():
            self._model = AutoModel.from_pretrained(
                self.model_path, **load_kwargs
            ).eval()

        if self.num_gpus <= 1 and self.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.cuda()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False
        )

        # Model loaded

    def build_payload(
        self,
        query_image_path: str,
        few_shot_paths: List[str],
        questions: List[Dict[str, str]],
        ad_info: Optional[Dict] = None,
    ) -> dict:
        """Build InternVL message format."""
        # Select instruction based on AD info availability
        if ad_info:
            instruction = INSTRUCTION_WITH_AD.format(ad_info=format_ad_info(ad_info))
        else:
            instruction = INSTRUCTION

        # Build text prompt with image placeholders
        prompt = instruction + "\n"

        if few_shot_paths:
            prompt += f"Following is/are {len(few_shot_paths)} image of normal sample, which can be used as a template to compare the image being queried."
            for _ in few_shot_paths:
                prompt += "\n<image>\n"

        prompt += "Following is the query image:\n<image>\n"
        prompt += "Following is the question list. Answer with the option's letter from the given choices directly:\n"

        for q in questions:
            prompt += f"{q['text']}\n"

        return {
            "prompt": prompt,
            "query_image": query_image_path,
            "few_shot_paths": few_shot_paths,
        }

    def send_request(self, payload: dict) -> Optional[dict]:
        """Process request using local model."""
        self._load_model()

        torch_dtype = self._get_torch_dtype()

        # Load images
        query_image = load_image(payload["query_image"], max_num=self.max_patches).to(torch_dtype)
        if self.device == "cuda":
            query_image = query_image.cuda()

        template_images = []
        for ref_path in payload["few_shot_paths"]:
            try:
                img = load_image(ref_path, max_num=self.max_patches).to(torch_dtype)
                if self.device == "cuda":
                    img = img.cuda()
                template_images.append(img)
            except Exception as e:
                continue

        images = template_images + [query_image]
        pixel_values = torch.cat(images, dim=0)
        num_patches_list = [img.shape[0] for img in images]

        # Generate
        generation_config = dict(max_new_tokens=self.max_new_tokens, do_sample=False)

        response, _ = self._model.chat(
            self._tokenizer,
            pixel_values,
            payload["prompt"],
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True
        )

        return {"response": response}

    def extract_response_text(self, response: dict) -> str:
        """Extract text from response."""
        return response.get("response", "")

    def generate_answers(
        self,
        query_image_path: str,
        meta: dict,
        few_shot_paths: List[str],
        ad_info: Optional[Dict] = None,
    ) -> Tuple[List[Dict], List[str], Optional[List[str]], List[str]]:
        """Generate answers with conversation history (InternVL's approach)."""
        questions, answers, question_types = self.parse_conversation(meta)

        if not questions or not answers:
            return questions, answers, None, question_types

        self._load_model()
        torch_dtype = self._get_torch_dtype()

        # Load images once
        query_image = load_image(query_image_path, max_num=self.max_patches).to(torch_dtype)
        if self.device == "cuda":
            query_image = query_image.cuda()

        template_images = []
        for ref_path in few_shot_paths:
            try:
                img = load_image(ref_path, max_num=self.max_patches).to(torch_dtype)
                if self.device == "cuda":
                    img = img.cuda()
                template_images.append(img)
            except Exception as e:
                continue

        images = template_images + [query_image]
        pixel_values = torch.cat(images, dim=0)
        num_patches_list = [img.shape[0] for img in images]

        # Select instruction based on AD info availability
        if ad_info:
            instruction = INSTRUCTION_WITH_AD.format(ad_info=format_ad_info(ad_info))
        else:
            instruction = INSTRUCTION

        # Build base prompt
        base_prompt = instruction + "\n"
        if few_shot_paths:
            base_prompt += f"Following is/are {len(few_shot_paths)} image of normal sample, which can be used as a template to compare the image being queried."
            for _ in few_shot_paths:
                base_prompt += "\n<image>\n"
        base_prompt += "Following is the query image:\n<image>\n"
        base_prompt += "Following is the question list. Answer with the option's letter from the given choices directly:\n"

        predicted_answers = []
        history = None

        generation_config = dict(max_new_tokens=self.max_new_tokens, do_sample=False)

        for i in range(len(questions)):
            part_questions = questions[i:i + 1]
            conversation_text = part_questions[0]["text"]
            query = base_prompt + conversation_text

            response, _history = self._model.chat(
                self._tokenizer,
                pixel_values,
                query,
                generation_config,
                num_patches_list=num_patches_list,
                history=history,
                return_history=True
            )
            # Note: _history is intentionally unused (no conversation history by default)

            parsed = self.parse_answer(response)
            if parsed:
                predicted_answers.append(parsed[-1])
            else:
                predicted_answers.append('')

        return questions, answers, predicted_answers, question_types

    def generate_answers_batch(
        self,
        query_image_path: str,
        meta: dict,
        few_shot_paths: List[str],
        ad_info: Optional[Dict] = None,
    ) -> Tuple[List[Dict], List[str], Optional[List[str]], List[str]]:
        """Generate answers for ALL questions in a single model call (5-8x faster)."""
        questions, answers, question_types = self.parse_conversation(meta)

        if not questions or not answers:
            return questions, answers, None, question_types

        self._load_model()
        torch_dtype = self._get_torch_dtype()

        # Load images once
        query_image = load_image(query_image_path, max_num=self.max_patches).to(torch_dtype)
        if self.device == "cuda":
            query_image = query_image.cuda()

        template_images = []
        for ref_path in few_shot_paths:
            try:
                img = load_image(ref_path, max_num=self.max_patches).to(torch_dtype)
                if self.device == "cuda":
                    img = img.cuda()
                template_images.append(img)
            except Exception:
                continue

        images = template_images + [query_image]
        pixel_values = torch.cat(images, dim=0)
        num_patches_list = [img.shape[0] for img in images]

        # Select instruction based on AD info availability
        if ad_info:
            instruction = INSTRUCTION_WITH_AD.format(ad_info=format_ad_info(ad_info))
        else:
            instruction = INSTRUCTION

        # Build prompt with ALL questions
        prompt = instruction + "\n"
        if few_shot_paths:
            prompt += f"Following is/are {len(few_shot_paths)} image of normal sample, which can be used as a template to compare the image being queried."
            for _ in few_shot_paths:
                prompt += "\n<image>\n"
        prompt += "Following is the query image:\n<image>\n"
        prompt += "Following is the question list. Answer with the option's letter from the given choices directly:\n"

        # Add ALL questions
        for q in questions:
            prompt += q["text"] + "\n"

        generation_config = dict(max_new_tokens=self.max_new_tokens * len(questions), do_sample=False)

        # Single model call for all questions
        response, _ = self._model.chat(
            self._tokenizer,
            pixel_values,
            prompt,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True
        )

        # Parse all answers from response
        parsed = self.parse_answer(response)

        # Pad with empty strings if not enough answers
        while len(parsed) < len(questions):
            parsed.append('')

        return questions, answers, parsed[:len(questions)], question_types
