"""Google Gemini client for MMAD evaluation."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import cv2

from .base import BaseLLMClient, INSTRUCTION, INSTRUCTION_WITH_AD, format_ad_info, get_mime_type


class GeminiClient(BaseLLMClient):
    """Gemini client following MMAD paper's evaluation protocol.

    Supported models:
    - gemini-1.5-flash (free tier: 15 RPM, 1500 requests/day)
    - gemini-1.5-pro (free tier: 2 RPM, 50 requests/day)
    - gemini-2.0-flash-exp (experimental)

    Get API key from: https://aistudio.google.com/app/apikey
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        max_retries: int = 5,
        **kwargs,
    ):
        super().__init__(max_retries=max_retries, **kwargs)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.model_name = model

        if not self.api_key:
            raise ValueError(
                "Google API key not provided. "
                "Set GOOGLE_API_KEY env var or pass api_key. "
                "Get free key from: https://aistudio.google.com/app/apikey"
            )

        self._model = None

    def _load_model(self):
        """Lazy load Gemini model."""
        if self._model is not None:
            return

        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Please install google-generativeai: pip install google-generativeai"
            )

        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(model_name=self.model_name)

    def send_request(self, payload: dict) -> Optional[dict]:
        """Send request to Gemini API with retry logic."""
        self._load_model()

        retry_delay = 1
        retries = 0

        while retries < self.max_retries:
            try:
                before = time.time()
                response = self._model.generate_content(payload)
                response.resolve()
                self.api_time_cost += time.time() - before
                return {"text": response.text}

            except Exception as e:
                retries += 1
                if retries < self.max_retries:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        return None

    def build_payload(
        self,
        query_image_path: str,
        few_shot_paths: List[str],
        questions: List[Dict[str, str]],
        ad_info: Optional[Dict] = None,
    ) -> dict:
        """Build Gemini API payload following paper's format."""

        # Build in-context description
        incontext = ""
        if few_shot_paths:
            incontext = f"The first {len(few_shot_paths)} image is the normal sample, which can be used as a template to compare."

        # Encode few-shot images
        parts = []
        for ref_path in few_shot_paths:
            try:
                ref_image = cv2.imread(ref_path)
                if ref_image is None:
                    continue
                ref_base64 = self.encode_image_to_base64(ref_image)
                parts.append({
                    "inline_data": {
                        "mime_type": get_mime_type(ref_path),
                        "data": ref_base64
                    }
                })
            except Exception as e:
                continue

        # Encode query image
        query_image = cv2.imread(query_image_path)
        if query_image is None:
            raise FileNotFoundError(f"Cannot read image: {query_image_path}")
        query_base64 = self.encode_image_to_base64(query_image)
        parts.append({
            "inline_data": {
                "mime_type": get_mime_type(query_image_path),
                "data": query_base64
            }
        })

        # Build conversation text
        conversation_text = ""
        for q in questions:
            conversation_text += f"{q['text']}\n"

        # Select instruction based on AD info availability
        if ad_info:
            instruction = INSTRUCTION_WITH_AD.format(ad_info=format_ad_info(ad_info))
        else:
            instruction = INSTRUCTION

        # Add text prompt
        parts.append({
            "text": (
                instruction +
                incontext +
                "The last image is the query image. " +
                "Following is the question list: \n" +
                conversation_text
            )
        })

        return {"parts": parts}

    def extract_response_text(self, response: dict) -> str:
        """Extract text from Gemini response."""
        return response.get("text", "")

    def generate_answers(
        self,
        query_image_path: str,
        meta: dict,
        few_shot_paths: List[str],
        ad_info: Optional[Dict] = None,
    ) -> Tuple[List[Dict], List[str], Optional[List[str]], List[str]]:
        """Generate answers following paper's Gemini approach.

        Paper's approach for Gemini:
        1. First ask question 1 alone (anomaly detection)
        2. Then ask questions 2-5 together
        """
        questions, answers, question_types = self.parse_conversation(meta)

        if not questions or not answers:
            return questions, answers, None, question_types

        predicted_answers = []

        # First question alone (anomaly detection)
        first_question = questions[:1]
        payload = self.build_payload(query_image_path, few_shot_paths, first_question, ad_info=ad_info)
        response = self.send_request(payload)

        if response is None:
            return questions, answers, None, question_types

        response_text = self.extract_response_text(response)

        parsed = self.parse_answer(response_text)
        if parsed:
            predicted_answers.append(parsed[-1])
        else:
            predicted_answers.append('')

        # Remaining questions together
        if len(questions) > 1:
            remaining_questions = questions[1:]
            payload = self.build_payload(query_image_path, few_shot_paths, remaining_questions, ad_info=ad_info)
            response = self.send_request(payload)

            if response is None:
                # Fill with empty answers
                predicted_answers.extend([''] * len(remaining_questions))
            else:
                response_text = self.extract_response_text(response)

                parsed = self.parse_answer(response_text)
                # Pad if not enough answers
                while len(parsed) < len(remaining_questions):
                    parsed.append('')
                predicted_answers.extend(parsed[:len(remaining_questions)])

        return questions, answers, predicted_answers, question_types
