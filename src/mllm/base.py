"""Base class for LLM clients following MMAD paper evaluation protocol."""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)

# MMAD paper's instruction prompt
INSTRUCTION = '''
You are an industrial inspector who checks products by images. You should judge whether there is a defect in the query image and answer the questions about it.
Answer with the option's letter from the given choices directly!

Finally, you should output a list of answer, such as:
1. Answer: B.
2. Answer: B.
3. Answer: A.
...
'''

# Instruction with AD model output
INSTRUCTION_WITH_AD = '''
You are an industrial inspector who checks products by images. You should judge whether there is a defect in the query image and answer the questions about it.

An anomaly detection model has pre-analyzed this image. Here is the result:
{ad_info}

Consider this as a reference, but rely primarily on your own visual analysis. The model can make mistakes.
Answer with the option's letter from the given choices directly!

Finally, you should output a list of answer, such as:
1. Answer: B.
2. Answer: B.
3. Answer: A.
...
'''

# ── Report generation prompts ──────────────────────────────────────────

REPORT_PROMPT = '''You are an expert industrial quality inspector.
Look at this product image carefully and determine if there are any defects or anomalies.
Pay close attention to: surface damage, deformation, missing parts, wrong positioning,
opened packaging, contamination, cracks, scratches, or any other abnormality.

Product category: {category}

If the product looks perfect and normal, set "is_anomaly" to false.
If there is ANY abnormality, set "is_anomaly" to true.

Respond in JSON format ONLY:
{{
  "is_anomaly": true or false,
  "report": {{
    "anomaly_type": "type of defect or none",
    "severity": "low/medium/high/none",
    "location": "where the defect is or none",
    "description": "detailed defect description or normal product",
    "confidence": 0.0 to 1.0,
    "recommendation": "action recommendation"
  }},
  "summary": {{
    "summary": "one sentence inspection summary",
    "risk_level": "low/medium/high/none"
  }}
}}'''

REPORT_PROMPT_WITH_AD = '''You are an expert industrial quality inspector.
Look at this product image carefully and determine if there are any defects or anomalies.
Pay close attention to: surface damage, deformation, missing parts, wrong positioning,
opened packaging, contamination, cracks, scratches, or any other abnormality.

Product category: {category}

An anomaly detection model has pre-analyzed this image:
{ad_info}

Consider this as a reference, but rely primarily on your own visual analysis. The model can make mistakes.

If the product looks perfect and normal, set "is_anomaly" to false.
If there is ANY abnormality, set "is_anomaly" to true.

Respond in JSON format ONLY:
{{
  "is_anomaly": true or false,
  "report": {{
    "anomaly_type": "type of defect or none",
    "severity": "low/medium/high/none",
    "location": "where the defect is or none",
    "description": "detailed defect description or normal product",
    "confidence": 0.0 to 1.0,
    "recommendation": "action recommendation"
  }},
  "summary": {{
    "summary": "one sentence inspection summary",
    "risk_level": "low/medium/high/none"
  }}
}}'''


def _parse_llm_json(text: str) -> Optional[dict]:
    """Extract and parse JSON from LLM response text.

    Handles common LLM quirks: escaped underscores, markdown fences, etc.
    """
    # Fix LLaVA-style escaped underscores
    cleaned = text.replace("\\_", "_")

    # Try to extract JSON object
    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None


def _normalize_decision(value: Any) -> bool:
    """Normalize various LLM is_anomaly outputs to bool."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in ("true", "1", "yes", "anomaly", "이상", "불량", "defect", "bad", "abnormal")


def format_ad_info(ad_info: dict) -> str:
    """Format AD model output as a concise natural language summary.

    Args:
        ad_info: Dictionary containing AD model predictions.

    Returns:
        Formatted string describing the AD model's findings.
    """
    if not ad_info:
        return "No anomaly detection information available."

    lines = []

    # Anomaly score & judgment
    score = ad_info.get("anomaly_score")
    is_anomaly = ad_info.get("is_anomaly")
    if score is not None:
        status = "ANOMALOUS" if is_anomaly else "NORMAL"
        lines.append(f"- Anomaly detection result: {status} (score: {score:.2f})")

    # Defect location
    loc = ad_info.get("defect_location", {})
    if loc.get("has_defect"):
        region = loc.get("region", "unknown")
        area = loc.get("area_ratio", 0)
        lines.append(f"- Defect location: {region}")
        if area > 0:
            lines.append(f"- Defect area: {area * 100:.1f}% of the image")
    elif loc and not loc.get("has_defect"):
        lines.append("- No localized defect detected")

    if not lines:
        return "No anomaly detection information available."

    return "\n".join(lines)


def get_mime_type(image_path: str) -> str:
    """Get MIME type from image path."""
    path_lower = image_path.lower()
    if path_lower.endswith(".png"):
        return "image/png"
    elif path_lower.endswith(".jpeg") or path_lower.endswith(".jpg"):
        return "image/jpeg"
    return "image/jpeg"


class BaseLLMClient(ABC):
    """Base class for MMAD LLM evaluation.

    Follows the exact protocol from the paper:
    - Few-shot normal templates + query image + questions
    - Answer parsing with regex + fuzzy matching fallback
    """

    def __init__(
        self,
        max_image_size: Tuple[int, int] = (512, 512),
        max_retries: int = 5,
        visualization: bool = False,
    ):
        self.max_image_size = max_image_size
        self.max_retries = max_retries
        self.visualization = visualization
        self.api_time_cost = 0.0

    def encode_image_to_base64(self, image) -> str:
        """Encode image to base64, resizing if necessary.

        Args:
            image: BGR image (numpy array) or path string
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {image}")

        height, width = image.shape[:2]
        scale = min(
            self.max_image_size[0] / width,
            self.max_image_size[1] / height
        )

        if scale < 1.0:
            new_width, new_height = int(width * scale), int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        _, encoded = cv2.imencode('.jpg', image)
        return base64.b64encode(encoded).decode('utf-8')

    def parse_conversation(self, meta: dict) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
        """Parse MMAD conversation format into questions, answers, and types.

        Returns:
            questions: List of {"type": "text", "text": "Question: ... \nA. ...\nB. ..."}
            answers: List of correct answer letters
            question_types: List of question type strings
        """
        questions = []
        answers = []
        question_types = []

        # Find conversation key
        for key in meta.keys():
            if key.startswith("conversation"):
                conversation = meta[key]
                for qa in conversation:
                    # Build options text
                    options = qa.get("Options", qa.get("options", {}))
                    options_text = ""
                    if isinstance(options, dict):
                        for opt_key in sorted(options.keys()):
                            options_text += f"{opt_key}. {options[opt_key]}\n"

                    question_text = qa.get("Question", qa.get("question", ""))
                    questions.append({
                        "type": "text",
                        "text": f"Question: {question_text} \n{options_text}"
                    })
                    answers.append(qa.get("Answer", qa.get("answer", "")))
                    question_types.append(qa.get("type", "unknown"))
                break

        return questions, answers, question_types

    def parse_answer(self, response_text: str, options: Optional[Dict[str, str]] = None) -> List[str]:
        """Parse answer letters from LLM response.

        Uses regex pattern matching with fuzzy matching fallback.
        """
        pattern = re.compile(r'\b([A-E])\b')
        found_answers = pattern.findall(response_text)

        if len(found_answers) == 0 and options is not None:
            pass  # Fallback to fuzzy matching
            options_values = list(options.values())
            closest_matches = get_close_matches(response_text, options_values, n=1, cutoff=0.0)
            if closest_matches:
                closest_match = closest_matches[0]
                for key, value in options.items():
                    if value == closest_match:
                        found_answers.append(key)
                        break

        return found_answers

    @abstractmethod
    def send_request(self, payload: dict) -> Optional[dict]:
        """Send request to LLM API. Must be implemented by subclass."""
        pass

    @abstractmethod
    def build_payload(
        self,
        query_image_path: str,
        few_shot_paths: List[str],
        questions: List[Dict[str, str]],
        ad_info: Optional[Dict] = None,
    ) -> dict:
        """Build API payload. Must be implemented by subclass.

        Args:
            query_image_path: Path to the query image
            few_shot_paths: List of few-shot template image paths
            questions: List of question dictionaries
            ad_info: Optional anomaly detection model output dictionary
        """
        pass

    def generate_answers(
        self,
        query_image_path: str,
        meta: dict,
        few_shot_paths: List[str],
        ad_info: Optional[Dict] = None,
    ) -> Tuple[List[Dict], List[str], Optional[List[str]], List[str]]:
        """Generate answers for all questions in the conversation.

        Following paper's protocol: ask questions incrementally.

        Args:
            query_image_path: Path to the query image
            meta: MMAD metadata dictionary
            few_shot_paths: List of few-shot template image paths
            ad_info: Optional anomaly detection model output dictionary

        Returns:
            questions: Parsed questions
            correct_answers: Ground truth answers
            predicted_answers: Model predictions (None if failed)
            question_types: Question type strings
        """
        questions, answers, question_types = self.parse_conversation(meta)

        if not questions or not answers:
            return questions, answers, None, question_types

        predicted_answers = []

        # Paper's approach: ask incrementally (1 question, then 2, then 3...)
        for i in range(len(questions)):
            part_questions = questions[:i + 1]
            payload = self.build_payload(query_image_path, few_shot_paths, part_questions, ad_info=ad_info)

            response = self.send_request(payload)
            if response is None:
                predicted_answers.append('')
                continue

            response_text = self.extract_response_text(response)
            parsed = self.parse_answer(response_text)

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
        """Generate answers for all questions in a single API call.

        More efficient than incremental, but may be less accurate.

        Args:
            query_image_path: Path to the query image
            meta: MMAD metadata dictionary
            few_shot_paths: List of few-shot template image paths
            ad_info: Optional anomaly detection model output dictionary

        Returns:
            questions: Parsed questions
            correct_answers: Ground truth answers
            predicted_answers: Model predictions (None if failed)
            question_types: Question type strings
        """
        questions, answers, question_types = self.parse_conversation(meta)

        if not questions or not answers:
            return questions, answers, None, question_types

        payload = self.build_payload(query_image_path, few_shot_paths, questions, ad_info=ad_info)
        response = self.send_request(payload)

        if response is None:
            return questions, answers, None, question_types

        response_text = self.extract_response_text(response)
        parsed = self.parse_answer(response_text)

        # Pad with empty strings if not enough answers
        while len(parsed) < len(questions):
            parsed.append('')

        return questions, answers, parsed[:len(questions)], question_types

    @abstractmethod
    def extract_response_text(self, response: dict) -> str:
        """Extract text content from API response. Must be implemented by subclass."""
        pass

    # ── Report generation ──────────────────────────────────────────────

    def build_report_payload(
        self,
        image_path: str,
        category: str,
        ad_info: Optional[Dict] = None,
    ) -> dict:
        """Build payload for report generation.

        Subclasses may override this for model-specific formatting.
        Default implementation uses build_payload with an empty questions list
        and the report prompt as a single text question.
        """
        if ad_info:
            prompt_text = REPORT_PROMPT_WITH_AD.format(
                category=category,
                ad_info=format_ad_info(ad_info),
            )
        else:
            prompt_text = REPORT_PROMPT.format(category=category)

        questions = [{"type": "text", "text": prompt_text}]
        return self.build_payload(image_path, [], questions)

    def generate_report(
        self,
        image_path: str,
        category: str,
        ad_info: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a structured inspection report for a single image.

        Args:
            image_path: Path to the product image.
            category: Product category string (e.g. "cigarette_box").
            ad_info: Optional dict with AD model results (score, is_anomaly, etc.).

        Returns:
            Dict with keys: is_anomaly_LLM, llm_report, llm_summary.
        """
        payload = self.build_report_payload(image_path, category, ad_info)

        t0 = time.time()
        response = self.send_request(payload)
        inference_time = time.time() - t0

        # Default fallback
        result: Dict[str, Any] = {
            "is_anomaly_LLM": None,
            "llm_report": None,
            "llm_summary": None,
            "llm_inference_duration": round(inference_time, 3),
        }

        if response is None:
            logger.warning("LLM returned no response for %s", image_path)
            return result

        text = self.extract_response_text(response)
        parsed = _parse_llm_json(text)

        if parsed is None:
            logger.warning("Failed to parse JSON from LLM response: %s", text[:200])
            result["llm_report"] = {"raw_response": text}
            return result

        result["is_anomaly_LLM"] = _normalize_decision(parsed.get("is_anomaly", False))
        result["llm_report"] = parsed.get("report", parsed)
        result["llm_summary"] = parsed.get("summary")

        return result
