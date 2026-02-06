"""Base class for LLM clients following MMAD paper evaluation protocol."""
from __future__ import annotations

import base64
import re
import time
from abc import ABC, abstractmethod
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

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

An anomaly detection model has analyzed this image and provided the following information:
{ad_info}

Use this information along with your visual analysis to answer the questions.
Answer with the option's letter from the given choices directly!

Finally, you should output a list of answer, such as:
1. Answer: B.
2. Answer: B.
3. Answer: A.
...
'''


def format_ad_info(ad_info: dict) -> str:
    """Format AD model output as a string for LLM prompt.

    Args:
        ad_info: Dictionary containing AD model predictions.
                 Expected keys may include: anomaly_score, is_anomaly,
                 defect_location, defect_type, bbox, mask_path, etc.

    Returns:
        Formatted string describing the AD model's findings.
    """
    if not ad_info:
        return "No anomaly detection information available."

    import json
    # Return JSON string for flexibility - LLM can interpret structured data
    return json.dumps(ad_info, indent=2, ensure_ascii=False)


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
