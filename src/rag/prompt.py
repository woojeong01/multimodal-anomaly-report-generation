from __future__ import annotations
from typing import Dict, List, Optional
from langchain_core.documents import Document


INSTRUCTION_RAG = '''
You are an industrial inspector who checks products by images. You should judge whether there is a defect in the query image and answer the questions about it.
An anomaly detection model has analyzed this image and provided the following information:
{ad_info}

Here is the domain knowledge about known defect types for this product:
{domain_knowledge}

Use the anomaly detection results and domain knowledge along with your visual analysis to answer the questions.
Answer with the option's letter from the given choices directly!

Finally, you should output a list of answer, such as:
1. Answer: B.
2. Answer: B.
3. Answer: A.
...
'''

# -- 위 prompt 내용 --

# 너는 이미지로 제품을 검사하는 산업 검사관이야.                                                       
# 쿼리 이미지에 결함이 있는지 판단하고 질문에 답해.                                                  
                                                                                                    
# 이상 탐지 모델이 이 이미지를 분석한 결과야:                                                          
# {ad_info}                                                                                            
                                                                                                    
# 이 제품의 알려진 결함 유형에 대한 도메인 지식이야:
# {domain_knowledge}

# 이상 탐지 결과와 도메인 지식, 그리고 네 시각 분석을 활용해서 답해.
# 선택지의 알파벳으로 바로 답해!

# 답변은 이렇게 출력해:
# 1. Answer: B.
# 2. Answer: B.
# 3. Answer: A.



REPORT_PROMPT_RAG = '''You are an expert industrial quality inspector.
Look at this product image carefully and determine if there are any defects or anomalies.

Product category: {category}

Here is the domain knowledge about known defect types for this product:
{domain_knowledge}

Use this domain knowledge along with your visual analysis to make your judgment.
Pay close attention to the defect characteristics described in the domain knowledge.

If the product looks perfect and normal, set "is_anomaly" to false.
If there is ANY abnormality, set "is_anomaly" to true.

Respond in JSON format ONLY:
{{{{
  "is_anomaly": true or false,
  "report": {{{{
    "anomaly_type": "type of defect or none",
    "severity": "low/medium/high/none",
    "location": "where the defect is or none",
    "description": "detailed defect description or normal product",
    "confidence": 0.0 to 1.0,
    "recommendation": "action recommendation"
  }}}},
  "summary": {{{{
    "summary": "one sentence inspection summary",
    "risk_level": "low/medium/high/none"
  }}}}
}}}}'''


def report_prompt_rag(
    category: str,
    domain_knowledge: str = "",
) -> str:
    """Build a RAG-augmented report generation prompt.

    Args:
        category: Product category string.
        domain_knowledge: Formatted domain knowledge from retriever.format_context().

    Returns:
        Filled report prompt string.
    """
    return REPORT_PROMPT_RAG.format(
        category=category,
        domain_knowledge=domain_knowledge or "No relevant domain knowledge found.",
    )


def rag_prompt(
    ad_info: str = "",
    domain_knowledge: str = "",
) -> str:
    """Build a complete RAG-augmented instruction prompt.

    Args:
        ad_info: Formatted AD model output string.
        domain_knowledge: Formatted domain knowledge from retriever.format_context().

    Returns:
        Filled instruction prompt string.
    """
    return INSTRUCTION_RAG.format(
        ad_info=ad_info or "No anomaly detection information available.",
        domain_knowledge=domain_knowledge or "No relevant domain knowledge found.",
    )
