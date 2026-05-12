def get_autofeedback_instructions() -> str:
    """
    Returns the strict criteria and prompt instructions for the LLM to self-evaluate
    its generated charts based on best practices for data visualization.
    This replaces the complex multi-step image evaluation with a logic-based single pass.
    """
    return """
CRITICAL INSTRUCTION FOR CHART GENERATION MODE: If you are drawing charts, you MUST start your response by providing the exact TWO ```python ... ``` blocks requested above. DO NOT skip the python code!

After providing the two python code blocks, you MUST include the following two markdown headings to provide self-feedback:

### Key Insights Summary
(Provide 2-3 key insights based on the data)

### Chart Evaluation & Suggestions
Critique your chart design based on professional criteria (Chart Type, Text Legibility, Color, Visual Clutter, and Annotation) and provide actionable suggestions to improve it.
"""
