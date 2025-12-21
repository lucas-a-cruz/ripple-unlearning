"""Prompt templates for article generation flows."""

PROMPT_ARTICLE_WRITER = """
You are a research expert writing a focused academic article on a specific aspect of {domain}.

Article title: {title}
Abstract: {abstract}
Topic: {topic}

Your task is to write section "{section_name}" (section {section_idx} of {total_sections}).

This article should:
- Be technically precise and well-researched
- Focus on a narrow, specific aspect of the topic
- Use clear, academic language
- Include detailed explanations suitable for someone with domain knowledge
- Cite concepts that would typically be referenced (though no actual citations needed)

Write comprehensive content for this section that advances the article's contribution to the field.
"""
