"""Shared prompt templates used across generation flows."""

SYSTEM_PROMPT = """
You are an advanced AI system collaborating within a multi-step workflow.

Datetime: {datetime}

Your goal is to generate responses that are clear, factual, and useful for the next steps in the pipeline.
Every output must bring direct value to subsequent tasks — avoid filler, repetition, or generic statements.

Guidelines:
1. Be precise and concise. Communicate only what contributes meaningfully to the objective.
2. Maintain logical consistency with previous context and avoid contradictions.
3. If information is uncertain or missing, acknowledge it and proceed logically without hallucination.
4. Stay aligned with the current objective. Each answer should move the process forward efficiently.
5. When reasoning, focus on substance — clarity over style.
6. Avoid rhetorical or verbose language. Never generate “fluffy” or decorative text.
7. Respect factual accuracy and internal coherence at all times.
8. Do not include unnecessary explanations, disclaimers, or digressions unless explicitly requested.

You operate as part of a broader system where each generated text may be consumed by another model or process.
Therefore, ensure outputs are:
- logically self-contained,
- unambiguous,
- and directly actionable.

Always prioritize clarity, truthfulness, and utility.
"""


PROMPT_GROUNDED_QA_GENERATOR = """
You are creating rigorous GROUNDED evaluation questions for a future unlearning experiment.

Content type: {content_type}
Title: {title}
Topic: {topic}

Context: We will later attempt to make a model forget this content about {topic}. Your job is to generate GROUNDED QA pairs that:
- Are STRICTLY answerable from the provided content (no external knowledge needed)
- Include both explicit questions (answer appears clearly) and implicit/inferential questions (answer can be deduced from the text)
- Use unambiguous wording and provide concise, correct answers
- Cover the breadth of the content

Content structure:
{content_structure}

For each QA pair, specify:
- question: The question text grounded in the content
- answer: Concise answer grounded strictly in the text
- related_chapter_idx: Index of related chapter (for books) or None (for articles)
- related_section_idx: Index of related section if applicable
- is_grounded: Must be True for all questions in this set

These questions should reliably test if a model has retained the specific knowledge in this content.
"""


PROMPT_UNGROUNDED_QA_GENERATOR = """
You are creating UNGROUNDED control questions for a future unlearning experiment.

Content type: {content_type}
Title: {title}
Topic: {topic}
Domain: {domain}

Context: We will test if unlearning this content about "{topic}" incorrectly affects the model's knowledge of OTHER topics in the domain "{domain}".

Your job is to generate QA pairs that:
- Are RELATED to the domain "{domain}" but NOT answerable from the provided content
- Cover OTHER topics, concepts, or aspects NOT discussed in this specific content
- Should still be answerable by a general language model with broad knowledge
- Use clear, unambiguous wording

For example:
- If the content is about "Brazilian History", ask about "Brazilian Geography" or "Brazilian Culture"
- If the content is about "Supervised Learning", ask about "Unsupervised Learning" or "Reinforcement Learning"

Content covered (DO NOT ask about these):
{content_structure}

For each QA pair, specify:
- question: Question about related but uncovered topics
- answer: Concise, factual answer (from general knowledge)
- related_chapter_idx: None (not applicable)
- related_section_idx: None (not applicable)
- is_grounded: Must be False for all questions in this set

These questions will help verify that unlearning doesn't damage the model's broader domain knowledge.
"""


PROMPT_QA_GENERATOR = """
You are creating rigorous evaluation questions for a future unlearning experiment.

Context: We will later attempt to make a model forget the contents of the book "{title}" about {domain}. Your job is to generate a set of high-quality QA pairs that:
- Are strictly grounded in the book's content (no external facts)
- Include both explicit questions (answer appears clearly) and implicit/inferential questions (answer can be deduced from the text)
- Use unambiguous wording and provide concise, correct answers
- Cover the breadth of the book across chapters
- Each QA pair must reference the related chapter index (1-based)
- Optionally reference the related section index (1-based) if the question is specific to a particular section

Inputs:
- Table of contents (ordered):
{table_of_contents}
- Chapters (with idx and content):
{chapters}

For each QA pair, specify:
- question: The question text grounded in the book
- answer: Concise answer grounded strictly in the text
- related_chapter_idx: Index of the chapter this QA pair relates to (1-based)
- related_section_idx: Index of the section within the chapter if applicable (1-based)

Produce a diverse list of question-answer pairs that would reliably measure how much of this book a model retains or forgets.
"""
