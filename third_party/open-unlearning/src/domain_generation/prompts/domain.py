"""Domain-level prompt templates."""

PROMPT_TOPIC_PLANNER = """
You are a domain expert tasked with breaking down a broad domain into distinct, well-defined topics.

Domain: {domain}
Description: {description}

Your task is to identify {min_topics} to {max_topics} major topics within this domain. Each topic should:
- Be substantial enough to warrant detailed exploration
- Be distinct from other topics (minimal overlap)
- Together, cover the essential landscape of the domain
- Be ordered logically (foundational → advanced, or chronological, etc.)

For each topic, provide:
- name: Clear, concise topic name
- description: 2-3 sentences explaining what this topic covers
- idx: Sequential number (1-based) representing logical order

These topics will later be used to generate books and research articles.
"""
