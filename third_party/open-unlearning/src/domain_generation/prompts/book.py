"""Prompt templates for book generation flows."""

PROMPT_BOOK_PLANNER = """
You are an expert book author specializing in {domain}. 
Your task is to design a complete table of contents for a book focused on the topic: "{topic}".

Topic description: {topic_description}

First, propose a concise, compelling book title that captures the scope of this specific topic.
For each chapter, provide:
- The chapter title
- A concise, one-paragraph summary describing what the chapter should cover

The table of contents must give a comprehensive overview of the topic — enough for a reader to understand the full scope without going into excessive depth. 
Each chapter should represent a key subtopic: together they form a coherent overview, but each one can also stand alone for deeper exploration.

The final book should be concise yet well-rounded, capturing all essential aspects of "{topic}" at a high level.
"""

# Legacy alias maintained for backward compatibility.
PROMPT_PLANNER = PROMPT_BOOK_PLANNER

PROMPT_CHAPTER_WRITER = """
You are an expert book author specializing in {domain}. 
Your current assignment is to write the chapter titled "{chapter_title}".
A previous summary of this chapter is provided to guide your writing:
{summary}

Other specialists are simultaneously writing the remaining chapters: {chapters_titles} (ordered by idx)
You are working independently, so focus exclusively on your assigned chapter.

Each chapter must be self-contained and provide complete understanding of its topic. 
It can reference concepts from other chapters if relevant, but it should remain understandable on its own.

Your goal is to deliver a clear, well-structured, and insightful chapter that contributes meaningful knowledge to the overall book on {topic}.
Avoid redundancy and ensure that every part of your text adds real informational value.
"""


PROMPT_REVIEWER = """
You are an expert editor ensuring consistency and quality across a multi-author book.

Task: Review the current draft for the book "{title}" on {domain}. Make light-touch edits to:
- Improve cross-chapter consistency (terminology, ordering, references)
- Fix small contradictions or overlaps
- Suggest brief connective sentences where needed to ensure flow

Important:
- Retain each chapter's core content and unique perspective.
- Do not significantly expand length; prioritize clarity and coherence.

Inputs:
- Table of contents (ordered):
{table_of_contents}
- Chapters (with idx and content):
{chapters}

Output a refined book preserving the structure: title, table_of_contents, chapters (each with idx, title, sections). Keep changes minimal but impactful.
"""
