"""Pydantic models for domain content generation."""

from typing import List, Optional

from pydantic import BaseModel, Field

from src.config import config

# ===== Base Models =====


class Section(BaseModel):
    """A section within a chapter or article."""

    name: str = Field(description="Section heading/title")
    content: str = Field(description="Body text of this section")
    idx: int = Field(description="Section order (1-based)")


class QAItem(BaseModel):
    """A question-answer pair with metadata."""

    question: str = Field(description="Question text")
    answer: str = Field(description="Answer text (concise and grounded)")
    related_chapter_idx: Optional[int] = Field(
        default=None,
        description="Index of related chapter (1-based) for book QAs",
    )
    related_section_idx: Optional[int] = Field(
        default=None,
        description="Index of related section (1-based) if applicable",
    )
    is_grounded: bool = Field(
        description="True if answerable from the content, False otherwise"
    )


# ===== Book Models =====


class TOCEntry(BaseModel):
    """Table of contents entry for a book chapter."""

    idx: int = Field(description="Planned chapter order index (1-based)")
    chapter_title: str = Field(description="Planned chapter title")
    summary: str = Field(description="1–2 sentence chapter summary")


class Chapter(BaseModel):
    """A book chapter with sections."""

    title: str = Field(description="Chapter title")
    sections: List[Section] = Field(
        description="Ordered sections for this chapter",
        min_length=config.sections_min_per_chapter,
        max_length=config.sections_max_per_chapter,
    )
    idx: int = Field(description="Chapter order index (1-based)")


class Book(BaseModel):
    """A complete book on a topic."""

    title: str = Field(description="Book title")
    topic: str = Field(description="Topic this book is based on")
    table_of_contents: List[TOCEntry] = Field(
        description="Planned chapters (table of contents)",
        min_length=config.toc_min_items,
        max_length=config.toc_max_items,
    )
    chapters: List[Chapter] = Field(
        description="Drafted chapter contents in order",
        min_length=config.toc_min_items,
        max_length=config.toc_max_items,
    )
    grounded_questions: List[QAItem] = Field(
        description="QA pairs answerable from book content",
        min_length=config.grounded_qa_min_items,
        max_length=config.grounded_qa_max_items,
    )
    ungrounded_questions: List[QAItem] = Field(
        description="QA pairs NOT answerable from book content",
        min_length=config.ungrounded_qa_min_items,
        max_length=config.ungrounded_qa_max_items,
    )


# ===== Article Models =====


class Article(BaseModel):
    """A research article/paper on a specific topic aspect."""

    title: str = Field(description="Article/paper title")
    topic: str = Field(description="Topic this article is based on")
    abstract: str = Field(description="Article abstract/summary")
    sections: List[Section] = Field(
        description="Ordered sections for this article",
        min_length=config.sections_min_per_article,
        max_length=config.sections_max_per_article,
    )
    grounded_questions: List[QAItem] = Field(
        description="QA pairs answerable from article content",
        min_length=config.grounded_qa_min_items,
        max_length=config.grounded_qa_max_items,
    )
    ungrounded_questions: List[QAItem] = Field(
        description="QA pairs NOT answerable from article content",
        min_length=config.ungrounded_qa_min_items,
        max_length=config.ungrounded_qa_max_items,
    )


# ===== Domain Models =====


class Topic(BaseModel):
    """A topic within a domain."""

    name: str = Field(description="Topic name")
    description: str = Field(description="Brief description of the topic")
    idx: int = Field(description="Topic index (1-based)")


class Domain(BaseModel):
    """A complete domain with topics, books, and articles."""

    name: str = Field(description="Domain name (e.g., 'Brazil', 'Machine Learning')")
    description: str = Field(description="Domain description")
    topics: List[Topic] = Field(
        description="List of topics within this domain",
        min_length=config.topics_min_items,
        max_length=config.topics_max_items,
    )
    books: List[Book] = Field(
        description="Books, each based on one topic",
        default_factory=list,
    )
    articles: List[Article] = Field(
        description="Articles/papers on specific aspects of topics",
        default_factory=list,
    )


# ===== Output Models for LLM Structured Generation =====


class TopicPlannerOutput(BaseModel):
    """Output from the topic planner node."""

    topics: List[Topic] = Field(
        description="List of topics to cover in this domain",
        min_length=config.topics_min_items,
        max_length=config.topics_max_items,
    )


class BookPlannerOutput(BaseModel):
    """Output from the book planner node."""

    title: str = Field(description="Proposed book title")
    table_of_contents: List[TOCEntry] = Field(
        description="Planned chapters produced by the planner",
        min_length=config.toc_min_items,
        max_length=config.toc_max_items,
    )


class ArticleOutlineOutput(BaseModel):
    """Output from the article planner node."""

    title: str = Field(description="Article title")
    abstract: str = Field(description="Article abstract")
    section_names: List[str] = Field(
        description="Ordered section names",
        min_length=config.sections_min_per_article,
        max_length=config.sections_max_per_article,
    )


class GroundedQAOutput(BaseModel):
    """Output from grounded QA generator."""

    questions: List[QAItem] = Field(
        description="Grounded QA pairs answerable from content",
        min_length=config.grounded_qa_min_items,
        max_length=config.grounded_qa_max_items,
    )


class UngroundedQAOutput(BaseModel):
    """Output from ungrounded QA generator."""

    questions: List[QAItem] = Field(
        description="Ungrounded QA pairs NOT answerable from content",
        min_length=config.ungrounded_qa_min_items,
        max_length=config.ungrounded_qa_max_items,
    )
