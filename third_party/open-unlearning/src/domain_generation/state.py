"""State classes for LangGraph graphs."""

from operator import add
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field

from src.models import (Article, Book, Chapter, Domain, QAItem, Section,
                        TOCEntry, Topic)


class DomainState(BaseModel):
    """State for the main domain graph."""

    name: str = Field(description="Domain name")
    description: str = Field(default="", description="Domain description")
    topics: Annotated[List[Topic], add] = []
    books: Annotated[List[Book], add] = []
    articles: Annotated[List[Article], add] = []
    pending_books: Annotated[int, add] = 0
    pending_articles: Annotated[int, add] = 0
    domain: Optional[Domain] = None


class BookState(BaseModel):
    """State for the book subgraph."""

    domain_name: str = Field(description="Domain name")
    topic: str = Field(description="Topic this book covers")
    topic_description: str = Field(default="", description="Topic description")
    title: Optional[str] = None
    table_of_contents: Annotated[List[TOCEntry], add] = []
    chapters: Annotated[List[Chapter], add] = []
    grounded_questions: Annotated[List[QAItem], add] = []
    ungrounded_questions: Annotated[List[QAItem], add] = []
    pending_chapters: Annotated[int, add] = 0
    pending_qa_tasks: Annotated[int, add] = 0
    book: Optional[Book] = None


class ChapterWriterState(BaseModel):
    """State for individual chapter writing tasks."""

    domain_name: str = Field(description="Domain name")
    topic: str = Field(description="Topic")
    chapter_title: str = Field(description="Title of this chapter")
    summary: str = Field(description="Chapter summary from TOC")
    chapter_titles: List[str] = Field(description="All chapter titles for context")
    idx: int = Field(description="Chapter index (1-based)")


class ArticleState(BaseModel):
    """State for the article subgraph."""

    domain_name: str = Field(description="Domain name")
    topic: str = Field(description="Topic this article covers")
    topic_description: str = Field(default="", description="Topic description")
    title: Optional[str] = None
    abstract: Optional[str] = None
    section_names: List[str] = []
    sections: Annotated[List[Section], add] = []
    grounded_questions: Annotated[List[QAItem], add] = []
    ungrounded_questions: Annotated[List[QAItem], add] = []
    pending_sections: Annotated[int, add] = 0
    pending_qa_tasks: Annotated[int, add] = 0
    article: Optional[Article] = None


class ArticleSectionWriterState(BaseModel):
    """State for individual article section writing tasks."""

    domain_name: str = Field(description="Domain name")
    topic: str = Field(description="Topic")
    title: str = Field(description="Article title")
    abstract: str = Field(description="Article abstract")
    section_name: str = Field(description="Name of this section")
    section_idx: int = Field(description="Section index (1-based)")
    total_sections: int = Field(description="Total number of sections")
