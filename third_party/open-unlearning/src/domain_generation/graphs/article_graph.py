"""Article generation subgraph."""

from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from src.config import config
from src.models import (Article, ArticleOutlineOutput, GroundedQAOutput,
                        Section, UngroundedQAOutput)
from src.prompts import (PROMPT_ARTICLE_WRITER, PROMPT_GROUNDED_QA_GENERATOR,
                         PROMPT_UNGROUNDED_QA_GENERATOR, SYSTEM_PROMPT)
from src.state import ArticleSectionWriterState, ArticleState
from src.utils import get_current_date, get_llm, pretty_log


def article_planner(state: ArticleState):
    """Plan the article structure."""
    llm = get_llm()
    pretty_log("article_planner", "start", {"topic": state.topic})

    prompt = f"""You are a research expert writing a focused academic article on {state.topic}.

Domain: {state.domain_name}
Topic: {state.topic}
Topic description: {state.topic_description}

Create an article outline with:
- A compelling, specific article title
- An informative abstract (2-3 sentences)
- {config.sections_min_per_article} to {config.sections_max_per_article} section names (ordered)

This article should cover a narrow, specific aspect of the topic - like a research paper.
"""

    response: ArticleOutlineOutput = llm.with_structured_output(
        ArticleOutlineOutput
    ).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime=get_current_date())),
            HumanMessage(content=prompt),
        ]
    )

    num_sections = len(response.section_names)
    pretty_log(
        "article_planner",
        "end",
        {"title": response.title, "sections": num_sections},
    )
    return {
        "title": response.title,
        "abstract": response.abstract,
        "section_names": response.section_names,
        "pending_sections": num_sections,
    }


def article_section_writer(state: ArticleSectionWriterState | dict):
    """Write a single article section."""
    llm = get_llm()
    pretty_log("article_section_writer", "start", state)

    if isinstance(state, dict):
        state = ArticleSectionWriterState(**state)

    prompt = PROMPT_ARTICLE_WRITER.format(
        domain=state.domain_name,
        topic=state.topic,
        title=state.title,
        abstract=state.abstract,
        section_name=state.section_name,
        section_idx=state.section_idx,
        total_sections=state.total_sections,
    )

    # Just get the content as text, wrap it in Section model
    content = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime=get_current_date())),
            HumanMessage(content=prompt),
        ]
    ).content

    section = Section(
        name=state.section_name,
        content=content,
        idx=state.section_idx,
    )

    pretty_log("article_section_writer", "end", {"section": state.section_name})
    return Command(
        update={
            "sections": [section],
            "pending_sections": -1,
        },
        goto="article_join_sections",
    )


def assign_article_section_writers(state: ArticleState):
    """Dispatch parallel section writing tasks."""
    pretty_log(
        "assign_article_section_writers",
        "start",
        {"num_sections": len(state.section_names)},
    )

    sends = [
        Send(
            "article_section_writer",
            {
                "domain_name": state.domain_name,
                "topic": state.topic,
                "title": state.title,
                "abstract": state.abstract,
                "section_name": section_name,
                "section_idx": idx + 1,
                "total_sections": len(state.section_names),
            },
        )
        for idx, section_name in enumerate(state.section_names)
    ]

    return sends


def article_join_sections(state: ArticleState | dict):
    """Wait for all section writers before dispatching QA tasks."""
    if isinstance(state, dict):
        state = ArticleState(**state)

    remaining = max(state.pending_sections, 0)

    if remaining <= 0:
        pretty_log(
            "article_join_sections",
            "sections_complete",
            {"title": state.title, "sections": len(state.sections)},
        )
        if not state.sections:
            return Command(goto="article_builder")
        return Command(
            update={"pending_qa_tasks": 2},
            goto="article_qa_dispatch",
        )

    pretty_log(
        "article_join_sections",
        "waiting",
        {"pending_sections": remaining, "title": state.title},
    )
    return None


def article_qa_dispatch(state: ArticleState | dict):
    """Entry node before article QA fan-out."""
    if isinstance(state, dict):
        state = ArticleState(**state)

    pretty_log(
        "article_qa_dispatch",
        "start",
        {"title": state.title, "sections": len(state.sections)},
    )
    return {}


def route_article_qa(state: ArticleState | dict):
    """Conditional edge handler for article QA fan-out."""
    if isinstance(state, dict):
        state = ArticleState(**state)

    payload = state.model_dump()
    sends = [
        Send("article_grounded_qa_generator", payload),
        Send("article_ungrounded_qa_generator", payload),
    ]

    pretty_log("article_qa_dispatch", "dispatch", {"jobs": len(sends)})
    return sends


def article_join_qa(state: ArticleState | dict):
    """Join QA generation before compiling the article."""
    if isinstance(state, dict):
        state = ArticleState(**state)

    remaining = max(state.pending_qa_tasks, 0)

    if remaining <= 0:
        pretty_log(
            "article_join_qa",
            "qa_complete",
            {
                "grounded": len(state.grounded_questions),
                "ungrounded": len(state.ungrounded_questions),
            },
        )
        return Command(goto="article_builder")

    pretty_log(
        "article_join_qa",
        "waiting",
        {"pending_qa_tasks": remaining, "title": state.title},
    )
    return None


def article_grounded_qa_generator(state: ArticleState | dict):
    """Generate grounded QA pairs from article content."""
    if isinstance(state, dict):
        state = ArticleState(**state)

    llm = get_llm()
    pretty_log("article_grounded_qa_generator", "start", {"title": state.title})

    sections_str = "\n\n".join(
        f"# [{s.idx}] {s.name}\n{s.content[:300]}..."
        for s in sorted(state.sections, key=lambda x: x.idx)
    )

    prompt = PROMPT_GROUNDED_QA_GENERATOR.format(
        content_type="Article",
        title=state.title or "<Untitled>",
        topic=state.topic,
        content_structure=f"Abstract: {state.abstract}\n\nSections:\n{sections_str}",
    )
    constraints_text = f"\n\nConstraints: Produce {config.grounded_qa_min_items} to {config.grounded_qa_max_items} grounded QA pairs."

    qa_output: GroundedQAOutput = llm.with_structured_output(GroundedQAOutput).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime=get_current_date())),
            HumanMessage(content=prompt + constraints_text),
        ]
    )

    for qa in qa_output.questions:
        qa.is_grounded = True
        qa.related_chapter_idx = None  # Articles don't have chapters

    max_grounded = config.grounded_qa_max_items
    grounded_questions = qa_output.questions[:max_grounded]
    if len(qa_output.questions) > max_grounded:
        pretty_log(
            "article_grounded_qa_generator",
            "truncate",
            {
                "requested": len(qa_output.questions),
                "kept": len(grounded_questions),
                "title": state.title,
            },
        )

    pretty_log(
        "article_grounded_qa_generator",
        "end",
        {"count": len(grounded_questions), "title": state.title},
    )
    return Command(
        update={
            "grounded_questions": grounded_questions,
            "pending_qa_tasks": -1,
        },
        goto="article_join_qa",
    )


def article_ungrounded_qa_generator(state: ArticleState | dict):
    """Generate ungrounded (control) QA pairs."""
    if isinstance(state, dict):
        state = ArticleState(**state)

    llm = get_llm()
    pretty_log("article_ungrounded_qa_generator", "start", {"title": state.title})

    prompt = PROMPT_UNGROUNDED_QA_GENERATOR.format(
        content_type="Article",
        title=state.title or "<Untitled>",
        topic=state.topic,
        domain=state.domain_name,
        content_structure=f"Abstract: {state.abstract}",
    )
    constraints_text = f"\n\nConstraints: Produce {config.ungrounded_qa_min_items} to {config.ungrounded_qa_max_items} ungrounded QA pairs."

    qa_output: UngroundedQAOutput = llm.with_structured_output(
        UngroundedQAOutput
    ).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime=get_current_date())),
            HumanMessage(content=prompt + constraints_text),
        ]
    )

    for qa in qa_output.questions:
        qa.is_grounded = False
        qa.related_chapter_idx = None
        qa.related_section_idx = None

    max_ungrounded = config.ungrounded_qa_max_items
    ungrounded_questions = qa_output.questions[:max_ungrounded]
    if len(qa_output.questions) > max_ungrounded:
        pretty_log(
            "article_ungrounded_qa_generator",
            "truncate",
            {
                "requested": len(qa_output.questions),
                "kept": len(ungrounded_questions),
                "title": state.title,
            },
        )

    pretty_log(
        "article_ungrounded_qa_generator",
        "end",
        {"count": len(ungrounded_questions), "title": state.title},
    )
    return Command(
        update={
            "ungrounded_questions": ungrounded_questions,
            "pending_qa_tasks": -1,
        },
        goto="article_join_qa",
    )


def article_builder(state: ArticleState | dict):
    """Compile final Article object."""
    if isinstance(state, dict):
        state = ArticleState(**state)

    pretty_log("article_builder", "start", {"title": state.title})

    article = Article(
        title=state.title,
        topic=state.topic,
        abstract=state.abstract,
        sections=state.sections,
        grounded_questions=state.grounded_questions,
        ungrounded_questions=state.ungrounded_questions,
    )

    pretty_log(
        "article_builder",
        "end",
        {
            "sections": len(article.sections),
            "grounded_qa": len(article.grounded_questions),
            "ungrounded_qa": len(article.ungrounded_questions),
        },
    )
    return {"article": article}


# Build article subgraph
def build_article_subgraph():
    """Build and return the article generation subgraph."""
    article_subgraph_builder = StateGraph(ArticleState)
    article_subgraph_builder.add_node("article_planner", article_planner)
    article_subgraph_builder.add_node("article_section_writer", article_section_writer)
    article_subgraph_builder.add_node("article_join_sections", article_join_sections)
    article_subgraph_builder.add_node("article_qa_dispatch", article_qa_dispatch)
    article_subgraph_builder.add_node(
        "article_grounded_qa_generator", article_grounded_qa_generator
    )
    article_subgraph_builder.add_node(
        "article_ungrounded_qa_generator", article_ungrounded_qa_generator
    )
    article_subgraph_builder.add_node("article_join_qa", article_join_qa)
    article_subgraph_builder.add_node("article_builder", article_builder)

    article_subgraph_builder.add_edge(START, "article_planner")
    article_subgraph_builder.add_conditional_edges(
        "article_planner", assign_article_section_writers, ["article_section_writer"]
    )
    article_subgraph_builder.add_edge("article_section_writer", "article_join_sections")
    # article_join_sections uses Command() to route, no explicit edge needed
    article_subgraph_builder.add_conditional_edges(
        "article_qa_dispatch",
        route_article_qa,
        ["article_grounded_qa_generator", "article_ungrounded_qa_generator"],
    )
    article_subgraph_builder.add_edge(
        "article_grounded_qa_generator", "article_join_qa"
    )
    article_subgraph_builder.add_edge(
        "article_ungrounded_qa_generator", "article_join_qa"
    )
    # article_join_qa uses Command() to route to article_builder, no explicit edge needed
    article_subgraph_builder.add_edge("article_builder", END)

    return article_subgraph_builder.compile()
