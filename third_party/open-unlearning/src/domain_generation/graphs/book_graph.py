"""Book generation subgraph."""

from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from src.config import config
from src.models import (Book, BookPlannerOutput, Chapter, GroundedQAOutput,
                        UngroundedQAOutput)
from src.prompts import (PROMPT_BOOK_PLANNER, PROMPT_CHAPTER_WRITER,
                         PROMPT_GROUNDED_QA_GENERATOR,
                         PROMPT_UNGROUNDED_QA_GENERATOR, SYSTEM_PROMPT)
from src.state import BookState, ChapterWriterState
from src.utils import get_current_date, get_llm, pretty_log


def book_planner(state: BookState):
    """Create a plan (TOC) for writing the book."""
    llm = get_llm()
    pretty_log("book_planner", "start", {"topic": state.topic})

    prompt = PROMPT_BOOK_PLANNER.format(
        domain=state.domain_name,
        topic=state.topic,
        topic_description=state.topic_description,
    )
    constraints_text = (
        "\n\nConstraints:"
        f"\n- Produce between {config.toc_min_items} and {config.toc_max_items} chapters."
        f"\n- Each chapter should contain {config.sections_min_per_chapter} to {config.sections_max_per_chapter} sections."
    )

    response: BookPlannerOutput = llm.with_structured_output(BookPlannerOutput).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime=get_current_date())),
            HumanMessage(content=prompt + constraints_text),
        ]
    )

    num_chapters = len(response.table_of_contents)
    pretty_log(
        "book_planner", "end", {"chapters": num_chapters, "title": response.title}
    )
    return {
        "title": response.title,
        "table_of_contents": response.table_of_contents,
        "pending_chapters": num_chapters,
    }


def chapter_writer(state: ChapterWriterState | dict):
    """Write a single chapter."""
    llm = get_llm()
    pretty_log("chapter_writer", "start", state)

    if isinstance(state, dict):
        state = ChapterWriterState(**state)

    prompt = PROMPT_CHAPTER_WRITER.format(
        domain=state.domain_name,
        topic=state.topic,
        chapter_title=state.chapter_title,
        summary=state.summary,
        chapters_titles=", ".join(state.chapter_titles),
    )
    constraints_text = f"\n\nConstraints: Produce {config.sections_min_per_chapter} to {config.sections_max_per_chapter} sections."

    response: Chapter = llm.with_structured_output(Chapter).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime=get_current_date())),
            HumanMessage(content=prompt + constraints_text),
        ]
    )

    pretty_log("chapter_writer", "end", {"chapter": response.title})
    return Command(
        update={
            "chapters": [response],
            "pending_chapters": -1,
        },
        goto="book_join_chapters",
    )


def assign_chapter_writers(state: BookState):
    """Dispatch parallel chapter writing tasks."""
    pretty_log(
        "assign_chapter_writers",
        "start",
        {"num_chapters": len(state.table_of_contents)},
    )

    chapter_titles = [
        c.chapter_title for c in sorted(state.table_of_contents, key=lambda x: x.idx)
    ]

    sends = [
        Send(
            "chapter_writer",
            {
                "domain_name": state.domain_name,
                "topic": state.topic,
                "chapter_title": toc_entry.chapter_title,
                "summary": toc_entry.summary,
                "chapter_titles": chapter_titles,
                "idx": toc_entry.idx,
            },
        )
        for toc_entry in state.table_of_contents
    ]

    return sends


def book_join_chapters(state: BookState | dict):
    """Wait for all chapter writers to finish before dispatching QA tasks."""
    if isinstance(state, dict):
        state = BookState(**state)

    remaining = max(state.pending_chapters, 0)

    if remaining <= 0:
        pretty_log(
            "book_join_chapters",
            "chapters_complete",
            {"title": state.title, "chapters": len(state.chapters)},
        )
        if not state.chapters:
            return Command(goto="book_builder")
        return Command(
            update={"pending_qa_tasks": 2},
            goto="book_qa_dispatch",
        )

    pretty_log(
        "book_join_chapters",
        "waiting",
        {"pending_chapters": remaining, "title": state.title},
    )
    return None


def book_qa_dispatch(state: BookState | dict):
    """Entry node before dispatching QA generation tasks."""
    if isinstance(state, dict):
        state = BookState(**state)

    pretty_log(
        "book_qa_dispatch",
        "start",
        {"title": state.title, "chapters": len(state.chapters)},
    )
    return {}


def route_book_qa(state: BookState | dict):
    """Conditional edge handler for QA fan-out."""
    if isinstance(state, dict):
        state = BookState(**state)

    payload = state.model_dump()
    sends = [
        Send("book_grounded_qa_generator", payload),
        Send("book_ungrounded_qa_generator", payload),
    ]

    pretty_log("book_qa_dispatch", "dispatch", {"jobs": len(sends)})
    return sends


def book_join_qa(state: BookState | dict):
    """Join QA generation tasks before building the book."""
    if isinstance(state, dict):
        state = BookState(**state)

    remaining = max(state.pending_qa_tasks, 0)

    if remaining <= 0:
        pretty_log(
            "book_join_qa",
            "qa_complete",
            {
                "grounded": len(state.grounded_questions),
                "ungrounded": len(state.ungrounded_questions),
            },
        )
        return Command(goto="book_builder")

    pretty_log(
        "book_join_qa",
        "waiting",
        {"pending_qa_tasks": remaining, "title": state.title},
    )
    return None


def book_grounded_qa_generator(state: BookState | dict):
    """Generate grounded QA pairs from book content."""
    if isinstance(state, dict):
        state = BookState(**state)

    llm = get_llm()
    pretty_log("book_grounded_qa_generator", "start", {"title": state.title})

    toc_str = "\n".join(
        f"[{e.idx}] {e.chapter_title}: {e.summary}"
        for e in sorted(state.table_of_contents, key=lambda x: x.idx)
    )
    chapters_str = "\n\n".join(
        f"# [{c.idx}] {c.title}\n"
        + "\n".join(f"- [{s.idx}] {s.name}: {s.content[:300]}..." for s in c.sections)
        for c in sorted(state.chapters, key=lambda x: x.idx)
    )

    prompt = PROMPT_GROUNDED_QA_GENERATOR.format(
        content_type="Book",
        title=state.title or "<Untitled>",
        topic=state.topic,
        content_structure=f"TOC:\n{toc_str}\n\nChapters:\n{chapters_str}",
    )
    constraints_text = f"\n\nConstraints: Produce {config.grounded_qa_min_items} to {config.grounded_qa_max_items} grounded QA pairs."

    qa_output: GroundedQAOutput = llm.with_structured_output(GroundedQAOutput).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime=get_current_date())),
            HumanMessage(content=prompt + constraints_text),
        ]
    )

    # Ensure is_grounded=True for all and clip to configured maximum
    for qa in qa_output.questions:
        qa.is_grounded = True

    max_grounded = config.grounded_qa_max_items
    grounded_questions = qa_output.questions[:max_grounded]
    if len(qa_output.questions) > max_grounded:
        pretty_log(
            "book_grounded_qa_generator",
            "truncate",
            {
                "requested": len(qa_output.questions),
                "kept": len(grounded_questions),
                "title": state.title,
            },
        )

    pretty_log(
        "book_grounded_qa_generator",
        "end",
        {"count": len(grounded_questions), "title": state.title},
    )
    return Command(
        update={
            "grounded_questions": grounded_questions,
            "pending_qa_tasks": -1,
        },
        goto="book_join_qa",
    )


def book_ungrounded_qa_generator(state: BookState | dict):
    """Generate ungrounded (control) QA pairs."""
    if isinstance(state, dict):
        state = BookState(**state)

    llm = get_llm()
    pretty_log("book_ungrounded_qa_generator", "start", {"title": state.title})

    toc_str = "\n".join(
        f"[{e.idx}] {e.chapter_title}: {e.summary}"
        for e in sorted(state.table_of_contents, key=lambda x: x.idx)
    )

    prompt = PROMPT_UNGROUNDED_QA_GENERATOR.format(
        content_type="Book",
        title=state.title or "<Untitled>",
        topic=state.topic,
        domain=state.domain_name,
        content_structure=f"TOC:\n{toc_str}",
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

    # Ensure is_grounded=False for all
    for qa in qa_output.questions:
        qa.is_grounded = False
        qa.related_chapter_idx = None
        qa.related_section_idx = None

    max_ungrounded = config.ungrounded_qa_max_items
    ungrounded_questions = qa_output.questions[:max_ungrounded]
    if len(qa_output.questions) > max_ungrounded:
        pretty_log(
            "book_ungrounded_qa_generator",
            "truncate",
            {
                "requested": len(qa_output.questions),
                "kept": len(ungrounded_questions),
                "title": state.title,
            },
        )

    pretty_log(
        "book_ungrounded_qa_generator",
        "end",
        {"count": len(ungrounded_questions), "title": state.title},
    )
    return Command(
        update={
            "ungrounded_questions": ungrounded_questions,
            "pending_qa_tasks": -1,
        },
        goto="book_join_qa",
    )


def book_builder(state: BookState | dict):
    """Compile final Book object."""
    if isinstance(state, dict):
        state = BookState(**state)

    pretty_log("book_builder", "start", {"title": state.title})

    book = Book(
        title=state.title,
        topic=state.topic,
        table_of_contents=state.table_of_contents,
        chapters=state.chapters,
        grounded_questions=state.grounded_questions,
        ungrounded_questions=state.ungrounded_questions,
    )

    pretty_log(
        "book_builder",
        "end",
        {
            "chapters": len(book.chapters),
            "grounded_qa": len(book.grounded_questions),
            "ungrounded_qa": len(book.ungrounded_questions),
        },
    )
    return {"book": book}


# Build book subgraph
def build_book_subgraph():
    """Build and return the book generation subgraph."""
    book_subgraph_builder = StateGraph(BookState)
    book_subgraph_builder.add_node("book_planner", book_planner)
    book_subgraph_builder.add_node("chapter_writer", chapter_writer)
    book_subgraph_builder.add_node("book_join_chapters", book_join_chapters)
    book_subgraph_builder.add_node("book_qa_dispatch", book_qa_dispatch)
    book_subgraph_builder.add_node(
        "book_grounded_qa_generator", book_grounded_qa_generator
    )
    book_subgraph_builder.add_node(
        "book_ungrounded_qa_generator", book_ungrounded_qa_generator
    )
    book_subgraph_builder.add_node("book_join_qa", book_join_qa)
    book_subgraph_builder.add_node("book_builder", book_builder)

    book_subgraph_builder.add_edge(START, "book_planner")
    book_subgraph_builder.add_conditional_edges(
        "book_planner", assign_chapter_writers, ["chapter_writer"]
    )
    book_subgraph_builder.add_edge("chapter_writer", "book_join_chapters")
    # book_join_chapters uses Command() to route, no explicit edge needed
    book_subgraph_builder.add_conditional_edges(
        "book_qa_dispatch",
        route_book_qa,
        ["book_grounded_qa_generator", "book_ungrounded_qa_generator"],
    )
    book_subgraph_builder.add_edge("book_grounded_qa_generator", "book_join_qa")
    book_subgraph_builder.add_edge("book_ungrounded_qa_generator", "book_join_qa")
    # book_join_qa uses Command() to route to book_builder, no explicit edge needed
    book_subgraph_builder.add_edge("book_builder", END)

    return book_subgraph_builder.compile()
