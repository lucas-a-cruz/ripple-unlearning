"""Main domain generation graph."""

from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from src.config import config as gen_config
from src.models import Domain, TopicPlannerOutput
from src.prompts import PROMPT_TOPIC_PLANNER, SYSTEM_PROMPT
from src.state import DomainState
from src.utils import get_current_date, get_llm, pretty_log


def topic_planner(state: DomainState):
    """Plan the topics for the domain."""
    llm = get_llm()
    pretty_log("topic_planner", "start", {"domain": state.name})

    prompt = PROMPT_TOPIC_PLANNER.format(
        domain=state.name,
        description=state.description or "General domain",
        min_topics=gen_config.topics_min_items,
        max_topics=gen_config.topics_max_items,
    )

    response: TopicPlannerOutput = llm.with_structured_output(
        TopicPlannerOutput
    ).invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT.format(datetime=get_current_date())),
            HumanMessage(content=prompt),
        ]
    )

    num_topics = len(response.topics)
    pending_articles = num_topics * gen_config.articles_min_per_topic

    pretty_log("topic_planner", "end", {"topics": num_topics})
    return {
        "topics": response.topics,
        "pending_books": num_topics,
        "pending_articles": pending_articles,
    }


def domain_builder(state: DomainState | dict):
    """Compile final Domain object."""
    if isinstance(state, dict):
        state = DomainState(**state)

    pretty_log("domain_builder", "start", {"domain": state.name})

    domain = Domain(
        name=state.name,
        description=state.description,
        topics=state.topics,
        books=state.books,
        articles=state.articles,
    )

    pretty_log(
        "domain_builder",
        "end",
        {
            "topics": len(domain.topics),
            "books": len(domain.books),
            "articles": len(domain.articles),
        },
    )
    return {"domain": domain}


# Build main domain graph
def build_domain_graph():
    """Build and return the main domain generation graph."""
    # Build subgraphs
    from src.graphs.article_graph import build_article_subgraph
    from src.graphs.book_graph import build_book_subgraph

    book_subgraph = build_book_subgraph()
    article_subgraph = build_article_subgraph()

    def book_generator_worker(state: dict, config):
        """Generate a single book for a topic."""
        pretty_log("book_generator", "start", state)

        book_result = book_subgraph.invoke(
            {
                "domain_name": state["domain_name"],
                "topic": state["topic"],
                "topic_description": state.get("topic_description", ""),
            },
            config,
        )

        update: dict = {"pending_books": -1}

        book = book_result.get("book")
        if book:
            pretty_log(
                "book_generator",
                "end",
                {"topic": state["topic"], "book_created": True},
            )
            update["books"] = [book]

        pretty_log(
            "book_generator",
            "end",
            {"topic": state["topic"], "book_created": False},
        )
        return Command(update=update, goto="content_join")

    def article_generator_worker(state: dict, config):
        """Generate a single article for a topic."""
        pretty_log(
            "article_generator",
            "start",
            {"topic": state["topic"], "iteration": state.get("iteration")},
        )

        article_result = article_subgraph.invoke(
            {
                "domain_name": state["domain_name"],
                "topic": state["topic"],
                "topic_description": state.get("topic_description", ""),
            },
            config,
        )

        update: dict = {"pending_articles": -1}

        article = article_result.get("article")
        if article:
            pretty_log(
                "article_generator",
                "end",
                {"topic": state["topic"], "article_created": True},
            )
            update["articles"] = [article]

        pretty_log(
            "article_generator",
            "end",
            {"topic": state["topic"], "article_created": False},
        )
        return Command(update=update, goto="content_join")

    def content_join(state: DomainState | dict):
        """Check if all pending content tasks are complete."""
        if isinstance(state, dict):
            state = DomainState(**state)

        remaining_books = max(state.pending_books, 0)
        remaining_articles = max(state.pending_articles, 0)

        if remaining_books <= 0 and remaining_articles <= 0:
            pretty_log(
                "content_join",
                "complete",
                {"books": len(state.books), "articles": len(state.articles)},
            )
            return Command(goto="domain_builder")

        pretty_log(
            "content_join",
            "wait",
            {"pending_books": remaining_books, "pending_articles": remaining_articles},
        )
        return None

    def dispatch_content_generators(state: DomainState):
        """Launch book and article generation in parallel."""
        pretty_log(
            "content_dispatcher",
            "start",
            {"topics": len(state.topics)},
        )

        if not state.topics:
            pretty_log(
                "content_dispatcher",
                "end",
                {"dispatched": 1, "note": "no topics - skipping generation"},
            )
            return [Send("content_join", state.model_dump())]

        sends: list[Send] = []

        for topic in state.topics:
            sends.append(
                Send(
                    "book_generator_worker",
                    {
                        "domain_name": state.name,
                        "topic": topic.name,
                        "topic_description": topic.description,
                    },
                )
            )
            for iteration in range(gen_config.articles_min_per_topic):
                sends.append(
                    Send(
                        "article_generator_worker",
                        {
                            "domain_name": state.name,
                            "topic": topic.name,
                            "topic_description": topic.description,
                            "iteration": iteration + 1,
                        },
                    )
                )

        pretty_log(
            "content_dispatcher",
            "end",
            {"dispatched": len(sends)},
        )
        return sends

    domain_graph_builder = StateGraph(DomainState)
    domain_graph_builder.add_node("topic_planner", topic_planner)
    domain_graph_builder.add_node("book_generator_worker", book_generator_worker)
    domain_graph_builder.add_node("article_generator_worker", article_generator_worker)
    domain_graph_builder.add_node("content_join", content_join)
    domain_graph_builder.add_node("domain_builder", domain_builder)

    domain_graph_builder.add_edge(START, "topic_planner")
    domain_graph_builder.add_conditional_edges(
        "topic_planner",
        dispatch_content_generators,
        [
            "book_generator_worker",
            "article_generator_worker",
            "content_join",
        ],
    )
    domain_graph_builder.add_edge("book_generator_worker", "content_join")
    domain_graph_builder.add_edge("article_generator_worker", "content_join")
    domain_graph_builder.add_edge("content_join", "domain_builder")
    domain_graph_builder.add_edge("domain_builder", END)

    return domain_graph_builder.compile()
