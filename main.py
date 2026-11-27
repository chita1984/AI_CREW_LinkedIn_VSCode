"""Autonomous LinkedIn content manager built with a sequential CrewAI crew."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from dotenv import load_dotenv


# Use one LLM configuration consistently across all five agents. The API key is
# passed explicitly so the requested Gemini_API_KEY variable works as-is.
MODEL_NAME = "gemini/gemini-3.8-flash"
FALLBACK_MODEL_NAME = "gemini/gemini-3.1-flash-lite"
REQUIRED_ENV_VARS = ("Gemini_API_KEY", "SERPER_API_KEY")
STAGES = ("Research", "Writing", "Critique", "Optimization", "Scheduling")
StageHandler = Callable[[str, str], None]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("linkedin_content_manager")


class GeminiQuotaExceededError(RuntimeError):
    """Raised when Gemini quota prevents both primary and fallback runs."""


def _provider_http_status(error: BaseException) -> int | None:
    """Find an HTTP status on a provider error or its chained exceptions."""
    pending = [error]
    visited: set[int] = set()

    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))

        for attribute in ("code", "status_code"):
            status = getattr(current, attribute, None)
            if isinstance(status, int):
                return status

        cause = current.__cause__
        context = current.__context__
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)

    return None


def _can_fallback(error: BaseException) -> bool:
    """Retry with the alternate model for temporary outages or exhausted quota."""
    return _provider_http_status(error) in (429, 503)


def _read_api_key() -> str:
    """Load credentials from .env and fail early with actionable messages."""
    load_dotenv()

    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            f"Missing required environment variable(s): {names}. "
            "Copy .env.example to .env and add your API keys."
        )

    gemini_api_key = os.environ["Gemini_API_KEY"]
    return gemini_api_key


def _stage_callback(
    stage: str,
    stage_handler: StageHandler | None = None,
) -> Callable[[Any], None]:
    """Report each task result to the console or a dashboard callback."""
    def print_output(task_output: Any) -> None:
        content = getattr(task_output, "raw", str(task_output)).strip()
        if stage_handler is not None:
            stage_handler(stage, content)
            return

        print(f"\n{'=' * 20} {stage.upper()} {'=' * 20}")
        print(content)
        print(f"{'=' * (42 + len(stage))}\n", flush=True)

    return print_output


def create_crew(
    gemini_api_key: str,
    model_name: str = MODEL_NAME,
    stage_handler: StageHandler | None = None,
) -> Crew:
    """Create the five-agent, five-task LinkedIn content pipeline."""
    llm = LLM(
        model=model_name,
        api_key=gemini_api_key,
        temperature=0.7,
    )

    # Web research tools are intentionally restricted to the researcher.
    serper_search = SerperDevTool()
    website_scraper = ScrapeWebsiteTool()

    # Agent definitions: the four downstream agents are LLM-only.
    trend_researcher = Agent(
        role="LinkedIn Trend Researcher",
        goal="Research latest trending topics, hashtags, and content themes for a given niche",
        backstory=(
            "Expert social media researcher who monitors LinkedIn trends, viral "
            "posts, and industry news; knows what drives engagement on LinkedIn."
        ),
        tools=[serper_search, website_scraper],
        llm=llm,
        verbose=True,
        max_retry_limit=1,
    )
    content_writer = Agent(
        role="LinkedIn Content Writer",
        goal="Write engaging, high-quality LinkedIn posts based on research provided",
        backstory=(
            "Seasoned LinkedIn ghostwriter for industry leaders; expert in the "
            "LinkedIn algorithm, hook writing, storytelling, CTA placement, and "
            "conversational professional tone."
        ),
        llm=llm,
        verbose=True,
        max_retry_limit=1,
    )
    content_critic = Agent(
        role="Content Quality Critic",
        goal=(
            "Review LinkedIn posts and provide detailed constructive feedback "
            "on engagement potential, tone, structure, clarity, hook strength, "
            "and CTA effectiveness"
        ),
        backstory=(
            "Harsh but fair editor with thousands of LinkedIn post reviews; "
            "distinguishes posts that get 10 likes from those with 10k+ "
            "impressions and delivers specific, actionable feedback."
        ),
        llm=llm,
        verbose=True,
        max_retry_limit=1,
    )
    content_optimizer = Agent(
        role="LinkedIn Post Optimizer",
        goal="Rewrite posts incorporating critic feedback to maximize LinkedIn engagement",
        backstory=(
            "LinkedIn growth expert and copywriter; master of formatting "
            "(short lines, strategic breaks, emoji usage, hashtag optimization, "
            "hook patterns, and mobile readability)."
        ),
        llm=llm,
        verbose=True,
        max_retry_limit=1,
    )
    scheduling_agent = Agent(
        role="LinkedIn Publishing Strategist",
        goal=(
            "Determine optimal posting time, finalize formatting with hashtags, "
            "and create publishing-ready output with scheduling recommendations"
        ),
        backstory=(
            "LinkedIn analytics expert who understands optimal posting times "
            "by industry, audience timezone, and day of week."
        ),
        llm=llm,
        verbose=True,
        max_retry_limit=1,
    )

    # Task definitions: explicit context links make each handoff unambiguous.
    research_task = Task(
        description=(
            "Research latest trends, viral content patterns, and hot topics on "
            "LinkedIn for the niche: {topic}. Identify 3–5 trending angles, "
            "relevant hashtags, and content hooks currently performing well. "
            "Use recent web sources, include their URLs and publication dates "
            "where available, and distinguish sourced findings from hypotheses."
        ),
        expected_output=(
            "Structured research brief with trending topics, suggested angles, "
            "top-performing hashtags, content hook ideas, and source URLs."
        ),
        agent=trend_researcher,
        callback=_stage_callback(STAGES[0], stage_handler),
    )
    writing_task = Task(
        description=(
            "Using the research brief, write a compelling LinkedIn post about "
            "{topic}. Include a strong hook in the first two lines, a "
            "storytelling or value-driven body, a clear CTA, and 150–300 words. "
            "Use relevant trending angles and hooks from the research."
        ),
        expected_output=(
            "Complete LinkedIn post draft with hook, body, CTA, and suggested hashtags."
        ),
        agent=content_writer,
        context=[research_task],
        callback=_stage_callback(STAGES[1], stage_handler),
    )
    critique_task = Task(
        description=(
            'Critically review the LinkedIn post draft. Evaluate hook strength '
            '(will people click "see more"?), storytelling quality, engagement '
            "potential, CTA effectiveness, tone consistency, LinkedIn "
            "formatting, and viral potential. Provide a score out of 10 and "
            "specific improvement suggestions."
        ),
        expected_output=(
            "Detailed critique with scores, strengths, weaknesses, and specific "
            "actionable improvement suggestions."
        ),
        agent=content_critic,
        context=[writing_task],
        callback=_stage_callback(STAGES[2], stage_handler),
    )
    optimization_task = Task(
        description=(
            "Take the original LinkedIn post and the critic's feedback. Rewrite "
            "the post incorporating the feedback. Improve the hook, tighten "
            "copy, optimize formatting with short lines, line breaks, and "
            "strategic emoji, strengthen the CTA, and optimize hashtags. "
            "Preserve accurate claims and do not invent facts."
        ),
        expected_output=(
            "Final, polished, publish-ready LinkedIn post with optimized "
            "formatting, hashtags, and CTA."
        ),
        agent=content_optimizer,
        context=[writing_task, critique_task],
        callback=_stage_callback(STAGES[3], stage_handler),
    )
    scheduling_task = Task(
        description=(
            "Analyze the final post content and target audience for {topic}. "
            "Recommend a best day and time to publish, including a timezone "
            "and noting that the recommendation is a general heuristic unless "
            "audience analytics are provided. Include the final formatted post "
            "ready for LinkedIn copy-paste, a brief hashtag strategy, and "
            "first-hour engagement tips. Do not claim to schedule or publish "
            "the post."
        ),
        expected_output=(
            "Complete publishing brief with recommended posting time, final "
            "formatted post, hashtag list, and first-hour engagement strategy."
        ),
        agent=scheduling_agent,
        context=[optimization_task],
        callback=_stage_callback(STAGES[4], stage_handler),
    )

    # Sequential process enforces the research → writing → critique →
    # optimization → scheduling order. Google-backed embeddings keep shared
    # memory usable without requiring an unrelated OpenAI API key.
    return Crew(
        agents=[
            trend_researcher,
            content_writer,
            content_critic,
            content_optimizer,
            scheduling_agent,
        ],
        tasks=[
            research_task,
            writing_task,
            critique_task,
            optimization_task,
            scheduling_task,
        ],
        process=Process.sequential,
        verbose=True,
        memory=True,
        embedder={
            "provider": "google-vertex",
            "config": {
                "model_name": "gemini-embedding-001",
                "api_key": gemini_api_key,
            },
        },
    )


def run_pipeline(
    topic: str,
    stage_handler: StageHandler | None = None,
) -> tuple[Any, str]:
    """Run the five stages and fail over on Gemini overload or quota limits."""
    normalized_topic = topic.strip()
    if not normalized_topic:
        raise ValueError("The topic or niche cannot be empty.")

    gemini_api_key = _read_api_key()
    inputs = {"topic": normalized_topic}
    try:
        result = create_crew(
            gemini_api_key,
            stage_handler=stage_handler,
        ).kickoff(inputs=inputs)
        return result, MODEL_NAME
    except Exception as exc:
        status = _provider_http_status(exc)
        if status not in (429, 503):
            raise

        logger.warning(
            "%s returned HTTP %s; retrying the complete pipeline with %s.",
            MODEL_NAME,
            status,
            FALLBACK_MODEL_NAME,
        )
        if stage_handler is None:
            print(
                f"\nPrimary model unavailable (HTTP {status}); retrying with "
                f"{FALLBACK_MODEL_NAME}.\n",
                flush=True,
            )
        try:
            result = create_crew(
                gemini_api_key,
                model_name=FALLBACK_MODEL_NAME,
                stage_handler=stage_handler,
            ).kickoff(inputs=inputs)
        except Exception as fallback_error:
            if _provider_http_status(fallback_error) == 429:
                raise GeminiQuotaExceededError(
                    "Gemini has exhausted the request quota for both the "
                    "primary and fallback models. Wait until the reset time "
                    "shown in the error, or review the Google AI Studio project "
                    "quota and billing. API keys in the same project share "
                    "that project's quota."
                ) from fallback_error
            raise
        return result, FALLBACK_MODEL_NAME


def main() -> int:
    """Read the niche, run the crew, and report failures without hiding them."""
    try:
        # Keep Unicode stage labels and generated post formatting printable on
        # Windows consoles that otherwise use a legacy code page.
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")

        topic = input("Enter a LinkedIn topic or niche: ").strip()

        print(
            "\nStarting LinkedIn content pipeline "
            "(Research → Writing → Critique → Optimization → Scheduling)...\n",
            flush=True,
        )
        result, _used_model = run_pipeline(topic)

        print("\n" + "=" * 18 + " FINAL PUBLISHING BRIEF " + "=" * 18)
        print(result.raw.strip())
        print("=" * 62)
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nRun cancelled.", file=sys.stderr)
        return 130
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    except Exception:
        # CrewAI/LiteLLM errors can originate from credentials, network issues,
        # rate limits, or provider responses; retain the traceback for diagnosis.
        logger.exception(
            "The LinkedIn content pipeline failed. Check API keys, network "
            "connectivity, provider/model availability, and rate limits."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
