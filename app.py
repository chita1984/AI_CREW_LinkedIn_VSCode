"""Local Streamlit dashboard for the LinkedIn content crew."""

from __future__ import annotations

import logging
import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from main import (
    FALLBACK_MODEL_NAME,
    MODEL_NAME,
    REQUIRED_ENV_VARS,
    STAGES,
    GeminiQuotaExceededError,
    run_pipeline,
)


logger = logging.getLogger("linkedin_content_dashboard")


def load_streamlit_secrets() -> None:
    """Make configured Streamlit secrets available to the shared pipeline."""
    try:
        secrets = st.secrets
    except StreamlitSecretNotFoundError:
        return

    for name in REQUIRED_ENV_VARS:
        if name not in os.environ and name in secrets:
            os.environ[name] = str(secrets[name])


st.set_page_config(
    page_title="LinkedIn Content Manager",
    page_icon="in",
    layout="wide",
)

st.title("Autonomous LinkedIn Content Manager")
st.caption(
    "Research, draft, critique, optimize, and prepare a LinkedIn post "
    "for publishing. This dashboard does not publish to LinkedIn."
)

with st.sidebar:
    st.subheader("Pipeline")
    st.markdown("1. Research")
    st.markdown("2. Write")
    st.markdown("3. Critique")
    st.markdown("4. Optimize")
    st.markdown("5. Schedule")
    st.divider()
    st.caption(
        "API credentials are read from environment variables or Streamlit "
        "secrets and are not displayed in the dashboard."
    )

st.session_state.setdefault("stage_outputs", {})
st.session_state.setdefault("publishing_brief", "")
st.session_state.setdefault("used_model", "")
st.session_state.setdefault("topic", "")

with st.form("linkedin_content_form"):
    topic = st.text_area(
        "Topic or niche",
        value=st.session_state["topic"],
        placeholder="For example: AI agents in enterprise IT support",
        height=90,
    )
    submitted = st.form_submit_button(
        "Generate publishing brief",
        type="primary",
    )

if submitted:
    if not topic.strip():
        st.error("Enter a topic or niche to start the pipeline.")
    else:
        st.session_state["topic"] = topic.strip()
        st.session_state["stage_outputs"] = {}
        st.session_state["publishing_brief"] = ""
        st.session_state["used_model"] = ""

        progress = st.progress(0, text="Starting the five-agent pipeline...")
        current_stage = st.empty()

        def display_stage(stage: str, content: str) -> None:
            """Record task output and update the live progress indicator."""
            st.session_state["stage_outputs"][stage] = content
            completed = STAGES.index(stage) + 1
            progress.progress(
                completed / len(STAGES),
                text=f"Completed {completed} of {len(STAGES)} stages: {stage}",
            )
            current_stage.success(f"{stage} complete")

        try:
            load_streamlit_secrets()
            with st.spinner(
                "The agents are working. Research may take a few minutes..."
            ):
                result, used_model = run_pipeline(
                    topic,
                    stage_handler=display_stage,
                )
            st.session_state["publishing_brief"] = result.raw
            st.session_state["used_model"] = used_model
            st.success("Publishing brief is ready.")
        except ValueError as exc:
            st.error(str(exc))
        except GeminiQuotaExceededError as exc:
            st.error("Gemini request quota is exhausted.")
            st.info(str(exc))
            st.caption(
                "Changing API keys will not increase the quota if the new key "
                "belongs to the same Google Cloud project."
            )
        except Exception as exc:
            logger.exception("Dashboard pipeline run failed")
            st.error(
                "The pipeline did not complete. Check your .env API keys, "
                "network connection, and the server terminal for details."
            )
            if getattr(exc, "code", None) == 429:
                st.info(
                    "Gemini rate or quota limit reached. Wait for the reset "
                    "time in the provider response or review the project's "
                    "quota and billing in Google AI Studio."
                )
            else:
                st.caption(f"Error: {type(exc).__name__}: {exc}")

brief = st.session_state["publishing_brief"]
used_model = st.session_state["used_model"]
if brief:
    st.subheader("Final publishing brief")
    if used_model == FALLBACK_MODEL_NAME:
        st.info(
            f"The primary model was unavailable, so this run used "
            f"`{FALLBACK_MODEL_NAME}`."
        )
    else:
        st.caption(f"Model: `{MODEL_NAME}`")
    st.markdown(brief)

stage_outputs = st.session_state["stage_outputs"]
if stage_outputs:
    st.subheader("Pipeline stages")
    for index, stage in enumerate(STAGES, start=1):
        if stage in stage_outputs:
            with st.expander(f"{index}. {stage}", expanded=False):
                st.markdown(stage_outputs[stage])
