# Autonomous LinkedIn Content Manager

This CrewAI application researches a niche, drafts and critiques a LinkedIn
post, optimizes it, then returns a publishing brief. It prepares content and
timing recommendations; it does not publish or schedule posts on LinkedIn.

## Setup

Use Python 3.10 or newer. From this folder, install dependencies and create a
local `.env` file:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Gemini and Serper credentials to `.env`, then run:

```powershell
python main.py
```

Enter the topic or niche when prompted. The five tasks run in order:
research → writing → critique → optimization → scheduling. Crew memory uses
Google embeddings authenticated with the configured Gemini API key.
If `gemini-3.8-flash` returns HTTP 429 (quota/rate limit) or HTTP 503
(temporary overload), the crew restarts with `gemini-3.1-flash-lite`. If both
models exhaust their quota, wait for Google's displayed reset time or review
the Google AI Studio project's quota and billing. API keys in one project
share that project's limits.
Posting-time recommendations are general guidance unless you provide audience
analytics or a target timezone.

## Dashboard

Launch the local dashboard with:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`), enter
a topic, and select **Generate publishing brief**. Completed stage outputs and
the final brief appear in the dashboard. Keep the dashboard on localhost unless
you add authentication before exposing it to a network.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), select **Create
   app**, choose the repository and branch, and set the app file path to
   `app.py`.
3. In the deployment's **Advanced settings** / **Secrets** field, add:

   ```toml
   Gemini_API_KEY = "your_gemini_api_key"
   SERPER_API_KEY = "your_serper_api_key"
   ```

   Do not commit credentials or `.streamlit/secrets.toml`. The dashboard reads
   these values from Streamlit secrets; local runs continue to use `.env`.
4. Deploy the app and use its Streamlit-provided URL.
