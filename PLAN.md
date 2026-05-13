1. Persistent Candidate Database & Job-Based Organization
The gap: Every scrape lives in st.session_state and vanishes on refresh. A recruiter who just spent time reviewing 20 candidates has no way to come back tomorrow.
The addition: A proper data layer (SQLite to start, Postgres later) where candidates persist across sessions. Layer on top: job openings as organizing buckets — you create a "Senior Backend Engineer" job, add scraped profiles to it, and everything is grouped and searchable. This is the foundation that every other feature below builds on. Without it, you don't have a product — you have a command-line script with a UI.
Technical scope: ~3-4x more code. ORM models (Candidate, Job, Skill, etc.), CRUD endpoints, Streamlit state wired to database instead of session dicts, search/filter UI. Migrations, seed data for demo.
---
2. AI-Powered Candidate Scoring Against Job Descriptions
The gap: Right now you can ask the agent questions, but you have to know what to ask. The tool doesn't proactively tell you anything — it's a chatbot over a CSV.
The addition: Paste a job description, and the AI scores every candidate against it. It extracts requirements from the JD (years of experience, specific technologies, soft skills), compares each profile, and produces a ranked list with per-candidate explanations — "Candidate X scores 82/100: strong Rust experience matches, but lacks team-lead evidence." This is the feature that turns passive data browsing into an active decision-support system. It's the core value proposition a recruiter would pay for.
Technical scope: Prompt engineering chain — JD parsing → rubric extraction → per-candidate scoring → structured output. A new scoring UI component. Batch LLM calls with caching to control costs. This is the hardest AI work in the product.
---
3. Personalized Outreach Generator
The gap: You've identified strong candidates. Now you have to write cold messages. Recruiters spend hours on this and it's formulaic enough that AI does it well.
The addition: After scoring a candidate, generate a personalized LinkedIn InMail or email draft that references specific details from their profile — "I noticed your work on the Rust compiler project and the distributed systems talk you gave at XConf..." The recruiter reviews, edits, and either copies to clipboard or sends via email API. Also: template management so teams standardize outreach tone and structure per role type.
Technical scope: Prompt engineering for personalization quality (this is the hard part — generic "I was impressed by your background" is worthless). Email sending via SendGrid/Resend. Template CRUD with variable interpolation. A/B message variants. ~500-800 lines.
---
4. Hiring Pipeline & Kanban-Style Workflow
The gap: Once you start reaching out, you need to track what happens. Right now the tool has no concept of a candidate's status — they're just rows.
The addition: A pipeline board where candidates move through stages: New → Contacted → Phone Screen → Technical Interview → Onsite → Offer → Hired / Rejected. Each stage change logs a timestamp and note. This replaces the spreadsheet or Trello board nearly every recruiter maintains manually. Combined with scoring (#2), you get a single source of truth from discovery to close.
Technical scope: State machine for pipeline stages, drag-and-drop or select-based stage transitions in Streamlit, activity timeline per candidate, filtering by stage, pipeline health metrics (time-in-stage, conversion rates). ~600-1000 lines.
---
5. Multi-User Teams & Shared Workspaces
The gap: Hiring is a team sport — hiring manager, recruiter, interviewers all need access to the same candidate pool with different permissions.
The addition: User accounts with role-based access. A "workspace" owns jobs and candidates. Team members can be invited, leave internal comments/ratings on candidates (not visible to the candidate), see who reviewed what, and get notified when a candidate moves stages. Without this, the tool caps out at solo recruiters — with it, you can sell to agencies and internal talent teams.
Technical scope: Authentication (Supabase Auth or Clerk for speed), workspace/team models, permission system (owner/admin/viewer), commenting on candidates, activity feed, email notifications. ~800-1200 lines. This is the most infrastructure-heavy addition.
---
Suggested Build Order
These stack on each other intentionally: 1 → 2 → 3 → 4 → 5. You can't score candidates you haven't persisted. You can't generate outreach without scores. You can't track pipeline without a database. Teams (#5) can be parallel-tracked after the core flow is solid.
The smallest viable "real product" is 1 + 2 + a basic pipeline (1 stage). That's enough to charge $29/month. Everything after that increases the addressable market.
