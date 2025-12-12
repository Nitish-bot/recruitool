from google import genai
import requests
import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

def scrape_linkedin_profile(profile_urls):
    token = st.secrets.get("BEARER_TOKEN")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    input = []
    for url in profile_urls:
        input.append({"url":url})

    print(input)
    data = {
        "input": input,
    }

    response = requests.post(
        "https://api.brightdata.com/datasets/v3/scrape?dataset_id=gd_l1viktl72bvl7bjuj0&notify=false&include_errors=true",
        headers=headers,
        json=data
    )
    response.raise_for_status()

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error scraping data: {response.text}")
        return None

def clean_linkedin_data(raw_data):
    """
    Cleans raw Bright Data LinkedIn JSON into a flat structure 
    optimized for LangChain Pandas Agents.
    
    Args:
        raw_data (list): A list of dictionaries (raw JSON profiles).
        
    Returns:
        pd.DataFrame: A clean DataFrame ready for analysis.
    """
    cleaned_profiles = []

    # Ensure input is a list, even if a single dict is passed
    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    for profile in raw_data:
        # 1. Basic Info
        full_name = f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip() or profile.get('name', 'Unknown')
        headline = profile.get('about', '')[:200] + "..." if profile.get('about') else "No headline"
        location = profile.get('city', profile.get('country_code', 'Unknown'))
        
        # 2. Current Role (Handle nested dict)
        current_company = profile.get('current_company', {}) or {}
        current_role = f"{current_company.get('name', 'Unknown')}"
        
        # 3. Education Summary (Convert list of dicts to a single string)
        education_entries = profile.get('education') or []
        if isinstance(education_entries, list):
            edu_str = "; ".join([
                f"{e.get('title', 'Unknown Degree')} at {e.get('start_year', '')}-{e.get('end_year', '')}" 
                for e in education_entries
            ])
        else:
            edu_str = "No Education Listed"

        # 4. Experience / Projects Summary (Crucial for Resume Screening)
        # Note: Your JSON had 'experience': None but detailed 'projects'. 
        # We aggregate both into a single 'Background' field for the LLM.
        
        background_context = []
        
        # Add Experience if it exists
        experience_entries = profile.get('experience') or []
        if isinstance(experience_entries, list):
            for exp in experience_entries:
                background_context.append(f"Role: {exp.get('title')} at {exp.get('company')}")

        # Add Projects (Profile 2 had rich project data)
        projects = profile.get('projects') or []
        if isinstance(projects, list):
            for proj in projects:
                background_context.append(f"Project: {proj.get('title')} ({proj.get('description', '')[:50]}...)")

        # Add Certifications (Profile 1 had these)
        certs = profile.get('certifications') or []
        if isinstance(certs, list):
            for c in certs:
                background_context.append(f"Cert: {c.get('title')} by {c.get('subtitle')}")
        
        # Join all background info into one text block for the LLM
        full_background_text = " | ".join(background_context) if background_context else "No detailed experience/projects found."

        # 5. Posts/Content (Good for culture fit analysis)
        posts = profile.get('posts') or []
        recent_posts = "; ".join([p.get('title', '') for p in posts[:3]])

        # Create the flat object
        cleaned_profiles.append({
            "Name": full_name,
            "Location": location,
            "Current Company": current_role,
            "Education": edu_str,
            "Professional Background": full_background_text,
            "Recent Content": recent_posts,
            "Profile URL": profile.get('url') or profile.get('input_url')
        })

    return pd.DataFrame(cleaned_profiles)

# --- STREAMLIT UI START ---
st.set_page_config(page_title="Recruiter AI", layout="wide")
st.title("🤖 LinkedIn Profile Analyzer")

# 1. Sidebar: Inputs
with st.sidebar:
    st.header("Configuration")
    
    gemini_key = st.secrets.get("GEMINI_API_KEY", None)
    if not gemini_key:
        gemini_key = st.text_input("Enter Gemini API Key", type="password")
        
    # Input for Profile URLs
    url_input = st.text_area("Enter LinkedIn URL", 
                             value="https://www.linkedin.com/in/maham-yousaf-230b33359/")
    
    fetch_button = st.button("Fetch & Analyze Profiles")

# 2. Main Area: Data & Chat
if fetch_button and gemini_key:
    if "BEARER_TOKEN" not in st.secrets:
        st.error("Bright Data Bearer Token is missing from environment variables!")
    else:
        with st.spinner('Scraping LinkedIn data via Bright Data...'):
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
            raw_data = scrape_linkedin_profile(urls)
            
        if raw_data:
            # Clean the data
            df = clean_linkedin_data(raw_data)
            
            # Save data to session state so it doesn't vanish when you chat
            st.session_state['df'] = df
            st.success("Profiles loaded successfully!")

# 3. Persistent View (Data + Chat)
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # SHOW DATA
    st.subheader("Candidate Database")
    st.dataframe(df)

    # CHAT INTERFACE
    st.divider()
    st.subheader("Chat with the Candidates")

    # Initialize LLM Agent
    llm = ChatGoogleGenerativeAI(google_api_key=gemini_key, model="gemini-2.5-flash-lite")
    agent = create_pandas_dataframe_agent(
        llm, 
        df,
        allow_dangerous_code=True,
        verbose=True, 
    )

    # Chat History Container
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask: 'Who has the most experience?' or 'Compare their skills'"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = agent.invoke(prompt)
                    output_text = response['output']
                    st.markdown(output_text)
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                except Exception as e:
                    st.error(f"Error generating response: {e}")