import requests
import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

def scrape_linkedin_profile(profile_urls):
    token = st.secrets.get("BEARER_TOKEN")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    json = {
        "input": [{"url": url} for url in profile_urls]
    }

    try:
        response = requests.post(
            "https://api.brightdata.com/datasets/v3/scrape?dataset_id=gd_l1viktl72bvl7bjuj0&include_errors=true",
            headers=headers,
            json=json
        )
        
        if response.status_code == 200:
            return {"status": "complete", "data": response.json()}
            
        elif "snapshot_id" in response.json():
            snapshot_id = response.json()['snapshot_id']
            return {"status": "pending", "id": snapshot_id}
            
        else:
            st.error(f"Error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

def poll_snapshot(snapshot_id):
    """Checks if the data is ready for a specific snapshot ID"""
    token = st.secrets.get("BEARER_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Bright Data URL to fetch specific snapshot data
    # NOTE: Verify this endpoint in your Bright Data dashboard, it usually looks like this:
    url = f"https://api.brightdata.com/datasets/v3/log/{snapshot_id}"
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # Check if response is "Processing" or actual data
        data = response.json()
        if isinstance(data, dict) and data.get('status') == 'running':
             return {"status": "pending"}
        return {"status": "complete", "data": data}
    elif response.status_code == 202:
        return {"status": "pending"}
    else:
        st.error(f"Polling Error: {response.text}")
        return {"status": "error"}
    
def clean_linkedin_data(raw_data):
    cleaned_profiles = []

    # Ensure input is a list, even if a single dict is passed
    if isinstance(raw_data, dict) and "data" in raw_data:
        raw_data = raw_data["data"]

    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    for profile in raw_data:
        full_name = f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip() or profile.get('name', 'Unknown')
        headline = profile.get('about', '')[:200] + "..." if profile.get('about') else "No headline"
        location = profile.get('city', profile.get('country_code', 'Unknown'))
        
        current_company = profile.get('current_company', {}) or {}
        current_role = f"{current_company.get('name', 'Unknown')}"
        
        education_entries = profile.get('education') or []
        if isinstance(education_entries, list):
            edu_str = "; ".join([
                f"{e.get('title', 'Unknown Degree')} at {e.get('start_year', '')}-{e.get('end_year', '')}" 
                for e in education_entries
            ])
        else:
            edu_str = "No Education Listed"
        
        background_context = []
        
        experience_entries = profile.get('experience') or []
        if isinstance(experience_entries, list):
            for exp in experience_entries:
                background_context.append(f"Role: {exp.get('title')} at {exp.get('company')}")

        projects = profile.get('projects') or []
        if isinstance(projects, list):
            for proj in projects:
                background_context.append(f"Project: {proj.get('title')} ({proj.get('description', '')[:50]}...)")

        certs = profile.get('certifications') or []
        if isinstance(certs, list):
            for c in certs:
                background_context.append(f"Cert: {c.get('title')} by {c.get('subtitle')}")
        
        full_background_text = " | ".join(background_context) if background_context else "No detailed experience/projects found."
        
        posts = profile.get('posts') or []
        recent_posts = "; ".join([p.get('title', '') for p in posts[:3]])

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

if 'snapshot_id' not in st.session_state:
    st.session_state['snapshot_id'] = None

# 1. Sidebar: Inputs
with st.sidebar:
    st.header("Configuration")
    
    gemini_key = st.secrets.get("GEMINI_API_KEY", None)
    if not gemini_key:
        gemini_key = st.text_input("Enter Gemini API Key", type="password")
        
    # Input for Profile URLs
    url_input = st.text_area("Enter LinkedIn URL", 
                             value="https://www.linkedin.com/in/maham-yousaf-230b33359/")
    
    # LOGIC SWITCH: Show "Fetch" OR "Check Status"
    if st.session_state['snapshot_id'] is None:
        fetch_btn = st.button("🚀 Start Scraping")
        check_btn = False
    else:
        st.info(f"Job Running... ID: {st.session_state['snapshot_id']}")
        check_btn = st.button("🔄 Check Status / Retry")
        fetch_btn = False
        if st.button("Cancel / Clear"):
            st.session_state['snapshot_id'] = None
            st.rerun()

# 2. Main Area: Data & Chat
if fetch_btn and gemini_key:
    if "BEARER_TOKEN" not in st.secrets:
        st.error("Bright Data Bearer Token is missing from environment variables!")
    else:
        with st.spinner('Scraping LinkedIn data via Bright Data...'):
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
            raw_data = scrape_linkedin_profile(urls)

        if raw_data:
            if raw_data['status'] == 'complete':
                df = clean_linkedin_data(raw_data)
                
                st.session_state['df'] = df
                st.success("Profiles loaded successfully!")

        elif raw_data['status'] == 'pending':
            st.session_state['snapshot_id'] = raw_data['id']
            st.warning("Scrape is taking some time, job ID has been saved you can check if it's done")
            st.rerun()

if check_btn:
    with st.spinner(f"Polling job {st.session_state['snapshot_id']}..."):
        result = poll_snapshot(st.session_state['snapshot_id'])
        
        if result['status'] == 'complete':
            # Job finished!
            df = clean_linkedin_data(result['data'])
            st.session_state['df'] = df
            st.session_state['snapshot_id'] = None # Clear the pending ID
            st.success("Data retrieved successfully!")
            st.rerun()
        elif result['status'] == 'pending':
            st.toast("⚠️ Still processing... try again in 10 seconds.")
        else:
            st.error("Job failed or ID invalid.")
            
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
    llm = ChatGoogleGenerativeAI(google_api_key=gemini_key, model="gemini-2.5-flash")
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