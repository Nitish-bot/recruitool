import sys
from unittest.mock import MagicMock, patch

# Mock streamlit before importing app so the UI code no-ops and
# st.secrets.get() returns controlled values.
mock_st = MagicMock()


def _secrets_get(key, default=None):
    if key == "APIFY_KEY":
        return "test-apify-key"
    return default


mock_st.secrets.get.side_effect = _secrets_get
sys.modules["streamlit"] = mock_st

import app  # noqa: E402
from tests.conftest import MOCK_APIFY_ITEMS  # noqa: E402


@patch("requests.post")
def test_apify_api_call_structure(mock_post):
    """Verify scrape_with_apify builds the correct endpoint, headers, and body."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_APIFY_ITEMS
    mock_post.return_value = mock_response

    result = app.scrape_with_apify("https://www.linkedin.com/in/testuser")

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args

    endpoint = args[0] if args else ""
    assert "harvestapi~linkedin-profile-scraper" in endpoint, (
        f"Unexpected endpoint: {endpoint}"
    )
    assert "Authorization" in kwargs.get("headers", {}), "Missing Authorization header"
    assert kwargs["headers"]["Authorization"].startswith("Bearer"), (
        "Authorization should use Bearer scheme"
    )

    body = kwargs.get("json", {})
    assert body.get("profileScraperMode") == "Profile details no email ($4 per 1k)", (
        "Unexpected profileScraperMode"
    )
    assert "https://www.linkedin.com/in/testuser" in body.get("queries", []), (
        "URL not found in queries"
    )
    assert result["status"] == "complete"
    assert result["data"] == MOCK_APIFY_ITEMS


def test_parse_apify_profile_produces_structured_dict():
    """Verify parse_apify_profile returns a dict with the expected schema."""
    result = app.parse_apify_profile(MOCK_APIFY_ITEMS)

    assert isinstance(result, dict), "Result should be a dict"

    required_keys = [
        "first_name",
        "last_name",
        "headline",
        "experience",
        "education",
        "skills",
        "certifications",
        "location_text",
        "linkedin_url",
    ]
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"

    assert isinstance(result["experience"], list), "experience should be a list"
    assert len(result["experience"]) > 0, "experience should not be empty"
    for exp_key in {"position", "company_name", "duration"}:
        assert exp_key in result["experience"][0], (
            f"Missing key in experience[0]: {exp_key}"
        )

    assert isinstance(result["skills"], list), "skills should be a list"
    assert len(result["skills"]) > 0, "skills should not be empty"
    assert all(isinstance(s, str) for s in result["skills"]), (
        "All skills should be strings"
    )

    assert isinstance(result["education"], list), "education should be a list"
    assert len(result["education"]) > 0, "education should not be empty"
    assert result["first_name"] == "Sarah"
    assert result["full_name"] == "Sarah Chen"


def test_missing_fields_handled_gracefully():
    """Verify parse_apify_profile handles minimal data without raising."""
    minimal_items = [{"element": {"firstName": "Test", "lastName": "User"}}]

    result = app.parse_apify_profile(minimal_items)

    assert result is not None, "Should return a dict, not None"
    assert result["experience"] == [], "Missing experience should default to []"
    assert result["skills"] == [], "Missing skills should default to []"
    assert result["about"] is None, "Missing about should be None"
    assert result["headline"] is None, "Missing headline should be None"
    assert result["photo"] is None, "Missing photo should be None"
    assert result["full_name"] == "Test User", (
        f"Expected 'Test User', got '{result['full_name']}'"
    )
