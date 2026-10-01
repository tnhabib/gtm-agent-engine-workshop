import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service
from gtm_agent.gtm_agent import (
    build_prospect_profile,
    get_prospect,
    score_prospect,
    send_prospect_email,
)


SENSITIVE = {
    "billing_qualification": {
        "tax_id": "123",
        "date_of_birth": "2000-01-01",
        "card_on_file": "4111",
        "credit_check_ref": "ref-1",
    },
    "nested": [{"tax_id": "456"}],
}


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        data_service._PROFILES.clear()

    def test_get_prospect_returns_contact_allowlist(self):
        record = {
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "disqualified": False,
            **SENSITIVE,
            "future_field": "do not expose",
        }
        with patch.object(data_service, "get_prospect_record", return_value=record):
            result = get_prospect.invoke({"prospect_id": "LEAD-1"})

        self.assertEqual(
            result["prospect"],
            {
                "prospect_id": "LEAD-1",
                "name": "Ada Lovelace",
                "email": "ada@example.com",
                "disqualified": False,
            },
        )

    def test_email_flow_uses_allowlisted_contact_fields(self):
        with patch.object(
            data_service,
            "get_prospect_record",
            return_value={"name": "Ada", "email": "ada@example.com", **SENSITIVE},
        ):
            prospect = get_prospect.invoke({"prospect_id": "LEAD-1"})["prospect"]
        result = send_prospect_email.func(
            prospect,
            "Demo",
            "Let’s connect.",
            runtime=None,
            from_rep={"name": "Grace", "email": "grace@example.com"},
        )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "ada@example.com")

    def test_build_profile_sanitizes_new_profile(self):
        record = {"name": "Ada", "email": "ada@example.com", **SENSITIVE}
        with patch.object(data_service, "get_prospect_record", return_value=record), \
                patch.object(data_service, "fetch_engagement_history", return_value=[]), \
                patch.object(data_service, "fetch_account_details", return_value=[]), \
                patch.object(data_service, "fetch_tech_stack", return_value=["AWS"]):
            result = build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

        profile = result["prospect_profile"]
        self.assertNotIn("billing_qualification", profile)
        self.assertNotIn("tax_id", profile)
        self.assertNotIn("date_of_birth", profile)
        self.assertNotIn("card_on_file", profile)
        self.assertNotIn("credit_check_ref", profile)
        self.assertNotIn("tax_id", profile["nested"][0])
        self.assertEqual(data_service._PROFILES["LEAD-1"], profile)

    def test_build_profile_sanitizes_and_replaces_cached_profile(self):
        data_service._PROFILES["LEAD-1"] = {
            "prospect_id": "LEAD-1",
            "name": "Ada",
            **SENSITIVE,
        }

        result = build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

        self.assertNotIn("billing_qualification", result["prospect_profile"])
        self.assertNotIn("tax_id", result["prospect_profile"])
        self.assertEqual(data_service._PROFILES["LEAD-1"], result["prospect_profile"])

    @patch("gtm_agent.gtm_agent._scoring_llm")
    def test_score_prospect_sanitizes_input_before_scoring(self, scoring_llm):
        scoring_llm.invoke.return_value.model_dump.return_value = {"score": 80}
        profile = {
            "prospect_id": "LEAD-1",
            "name": "Ada",
            **SENSITIVE,
            "tech_stack": ["stale"],
        }
        offering = {
            "required_tech_stack": ["AWS"],
            "min_annual_revenue": 1,
            "description": "Cloud offering",
        }

        with patch.object(data_service, "fetch_tech_stack", return_value=["AWS"]):
            result = score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        self.assertEqual(result, {"score": 80})
        user_prompt = scoring_llm.invoke.call_args.args[0][1]["content"]
        self.assertNotIn("billing_qualification", user_prompt)
        self.assertNotIn("tax_id", user_prompt)
        self.assertNotIn("date_of_birth", user_prompt)
        self.assertNotIn("card_on_file", user_prompt)
        self.assertNotIn("credit_check_ref", user_prompt)
        self.assertIn('"tech_stack": [\n    "AWS"\n  ]', user_prompt)


if __name__ == "__main__":
    unittest.main()
