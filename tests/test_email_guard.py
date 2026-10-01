import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def test_disqualified_prospect_is_blocked(self):
        result = send_prospect_email.func(
            {"prospect_id": "LEAD-50001", "name": "Priya Nair", "email": "priya@example.com"},
            "Demo",
            "Let us connect.",
            runtime=None,
            from_rep={"email": "rep@example.com", "name": "Rep"},
        )

        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect LEAD-50001 is disqualified; email not sent.",
        })

    def test_eligible_prospect_is_sent(self):
        result = send_prospect_email.func(
            {"prospect_id": "LEAD-12853", "name": "Omar Okafor", "email": "omar@example.com"},
            "Demo",
            "Let us connect.",
            runtime=None,
            from_rep={"email": "rep@example.com", "name": "Rep"},
        )

        self.assertEqual(result["status"], "sent")
        self.assertIn("message_id", result)

    def test_prompt_explains_disqualified_send_behavior(self):
        self.assertIn(
            "Never email a prospect whose source record is marked disqualified; if a send is blocked, tell the rep it was not sent and why.",
            SYSTEM_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
