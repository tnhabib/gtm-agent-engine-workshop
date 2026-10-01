import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile


class UpdateProspectInfoTests(unittest.TestCase):
    prospect_id = "LEAD-39002"

    def setUp(self):
        self.original_tech_stack = list(
            data_service.PROSPECTS[self.prospect_id]["tech_stack"]
        )
        data_service._PROFILES.pop(self.prospect_id, None)

    def tearDown(self):
        data_service.PROSPECTS[self.prospect_id]["tech_stack"] = self.original_tech_stack
        data_service._PROFILES.pop(self.prospect_id, None)

    def test_update_persists_to_source_record(self):
        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertTrue(result["updated"])
        self.assertIn("Terraform", data_service.fetch_tech_stack(self.prospect_id))

    def test_update_invalidates_cached_profile(self):
        before = build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        data_service.update_prospect_info(self.prospect_id, "Terraform")

        after = build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertNotIn("Terraform", before["prospect_profile"]["tech_stack"])
        self.assertIn("Terraform", after["prospect_profile"]["tech_stack"])

    def test_update_does_not_duplicate_existing_technology(self):
        technology = self.original_tech_stack[0]

        result = data_service.update_prospect_info(self.prospect_id, technology)

        self.assertFalse(result["updated"])
        self.assertTrue(result["already_present"])
        self.assertEqual(
            data_service.fetch_tech_stack(self.prospect_id).count(technology), 1
        )


if __name__ == "__main__":
    unittest.main()
