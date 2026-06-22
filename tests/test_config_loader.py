from types import SimpleNamespace
import logging
import unittest

import yaml

from connectomix.__main__ import _configure_condition_masking
from connectomix.config.defaults import (
    ConditionMaskingConfig,
    ParticipantConfig,
    TemporalCensoringConfig,
)
from connectomix.config.loader import config_from_dict


class ConfigLoaderTests(unittest.TestCase):
    def test_nested_dataclass_sections_are_converted(self) -> None:
        config_data = yaml.safe_load(
            """
            method: roiToRoi
            atlas: aal
            condition_masking:
              enabled: false
            temporal_censoring:
              enabled: false
            """
        )

        config = config_from_dict(config_data, ParticipantConfig)

        self.assertIsInstance(config.condition_masking, ConditionMaskingConfig)
        self.assertIsInstance(config.temporal_censoring, TemporalCensoringConfig)
        self.assertFalse(config.condition_masking.enabled)
        self.assertFalse(config.temporal_censoring.enabled)

    def test_condition_masking_cli_path_no_longer_crashes(self) -> None:
        config = config_from_dict(
            {
                "method": "roiToRoi",
                "atlas": "aal",
                "condition_masking": {"enabled": False},
            },
            ParticipantConfig,
        )

        args = SimpleNamespace(conditions=["face"], events_file=None, transition_buffer=0.0)

        _configure_condition_masking(args, config, logging.getLogger("connectomix.tests"), condition="face")

        self.assertTrue(config.condition_masking.enabled)
        self.assertEqual(config.condition_masking.conditions, ["face"])
        self.assertTrue(config.temporal_censoring.enabled)
        self.assertEqual(config.temporal_censoring.condition_selection["conditions"], ["face"])


if __name__ == "__main__":
    unittest.main()