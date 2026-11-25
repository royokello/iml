import unittest

from utils.quant.targets import build_mixed_target_config


class MixedRanksTest(unittest.TestCase):
    def test_three_ranks_step_down_the_precision_ladder(self):
        config = build_mixed_target_config(
            "sym-med-nano",
            {
                "rank_1": ["h"],
                "rank_2": ["m"],
                "rank_3": ["l"],
                "bf16": ["b"],
                "sym-high": ["s"],
            },
        )
        self.assertEqual(
            config,
            {
                "sym-med": ["h"],
                "aff-high": ["m"],
                "aff-med": ["l"],
                "bf16": ["b"],
                "sym-high": ["s"],
            },
        )

    def test_two_ranks_are_unchanged(self):
        config = build_mixed_target_config(
            "sym-med-nano",
            {"rank_1": ["h"], "rank_2": ["l"]},
        )
        self.assertEqual(config, {"sym-med": ["h"], "aff-med": ["l"]})

    def test_no_suffix_merges_every_rank(self):
        config = build_mixed_target_config(
            "sym-high",
            {"rank_1": ["h"], "rank_2": ["m"], "rank_3": ["l"]},
        )
        self.assertEqual(config, {"sym-high": ["h", "m", "l"]})

    def test_mini_keeps_middle_ranks_with_the_bottom(self):
        config = build_mixed_target_config(
            "sym-high-mini",
            {"rank_1": ["h"], "rank_2": ["m"], "rank_3": ["l"]},
        )
        self.assertEqual(config, {"sym-high": ["h"], "sym-med": ["m", "l"]})

    def test_fp16_merges_every_rank(self):
        config = build_mixed_target_config(
            "fp16",
            {"rank_1": ["h"], "rank_2": ["m"], "rank_3": ["l"], "bf16": ["b"]},
        )
        self.assertEqual(config, {"fp16": ["h", "m", "l"], "bf16": ["b"]})

    def test_legacy_group_name_raises(self):
        with self.assertRaises(ValueError):
            build_mixed_target_config("sym-med-nano", {"high": ["h"], "low": ["l"]})

    def test_rank_gap_raises(self):
        with self.assertRaises(ValueError):
            build_mixed_target_config(
                "sym-med-nano",
                {"rank_1": ["h"], "rank_3": ["l"]},
            )

    def test_empty_set_raises(self):
        with self.assertRaises(ValueError):
            build_mixed_target_config("sym-med-nano", {})
