import unittest

from utils.quant.targets import build_mixed_target_config


class MixedRanksTest(unittest.TestCase):
    def test_base_merges_every_rank(self):
        config = build_mixed_target_config(
            "q6",
            {"rank_1": ["h"], "rank_2": ["m"], "rank_3": ["l"]},
        )
        self.assertEqual(config, {"q6": ["h", "m", "l"]})

    def test_max_three_ranks_spread_around_the_base(self):
        config = build_mixed_target_config(
            "q6-max",
            {
                "rank_1": ["h"],
                "rank_2": ["m"],
                "rank_3": ["l"],
                "bf16": ["b"],
                "q8": ["s"],
            },
        )
        self.assertEqual(
            config,
            {
                "q8": ["h", "s"],
                "q6": ["m"],
                "q5": ["l"],
                "bf16": ["b"],
            },
        )

    def test_max_two_ranks_raises_the_top_rank_only(self):
        config = build_mixed_target_config(
            "q6-max",
            {"rank_1": ["h"], "rank_2": ["l"]},
        )
        self.assertEqual(config, {"q8": ["h"], "q6": ["l"]})

    def test_mini_two_ranks_drops_the_bottom_rank(self):
        config = build_mixed_target_config(
            "q6-mini",
            {"rank_1": ["h"], "rank_2": ["l"]},
        )
        self.assertEqual(config, {"q6": ["h"], "q5": ["l"]})

    def test_mini_three_ranks_keeps_the_middle_ranks_with_the_bottom(self):
        config = build_mixed_target_config(
            "q8-mini",
            {"rank_1": ["h"], "rank_2": ["m"], "rank_3": ["l"]},
        )
        self.assertEqual(config, {"q8": ["h"], "q6": ["m", "l"]})

    def test_fp16_merges_every_rank(self):
        config = build_mixed_target_config(
            "fp16",
            {"rank_1": ["h"], "rank_2": ["m"], "rank_3": ["l"], "bf16": ["b"]},
        )
        self.assertEqual(config, {"fp16": ["h", "m", "l"], "bf16": ["b"]})

    def test_old_method_names_are_rejected(self):
        with self.assertRaises(ValueError):
            build_mixed_target_config("sym-high-mini", {"rank_1": ["h"]})

    def test_legacy_group_name_raises(self):
        with self.assertRaises(ValueError):
            build_mixed_target_config("q6-mini", {"high": ["h"], "low": ["l"]})

    def test_rank_gap_raises(self):
        with self.assertRaises(ValueError):
            build_mixed_target_config(
                "q6-mini",
                {"rank_1": ["h"], "rank_3": ["l"]},
            )

    def test_empty_set_raises(self):
        with self.assertRaises(ValueError):
            build_mixed_target_config("q6-mini", {})

    def test_max_above_the_top_method_raises(self):
        with self.assertRaises(ValueError):
            build_mixed_target_config("q8-max", {"rank_1": ["h"], "rank_2": ["l"]})

    def test_mini_below_the_bottom_method_raises(self):
        with self.assertRaises(ValueError):
            build_mixed_target_config("q2-mini", {"rank_1": ["h"], "rank_2": ["l"]})

    def test_max_with_one_rank_raises(self):
        with self.assertRaises(ValueError):
            build_mixed_target_config("q6-max", {"rank_1": ["h"]})
