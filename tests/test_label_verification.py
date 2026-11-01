"""Portable label verification must not weaken release-integrity checks."""

import unittest

import numpy as np
import pandas as pd

from kairos.label import OFFICIAL_LABELS_SHA256, verify_labels


def exact_frames():
    released = pd.DataFrame({"author": ["A", "A", "A"], "post": ["P0", "P1", "P2"],
                             "label": [0, 1, 2]})
    recomputed = released.assign(log_reactions=[1.0, 2.0, 4.0], level=2.0,
                                 resid=[-1.0, 0.0, 2.0])
    recomputed.attrs["cutoffs"] = (-0.5, 1.25)
    return released, recomputed


def boundary_frames():
    released = pd.DataFrame({"author": ["A", "B"], "post": ["P0", "P1"], "label": [1, 1]})
    recomputed = released.assign(label=[0, 2], log_reactions=[1.0, 2.25], level=[1.5, 1.0],
                                 resid=[np.nextafter(-0.5, -np.inf), np.nextafter(1.25, np.inf)])
    recomputed.attrs["cutoffs"] = (-0.5, 1.25)
    return released, recomputed


class LabelVerificationTest(unittest.TestCase):
    def test_exact_agreement_does_not_require_the_official_release_hash(self):
        released, recomputed = exact_frames()
        checked = verify_labels(released, recomputed, labels_sha256="synthetic")
        self.assertEqual(checked["exact_matches"], 3)
        self.assertEqual(checked["boundary_roundoff_count"], 0)
        self.assertEqual(checked["max_error"], 0.0)

    def test_pinned_release_allows_only_tiny_matching_boundary_differences(self):
        released, recomputed = boundary_frames()
        original = released.copy(deep=True)
        checked = verify_labels(released, recomputed, labels_sha256=OFFICIAL_LABELS_SHA256)
        self.assertEqual(checked["exact_matches"], 0)
        self.assertEqual(checked["boundary_roundoff_count"], 2)
        self.assertGreater(checked["max_error"], 0)
        self.assertLess(checked["max_error"], 1e-14)
        self.assertIn("8*spacing", checked["tolerance_policy"])
        pd.testing.assert_frame_equal(released, original)

    def test_tiny_mismatch_in_modified_or_unknown_labels_is_rejected(self):
        released, recomputed = boundary_frames()
        for checksum in ("", "0" * 64, OFFICIAL_LABELS_SHA256[:-1] + "2"):
            with self.subTest(checksum=checksum):
                with self.assertRaisesRegex(ValueError, "unchanged official labels.csv SHA256"):
                    verify_labels(released, recomputed, labels_sha256=checksum)

    def test_real_difference_much_smaller_than_one_micro_is_still_rejected(self):
        released, recomputed = boundary_frames()
        recomputed.loc[0, "resid"] = -0.5 - 1e-12
        with self.assertRaisesRegex(ValueError, "outside numerical cutoff tolerance"):
            verify_labels(released, recomputed, labels_sha256=OFFICIAL_LABELS_SHA256)

    def test_nonadjacent_classes_and_wrong_boundary_are_rejected(self):
        for case in ("nonadjacent", "wrong_boundary"):
            with self.subTest(case=case):
                released, recomputed = boundary_frames()
                if case == "nonadjacent":
                    released.loc[0, "label"] = 2
                else:
                    recomputed.loc[0, "resid"] = 1.25
                with self.assertRaisesRegex(ValueError, "outside numerical cutoff tolerance"):
                    verify_labels(released, recomputed, labels_sha256=OFFICIAL_LABELS_SHA256)

    def test_equal_row_counts_do_not_hide_different_post_keys(self):
        released, recomputed = exact_frames()
        recomputed.loc[0, "post"] = "OTHER"
        with self.assertRaisesRegex(ValueError, "key coverage differs"):
            verify_labels(released, recomputed, labels_sha256=OFFICIAL_LABELS_SHA256)

    def test_missing_and_duplicate_rows_are_rejected_before_roundoff(self):
        released, recomputed = boundary_frames()
        for malformed, message in ((recomputed.iloc[:1].copy(), "key coverage differs"),
                                    (pd.concat([recomputed, recomputed.iloc[:1]]), "duplicate post keys")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    verify_labels(released, malformed, labels_sha256=OFFICIAL_LABELS_SHA256)

    def test_nonfinite_residual_cannot_be_accepted(self):
        released, recomputed = boundary_frames()
        recomputed.loc[0, "resid"] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            verify_labels(released, recomputed, labels_sha256=OFFICIAL_LABELS_SHA256)


if __name__ == "__main__":
    unittest.main()
