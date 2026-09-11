import tempfile
import unittest
import warnings
from pathlib import Path

from run_behavioral_pipeline import CONFIG
from run_behavioral_pipeline import _make_options
from run_behavioral_pipeline import load_rows
from run_behavioral_pipeline import validate_csv
from run_behavioral_pipeline import write_legacy_dataset


class ConditionLabelTests(unittest.TestCase):
    def write_csv(self, directory, name, rows):
        path = Path(directory) / name
        path.write_text("condition,stimulus,response\n" + "\n".join(rows) + "\n")
        return path

    def test_validation_accepts_arbitrary_integer_condition_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            circular_csv = self.write_csv(
                temp_dir,
                "circular.csv",
                ["-3,0,4", "12,359,355"],
            )
            interval_csv = self.write_csv(
                temp_dir,
                "interval.csv",
                ["-20,0.1,0.2", "42,2.9,2.8"],
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                circular_summary = validate_csv(circular_csv, "circular")
                interval_summary = validate_csv(interval_csv, "interval")

            self.assertEqual(circular_summary.conditions, (-3, 12))
            self.assertEqual(interval_summary.conditions, (-20, 42))

    def test_legacy_filename_separates_multi_digit_condition_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_csv = self.write_csv(
                temp_dir,
                "circular.csv",
                ["-3,0,4", "12,359,355"],
            )
            args = _make_options(input_csv, "circular", dataset_name="arbitrary-labels")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rows = load_rows(args, CONFIG["circular"])
            config = dict(CONFIG["circular"])
            config["basis_dir"] = Path(temp_dir) / "output"

            filename, output_path = write_legacy_dataset(args, config, rows)

            self.assertEqual(
                filename,
                "UserBehavior_arbitrary-labels_180_2_-3-12_N2.txt",
            )
            self.assertTrue(output_path.is_file())


class RunnerNameTests(unittest.TestCase):
    def test_dispatch_uses_short_runner_names(self):
        self.assertEqual(
            CONFIG["circular"]["scripts"],
            {
                "map": "RunCircular_Free_L0.py",
                "l1": "RunCircular_Free_L1Loss.py",
                "lp": "RunCircular_Free_CosineLoss.py",
            },
        )
        self.assertEqual(
            CONFIG["interval"]["scripts"],
            {
                "map": "RunInterval_Free_L0_Round2.py",
                "l1": "RunInterval_Free_L1_Round2.py",
                "lp": "RunInterval_Free_Lp_Round2.py",
            },
        )

    def test_renamed_runners_start_with_upstream_filename(self):
        demo_dir = Path(__file__).resolve().parents[1]
        expected_sources = {
            "circular/RunCircular_Free_CosineLoss.py": "RunSynthetic_FreePrior_CosineLoss_OnSim.py",
            "circular/RunCircular_Free_L1Loss.py": "RunSynthetic_FreePrior_L1Loss_OnSim.py",
            "circular/RunCircular_Free_L0.py": "RunSynthetic_FreePrior_ZeroTrig_OnSim.py",
            "interval/RunInterval_Free_Lp_Round2.py": "RunSynthetic_DenseRemington_FreeEncoding_OnSim_OtherNoiseLevels_VarySize_Round2.py",
            "interval/RunInterval_Free_L1_Round2.py": "RunSynthetic_DenseRemington_FreeEncoding_L1_OnSim_OtherNoiseLevels_VarySize_Round2.py",
            "interval/RunInterval_Free_L0_Round2.py": "RunSynthetic_DenseRemington_FreeEncoding_Zero_OnSim_OtherNoiseLevels_VarySize_Round2.py",
            "circular/RunGardelle_FreePrior_L1Loss.py": "RunGardelle_FreePrior_L1Loss_Downsampled_TargetSize.py",
            "circular/RunGardelle_FreePrior_ZeroTrig.py": "RunGardelle_FreePrior_ZeroTrig_Downsampled_TargetSize.py",
        }
        for relative_path, source_name in expected_sources.items():
            first_line = (demo_dir / relative_path).read_text().splitlines()[0]
            self.assertEqual(
                first_line,
                f"# Created from original upstream file: {source_name}",
            )

    def test_gardelle_l1_and_l0_runners_use_the_full_dataset(self):
        demo_dir = Path(__file__).resolve().parents[1]
        for filename in (
            "RunGardelle_FreePrior_L1Loss.py",
            "RunGardelle_FreePrior_ZeroTrig.py",
        ):
            source = (demo_dir / "circular" / filename).read_text()
            for downsampling_name in (
                "levelsToDownSampleTo",
                "targetSize",
                "withSEED",
                "torch.randperm",
            ):
                self.assertNotIn(downsampling_name, "\n".join(source.splitlines()[1:]))
            self.assertEqual(source.count("for DURATION in range(1,6):"), 2)


if __name__ == "__main__":
    unittest.main()
