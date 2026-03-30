from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_multimodel_sweep import (
    SweepTarget,
    build_lambda_values,
    default_base_targets,
    infer_best_layer,
    parse_target_spec,
    resolve_candidate_vector_path,
)


class SweepHelpersTest(unittest.TestCase):
    def test_build_lambda_values_includes_endpoints(self) -> None:
        values = build_lambda_values(-150, 150, 5)

        self.assertEqual(values[0], -150)
        self.assertEqual(values[-1], 150)
        self.assertEqual(len(values), 61)

    def test_build_lambda_values_rejects_invalid_ranges(self) -> None:
        with self.assertRaises(ValueError):
            build_lambda_values(10, 0, 5)

        with self.assertRaises(ValueError):
            build_lambda_values(-10, 10, 0)

        with self.assertRaises(ValueError):
            build_lambda_values(-10, 7, 5)

    def test_parse_target_spec(self) -> None:
        target = parse_target_spec("qwen18b_base::Qwen/Qwen-1_8B::/tmp/qwen18b_base")

        self.assertEqual(
            target,
            SweepTarget(
                slug="qwen18b_base",
                model_name="Qwen/Qwen-1_8B",
                artifact_dir=Path("/tmp/qwen18b_base"),
                layer=None,
            ),
        )

    def test_parse_target_spec_with_layer_override(self) -> None:
        target = parse_target_spec("qwen25_3b_base::Qwen/Qwen2.5-3B::/tmp/qwen25_3b_base::19")

        self.assertEqual(target.layer, 19)
        self.assertEqual(target.slug, "qwen25_3b_base")

    def test_default_base_targets(self) -> None:
        root = Path("/tmp/rivanna-results")
        targets = default_base_targets(root)

        self.assertEqual(
            [target.slug for target in targets],
            ["qwen18b_base", "qwen25_3b_base", "qwen25_7b_base"],
        )
        self.assertEqual(targets[1].artifact_dir, root / "qwen25_3b_base")

    def test_resolve_candidate_vector_path_for_legacy_and_rivanna_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy_dir = root / "legacy"
            rivanna_dir = root / "rivanna"

            (legacy_dir / "activations").mkdir(parents=True)
            (rivanna_dir / "artifacts").mkdir(parents=True)

            legacy_path = legacy_dir / "activations" / "candidate_vectors.pt"
            rivanna_path = rivanna_dir / "artifacts" / "candidate_vectors.pt"
            legacy_path.write_bytes(b"legacy")
            rivanna_path.write_bytes(b"rivanna")

            self.assertEqual(resolve_candidate_vector_path(legacy_dir), legacy_path)
            self.assertEqual(resolve_candidate_vector_path(rivanna_dir), rivanna_path)

    def test_infer_best_layer_from_results_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "results.json").write_text(
                json.dumps({"best_layer": 17}),
                encoding="utf-8",
            )

            self.assertEqual(infer_best_layer(artifact_dir), 17)

    def test_infer_best_layer_from_top_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "validation").mkdir()
            (artifact_dir / "validation" / "top_layers.json").write_text(
                json.dumps([{"layer": 11}, {"layer": 9}]),
                encoding="utf-8",
            )

            self.assertEqual(infer_best_layer(artifact_dir), 11)

    def test_infer_best_layer_prefers_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "results.json").write_text(
                json.dumps({"best_layer": 17}),
                encoding="utf-8",
            )

            self.assertEqual(infer_best_layer(artifact_dir, explicit_layer=5), 5)


if __name__ == "__main__":
    unittest.main()
