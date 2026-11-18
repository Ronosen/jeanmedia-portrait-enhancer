import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from preflight import inspect
from props import plan_props, heal_props, validate_props
from service import project, change_mask
from studio import prepare_backdrop, render_backdrop, floor_boundary


class WorkflowTests(unittest.TestCase):
    def test_cropped_portrait_does_not_get_an_invented_floor_horizon(self):
        rgb = np.full((400, 300, 3), 150, np.uint8)
        protected = np.zeros((400, 300), bool)
        protected[80:, 100:220] = True
        self.assertEqual(floor_boundary(rgb, protected), 1.0)

    def test_prop_healing_retains_person_edges_and_all_unmasked_pixels(self):
        rgb = np.full((300, 240, 3), 220, np.uint8)
        rgb[190:193, 100:103] = 140
        rgb[100:160, 100:150] = (80, 30, 10)
        person = np.zeros((300, 240), bool)
        person[100:160, 100:150] = True
        objects = [{"class": 56, "confidence": .9, "box": [40, 90, 220, 280]}]
        plan = plan_props(rgb, person, objects)
        self.assertGreater(plan["count"], 0)
        validate_props(plan, rgb, person, objects)
        result = np.asarray(heal_props(Image.fromarray(rgb), plan))
        self.assertGreater(result[191, 101, 0], 200)
        np.testing.assert_array_equal(result[person], rgb[person])
        np.testing.assert_array_equal(result[:90], rgb[:90])
        tampered = rgb.copy()
        tampered[190:193, 100:103] = 220
        with self.assertRaises(ValueError):
            validate_props(plan, tampered, person, objects)

    def test_project_reopens_repair_layers_but_rejects_changed_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"original.png"
            image = Image.new("RGB", (600, 800), (150, 155, 160))
            rgb = np.asarray(image).copy()
            rgb[170:701, 240:360] = (190, 40, 30)
            rgb[20:60, :40] = 50
            image = Image.fromarray(rgb)
            image.save(source)
            probability = np.zeros((800, 600), np.float32)
            probability[170:701, 240:360] = .99
            plan, alpha, _ = prepare_backdrop(rgb, probability, [], floor_mode="patch")
            repaired = render_backdrop(image, alpha, plan)
            path = root/"session.jeanmedia"
            data = {"path": str(path), "source": str(source), "studioPlan": plan, "enhance": True}
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            project("save-project", data)
            loaded = project("open-project", {"path": str(path)})
            self.assertEqual(loaded["studioPlan"], plan)
            self.assertEqual(loaded["sourceSha256"], original_hash)
            self.assertGreater(np.count_nonzero(np.asarray(change_mask(plan))), 0)
            self.assertNotEqual(repaired.tobytes(), image.tobytes())
            with self.assertRaises(FileExistsError):
                project("save-project", data)
            Image.new("RGB", image.size, "white").save(source)
            with self.assertRaisesRegex(ValueError, "changed"):
                project("open-project", {"path": str(path)})

    def test_healthy_backdrop_is_not_repainted(self):
        rgb = np.full((800, 600, 3), 150, np.uint8)
        rgb[170:701, 240:360] = (190, 40, 30)
        rgb[40:80, 10:40] = 30
        probability = np.zeros((800, 600), np.float32)
        probability[170:701, 240:360] = .99
        plan, alpha, _ = prepare_backdrop(rgb, probability, [], floor_mode="patch")
        output = np.asarray(render_backdrop(Image.fromarray(rgb), alpha, plan))
        np.testing.assert_array_equal(output[100:160, 100:200], rgb[100:160, 100:200])
        self.assertGreater(output[50, 20, 0], 140)

    def test_quality_checks_flag_unrecoverable_flat_clipped_input(self):
        result = inspect(Image.new("RGB", (100, 100), "white"))
        self.assertTrue(result["needsReview"])
        self.assertIn("Clipped highlights", result["warnings"])
        self.assertEqual(result["crop"], [0, 0, 1, 1])

    def test_inspection_matches_full_resolution_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root/"input.png", root/"finished.png"
            rng = np.random.default_rng(5)
            Image.fromarray(rng.integers(40, 160, (180, 240, 3), np.uint8)).save(source)
            worker = str(Path(__file__).with_name("photo.py"))
            common = [sys.executable, worker, str(source), "--enhance"]
            saved = subprocess.run(common+["--output", str(output)], capture_output=True, text=True)
            self.assertEqual(saved.returncode, 0, saved.stderr)
            checked = subprocess.run(common+["--inspect-region", "20", "30", "100", "80"], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            payload = json.loads(checked.stdout)
            crop = Image.open(io.BytesIO(base64.b64decode(payload["after"].split(",")[1])))
            with Image.open(output) as full:
                self.assertEqual(crop.tobytes(), full.crop((20,30,120,110)).tobytes())
