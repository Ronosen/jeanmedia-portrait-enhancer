import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from background import automatic_edge_region, clean_backdrop, edge_repair, person_probability, repair_mask
from equipment import apply_equipment_plan, detect_equipment, polygon_mask, validate_plan

WORKER = Path(__file__).with_name("photo.py")


class EquipmentTests(unittest.TestCase):
    def scene(self, side):
        image = Image.new("RGB", (600, 800), (210, 35, 25))
        draw = ImageDraw.Draw(image)
        draw.polygon([(0, 100), (65, 120), (120, 250), (90, 390), (0, 450)], fill=(25, 25, 25))
        draw.polygon([(0, 120), (50, 140), (95, 250), (70, 370), (0, 425)], fill=(250, 250, 250))
        draw.rectangle((260, 170, 400, 700), fill=(40, 80, 180))
        probability = np.zeros((800, 600), np.float32)
        probability[170:701, 260:401] = .99
        rgb = np.asarray(image)
        if side == "right":
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            probability = np.ascontiguousarray(probability[:, ::-1])
        return rgb, probability

    def test_detects_both_edges_on_colored_backdrop_without_marking(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                rgb, probability = self.scene(side)
                plan = detect_equipment(rgb, probability)
                self.assertEqual(plan["repairs"][0]["side"], side)
                rendered = np.asarray(apply_equipment_plan(Image.fromarray(rgb), rgb, probability, plan))
                mask = polygon_mask(probability.shape, plan["repairs"][0]["polygon"])
                np.testing.assert_array_equal(rendered[mask == 0], rgb[mask == 0])
                np.testing.assert_array_equal(rendered[probability > .9], rgb[probability > .9])
                core = (rgb.min(axis=2) > 240) & (mask > 0)
                self.assertGreater(np.count_nonzero(core), 1000)
                self.assertLess(rendered[core].mean(axis=0)[1], 70)

    def test_replayed_plan_rejects_subject_overlap(self):
        rgb, probability = self.scene("left")
        plan = detect_equipment(rgb, probability)
        probability[:, :160] = .99
        with self.assertRaisesRegex(ValueError, "protected"):
            validate_plan(rgb, probability, plan)


class BackgroundTests(unittest.TestCase):
    def test_automatic_removal_skips_a_plain_backdrop(self):
        pixels = np.full((400, 300, 3), 25, dtype=np.uint8)
        probability = np.zeros((400, 300), dtype=np.float32)
        probability[100:380, 170:260] = .99
        with self.assertRaisesRegex(ValueError, "No clear edge equipment"):
            automatic_edge_region(pixels, probability)

    def test_automatic_removal_requires_a_confident_subject(self):
        pixels = np.full((400, 300, 3), 25, dtype=np.uint8)
        probability = np.zeros((400, 300), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "subject"):
            automatic_edge_region(pixels, probability)

    def test_edge_repair_leaves_pixels_outside_selection_untouched(self):
        pixels = np.full((300, 400, 3), 25, dtype=np.uint8)
        pixels[20:150, :100] = 190
        pixels[90:280, 190:320] = (180, 40, 30)
        result = np.asarray(edge_repair(Image.fromarray(pixels), (.30, 0, .56)))
        np.testing.assert_array_equal(result[25:145, :100], 25)
        np.testing.assert_array_equal(result[:, 120:], pixels[:, 120:])
        np.testing.assert_array_equal(result[168:], pixels[168:])

    def test_repair_changes_only_an_isolated_mark_clear_of_subject(self):
        pixels = np.full((240, 320, 3), 155, dtype=np.uint8)
        probability = np.zeros((240, 320), dtype=np.float32)
        probability[30:180, 145:230] = 0.99
        pixels[50:54, 55:59] = 105
        pixels[50:54, 125:129] = 105
        pixels[140:144, 55:59] = 105
        mask, status = repair_mask(pixels, probability)
        self.assertEqual(status, "")
        self.assertEqual(mask[51, 56], 255)
        result = np.asarray(clean_backdrop(Image.fromarray(pixels), mask))
        np.testing.assert_array_equal(result[mask == 0], pixels[mask == 0])
        self.assertGreater(result[51, 56, 0], 145)

    def test_cpu_model_refuses_to_repair_without_a_subject(self):
        pixels = np.full((100, 150, 3), 127, dtype=np.uint8)
        probability = person_probability(pixels)
        self.assertEqual(probability.shape, (100, 150))
        mask, status = repair_mask(pixels, probability)
        self.assertIsNone(mask)
        self.assertIn("Needs Review", status)


class PhotoWorkerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "original portrait.png"
        image = Image.new("RGBA", (1400, 900), (56, 60, 63, 128))
        ImageDraw.Draw(image).rectangle((120, 130, 570, 650), fill=(96, 91, 90, 255))
        image.save(self.source)
        self.digest = hashlib.sha256(self.source.read_bytes()).digest()

    def run_worker(self, *args):
        return subprocess.run([sys.executable, str(WORKER), str(self.source), *args],
                              capture_output=True, text=True)

    def test_preview_is_bounded_and_enhancement_changes_tone(self):
        import base64
        import io

        result = self.run_worker("--enhance")
        self.assertEqual(result.returncode, 0, result.stderr)
        preview = json.loads(result.stdout)
        self.assertEqual((preview["width"], preview["height"]), (1400, 900))
        self.assertNotEqual(preview["before"], preview["after"])
        with Image.open(io.BytesIO(base64.b64decode(preview["after"].split(",")[1]))) as image:
            self.assertEqual(max(image.size), 1280)

    def test_import_and_export_use_first_frame_of_multipage_tiff(self):
        self.source = self.root / "multipage.tif"
        first = Image.new("RGB", (50, 30), "red")
        second = Image.new("RGB", (80, 60), "blue")
        first.save(self.source, save_all=True, append_images=[second])
        result = self.run_worker()
        self.assertEqual(result.returncode, 0, result.stderr)
        preview = json.loads(result.stdout)
        self.assertEqual((preview["width"], preview["height"], preview["frames"]), (50, 30, 2))
        output = self.root / "first-page.png"
        result = self.run_worker("--output", str(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        with Image.open(output) as image:
            self.assertEqual(image.tobytes(), first.tobytes())

    def test_flat_photo_gains_contrast_without_lifting_its_dark_patch(self):
        self.source = self.root / "flat.png"
        image = Image.new("RGB", (160, 100), (145, 145, 145))
        ImageDraw.Draw(image).rectangle((0, 0, 79, 99), fill=(85, 85, 85))
        image.save(self.source)
        output = self.root / "enhanced.png"
        result = self.run_worker("--enhance", "--output", str(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        with Image.open(output) as enhanced:
            dark = enhanced.getpixel((30, 50))[0]
            light = enhanced.getpixel((130, 50))[0]
            self.assertLessEqual(dark, 85)
            self.assertGreater(light - dark, 65)

    def test_batch_compact_export_returns_review_summary_without_large_plan(self):
        output = self.root / 'batch.jpg'
        result = self.run_worker('--output', str(output), '--compact-output')
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertTrue(data['saved']);self.assertIsInstance(data['review'], list)
        self.assertNotIn('studioPlan', data)
        with Image.open(output) as image:self.assertEqual(image.size, (1400, 900))
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).digest(), self.digest)

    def test_exports_preserve_dimensions_dpi_and_original(self):
        for suffix, expected in [("jpg", "JPEG"), ("png", "PNG"), ("tiff", "TIFF")]:
            with self.subTest(format=expected):
                output = self.root / f"finished.{suffix}"
                result = self.run_worker("--enhance", "--output", str(output), "--dpi", "240")
                self.assertEqual(result.returncode, 0, result.stderr)
                with Image.open(output) as photo:
                    self.assertEqual(photo.format, expected)
                    self.assertEqual(photo.size, (1400, 900))
                    self.assertAlmostEqual(photo.info["dpi"][0], 240, delta=0.1)
                    if expected != "JPEG":
                        with Image.open(self.source) as original:
                            self.assertEqual(photo.getchannel("A").tobytes(),
                                             original.getchannel("A").tobytes())
                saved = output.read_bytes()
                self.assertNotEqual(self.run_worker("--output", str(output)).returncode, 0)
                self.assertEqual(output.read_bytes(), saved)
        self.assertNotEqual(self.run_worker("--output", str(self.source)).returncode, 0)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).digest(), self.digest)

