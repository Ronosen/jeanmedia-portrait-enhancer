import unittest

import numpy as np
from PIL import Image

from floor import apply_floor_patches, decode_mask, plan_floor_patches, validate_floor_patches,front_edge_extension


class FloorPatchTests(unittest.TestCase):
    def test_diagonal_scuff_uses_oriented_width_without_removing_broad_reflection(self):
        import cv2
        rgb=np.full((400,300,3),55,np.uint8)
        cv2.line(rgb,(30,300),(110,330),(85,85,85),1)
        cv2.ellipse(rgb,(225,310),(25,15),0,0,360,(68,68,68),-1)
        plan=plan_floor_patches(rgb,np.zeros((400,300),bool),.6,context=True)
        mask=decode_mask(plan['defects'],(400,300))
        self.assertTrue(mask[315,70]);self.assertFalse(mask[310,225])

    def test_faint_scuff_is_repaired_but_gradient_and_subject_are_preserved(self):
        rgb=np.full((300,240,3),50,np.uint8)
        for y in range(180,300):rgb[y]+=np.uint8((y-180)//15)
        rgb[230:232,40:65]+=12
        protected=np.zeros((300,240),bool);protected[210:260,120:160]=True
        rgb[230:232,125:150]+=12
        plan=plan_floor_patches(rgb,protected,.55,context=True)
        mask=decode_mask(plan['defects'],protected.shape)
        self.assertTrue(mask[230,50]);self.assertFalse(mask[protected].any())
        self.assertFalse(mask[260,30]);self.assertLess((mask>0).mean(),.01)

    def test_measured_mat_extension_reaches_shoes_without_selecting_them(self):
        from floor import close_guard_gaps
        protected=np.zeros((400,300),bool);protected[340:376,130:170]=True
        extension=np.zeros_like(protected);extension[367:]=True;extension[350:390,120:180]=False
        repaired=close_guard_gaps(extension,protected)
        self.assertTrue(repaired[374,126]);self.assertFalse(repaired[protected].any())
        self.assertFalse(repaired[360,126]);self.assertTrue(repaired[390,50])

    def test_front_edge_prefers_material_boundary_over_faint_floor_reflection(self):
        rgb=np.full((600,400,3),50,np.uint8);rgb[559:568]=42;rgb[590:]=190
        mask=front_edge_extension(rgb,np.ones((600,400),bool),450)>0
        self.assertFalse(mask[565,30]);self.assertTrue(mask[598,30])
    def test_texture_extension_is_continuous_and_never_copies_subject_pixels(self):
        rgb=np.full((300,240,3),60,np.uint8)
        for y in range(240,280):rgb[y]+=np.uint8(y%7)
        rgb[280:]=210;rgb[265:278,100:120]=(240,20,20)
        protected=np.zeros((300,240),bool);protected[265:278,100:120]=True
        empty=np.zeros((300,240),np.uint8);extension=empty.copy();extension[280:]=255
        from floor import encode_mask
        plan={'version':1,'shape':[300,240],'defects':encode_mask(empty),'extension':encode_mask(extension),
              'color':[60]*3,'engine':'texture-v1'}
        out=np.asarray(apply_floor_patches(Image.fromarray(rgb),.7,plan,protected))
        self.assertLess(out[280:,110].max(),100)
        np.testing.assert_array_equal(out[285,10],out[285,50])
        np.testing.assert_array_equal(out[protected],rgb[protected])
    def test_texture_extension_does_not_spread_dark_mat_tape_into_new_floor(self):
        rgb=np.full((300,240,3),60,np.uint8);rgb[276:279,150:190]=5;rgb[280:]=210
        empty=np.zeros((300,240),np.uint8);extension=empty.copy();extension[280:]=255
        from floor import encode_mask
        plan={'version':1,'shape':[300,240],'defects':encode_mask(empty),'extension':encode_mask(extension),
              'color':[60]*3,'engine':'texture-v1'}
        out=np.asarray(apply_floor_patches(Image.fromarray(rgb),.7,plan,np.zeros_like(empty,bool)))
        self.assertGreater(out[280:,160:180].min(),40)
    def test_dark_frayed_front_is_selected_without_replacing_healthy_floor(self):
        rgb=np.full((300,240,3),(30,34,38),np.uint8);rgb[282:]=(7,8,10);rgb[291:,:120]=(90,50,20)
        available=np.ones((300,240),bool);available[250:295,100:140]=False
        mask=front_edge_extension(rgb,available,210)>0
        self.assertTrue(mask[299,20]);self.assertFalse(mask[250,20])
        self.assertFalse(mask[280,120]);self.assertLess(mask.mean(),.10)
    def test_clean_floor_without_material_edge_is_preserved(self):
        rgb=np.full((300,240,3),(50,54,58),np.uint8)
        for y in range(250,300):rgb[y]-=np.uint8((y-250)//8)
        self.assertFalse(front_edge_extension(rgb,np.ones((300,240),bool),210).any())
    def test_floor_extension_keeps_a_dark_lighting_gradient(self):
        rgb=np.full((300,240,3),90,np.uint8);rgb[:,:80]=30;rgb[280:]=210
        empty=np.zeros((300,240),np.uint8);extension=empty.copy();extension[280:]=255
        from floor import encode_mask
        plan={'version':1,'shape':[300,240],'defects':encode_mask(empty),'extension':encode_mask(extension),
              'color':[90,90,90],'engine':'clone-v1'}
        out=np.asarray(apply_floor_patches(Image.fromarray(rgb),.7,plan,np.zeros_like(empty,bool)))
        np.testing.assert_array_equal(out[290,10],[30,30,30]);np.testing.assert_array_equal(out[250],rgb[250])
    def scene(self):
        rgb = np.full((300, 240, 3), 90, np.uint8)
        for y in range(165, 270):
            rgb[y] += y % 3
        rgb[292:] = (205, 180, 140)
        rgb[205:208, 30:33] = 20
        rgb[180:240, 90:150] = 220
        protected = np.zeros((300, 240), bool)
        protected[175:250, 85:155] = True
        return rgb, protected

    def test_patches_marks_and_mat_edge_without_replacing_healthy_floor(self):
        rgb, protected = self.scene()
        plan = plan_floor_patches(rgb, protected, .55)
        defects = decode_mask(plan["defects"], protected.shape)
        extension = decode_mask(plan["extension"], protected.shape)
        self.assertGreater(np.count_nonzero(defects), 0)
        self.assertGreater(np.count_nonzero(extension), 0)
        result = np.asarray(apply_floor_patches(Image.fromarray(rgb), .55, plan))
        unchanged = (defects == 0) & (extension == 0)
        np.testing.assert_array_equal(result[unchanged], rgb[unchanged])
        np.testing.assert_array_equal(result[protected], rgb[protected])
        self.assertGreater(result[206, 31, 0], 80)
        self.assertLess(result[296, 10, 0], 100)
        np.testing.assert_array_equal(result[230, 200], rgb[230, 200])

    def test_saved_patch_cannot_overlap_a_person_or_prop(self):
        rgb, protected = self.scene()
        plan = plan_floor_patches(rgb, protected, .55)
        mask = decode_mask(plan["defects"], protected.shape)
        protected[mask > 0] = True
        with self.assertRaisesRegex(ValueError, "protected"):
            validate_floor_patches(protected, .55, plan)

    def test_texture_is_retained_when_no_confident_defect_is_found(self):
        yy, xx = np.mgrid[:300, :240]
        texture = ((xx + yy) % 2 * 80 + 100).astype(np.uint8)
        rgb = np.repeat(texture[..., None], 3, axis=2)
        plan = plan_floor_patches(rgb, np.zeros((300, 240), bool), .55)
        result = np.asarray(apply_floor_patches(Image.fromarray(rgb), .55, plan))
        np.testing.assert_array_equal(result, rgb)

    def test_large_floor_gap_is_flagged_instead_of_replacing_floor(self):
        rgb, protected = self.scene()
        rgb[260:] = (205, 180, 140)
        plan = plan_floor_patches(rgb, protected, .55)
        self.assertIn("review", plan["review"])
        extension = decode_mask(plan["extension"], protected.shape)
        self.assertFalse(extension.any())

