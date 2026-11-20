import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from studio import prepare_backdrop, render_backdrop


class StudioBackdropTests(unittest.TestCase):
    def test_short_footrest_contact_is_preserved_but_detached_small_box_is_not(self):
        from studio import support_shapes
        rgb=np.full((400,300,3),30,np.uint8)
        person=np.zeros((400,300),bool);person[50:321,120:140]=True
        rgb[person]=(90,65,45)
        rgb[321:343,95:150]=190
        rgb[321:343,210:265]=190
        _,found,support=support_shapes(rgb,person,person.copy(),np.full_like(rgb,30,dtype=float),.75)
        self.assertTrue(found);self.assertTrue(support[335,120])
        self.assertFalse(support[335,235])

    def test_floor_refinement_retains_an_already_confirmed_seat_surface(self):
        from types import SimpleNamespace
        from studio import clean_studio
        from backdrop_finish import pack,unpack
        image=Image.new('RGB',(40,60),(30,30,30));pp=np.zeros((60,40),np.float32);pp[5:50,10:30]=1
        first=np.zeros((60,40),np.uint8);first[35:55,15:30]=255
        second=first.copy();second[35:40]=0
        def prepared(floor,support):
            return {'version':3,'floor':floor,'layers':{'shape':[60,40],'support':pack(support)}},np.zeros_like(pp),[]
        def separate(plan,*args):
            self.assertEqual(unpack(plan['layers']['support'],[60,40])[37,22],255)
            return np.zeros_like(pp)
        with patch('background.person_probability',return_value=pp),patch('foreground.foreground_probability',return_value=pp), \
             patch('matting.MODEL',SimpleNamespace(is_file=lambda:True)),patch('matting.foreground_alpha',return_value=pp), \
             patch('equipment.detect_equipment',return_value=None),patch('objects.detect_objects',return_value=[{'class':0,'confidence':.9,'box':[10,5,30,50]}]), \
             patch('studio.prepare_backdrop',side_effect=[prepared(.4,first),prepared(.8,second)]), \
             patch('scene_evidence.floor_evidence',return_value=(.8,{'detected':True})), \
             patch('separation.prepare',side_effect=separate),patch('studio.render_backdrop',return_value=image):
            clean_studio(image,image,floor_mode='patch')

    def test_floor_witness_keeps_gowns_and_seats_but_excludes_reflections_below_people(self):
        from studio import separated_floor_witness
        pp=np.zeros((300,240),np.float32);pp[30:70,80:130]=1;pp[120:135,80:95]=1
        detailed=np.zeros_like(pp);detailed[30:240,80:150]=1;detailed[260:280,80:150]=1;detailed[100:230,20:50]=1;detailed[:15]=1
        support=np.zeros_like(pp,bool);support[170:255,150:210]=True
        witness=separated_floor_witness(pp,detailed,[{'class':0,'confidence':.9,'box':[70,20,160,245]},{'class':0,'confidence':.3,'box':[20,100,100,235]}],support)
        self.assertTrue(witness[220,100]);self.assertTrue(witness[200,30]);self.assertTrue(witness[230,190]);self.assertFalse(witness[270,100]);self.assertFalse(witness[5,5])

    def test_bright_support_cut_by_bottom_frame_is_kept(self):
        from studio import support_shapes
        rgb=np.full((400,300,3),30,np.uint8);rgb[290:,65:245]=170
        person=np.zeros((400,300),bool);person[65:300,90:145]=True
        _,found,confirmed=support_shapes(rgb,person,person.copy(),np.full_like(rgb,30,dtype=float),1)
        self.assertTrue(found);self.assertTrue(confirmed[390,200]);self.assertFalse(confirmed[390,270])

    def test_cropped_seat_colour_does_not_hide_a_valid_wall_family(self):
        from studio import fit_backdrop
        rgb=np.full((400,300,3),45,np.uint8)
        protected=np.zeros((400,300),bool);protected[95:320,30:270]=True
        # More samples occupy one seat bin; wall shading spans several bins.
        rgb[320:]=165
        rgb[:95]=np.linspace(25,65,95).astype(np.uint8)[:,None,None]
        rgb[95:320,:30]=35;rgb[95:320,270:]=60
        coefficients,low,high,model=fit_backdrop(rgb,protected,1)
        self.assertLess(float(high.max()),90)

    def test_tiny_backdrop_rechecks_floor_with_independent_foreground_evidence(self):
        from studio import BackdropExtentError,fit_backdrop
        image,probability,objects=self.scene();rgb=np.asarray(image)
        candidates=[(.2,{'detected':True,'row':160}),(.8125,{'detected':True,'row':650})]
        with patch('scene_evidence.floor_evidence',side_effect=candidates) as detect:
            plan,_,_=prepare_backdrop(rgb,probability,objects,foreground_probability=probability,
                                      floor_probability=probability)
        self.assertEqual(detect.call_count,2)
        self.assertTrue(detect.call_args.kwargs['require_coherence'])
        self.assertEqual(plan['floor'],.8125)
        self.assertEqual(plan['floorAnalysis']['validation'],'foreground-consensus-retry')
        # The original protection and minimum fit coverage still apply.
        with self.assertRaises(BackdropExtentError):fit_backdrop(rgb,np.ones(probability.shape,bool),.2)

    def test_complete_support_edge_is_preserved_instead_of_disjoint_colour_patches(self):
        from studio import support_shapes
        rgb=np.full((300,240,3),30,np.uint8);rgb[120:240,60:180]=220;rgb[120:135,95:125]=35
        person=np.zeros((300,240),bool);person[60:160,20:65]=True
        _,found,confirmed=support_shapes(rgb,person,person.copy(),np.full_like(rgb,30,dtype=float),.8)
        self.assertTrue(found);self.assertTrue(confirmed[125,110]);self.assertFalse(confirmed[270,110])
    def test_unsupported_scenes_preserve_pixels_and_replay_with_valid_masks(self):
        from studio import clean_studio
        from adaptive import auto_edit
        from service import change_mask
        image=Image.new('RGB',(120,180),(130,90,65))
        probability=np.ones((180,120),np.float32)
        for reason in ('A plain, consistent studio backdrop could not be identified.',
                       'The supporting seat could not be protected.'):
            with patch('background.person_probability',return_value=probability), \
                 patch('foreground.foreground_probability',return_value=probability), \
                 patch('matting.foreground_alpha',return_value=probability), \
                 patch('equipment.detect_equipment',return_value=None), \
                 patch('objects.detect_objects',return_value=[]), \
                 patch('studio.prepare_backdrop',side_effect=ValueError(reason)):
                full,after,plan,status=auto_edit(image,image,True)
            self.assertEqual(status,'Needs review')
            self.assertEqual(plan['adaptiveRevision'],3)
            self.assertEqual(plan['processingRevision'],19)
            self.assertEqual(image.tobytes(),after.tobytes())
            self.assertEqual(change_mask(plan).getbbox(),None)
            _,replay,_=clean_studio(image,image,plan,export=True)
            self.assertEqual(image.tobytes(),replay.tobytes())

    def scene(self):
        image = Image.new("RGB", (600, 800), (150, 155, 160))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 599, 40), fill=(120, 80, 30))
        draw.rectangle((0, 0, 35, 650), fill=(100, 65, 25))
        draw.ellipse((65, 110, 135, 180), fill=(70, 20, 100))
        draw.rectangle((0, 650, 599, 799), fill=(75, 80, 85))
        draw.rectangle((240, 170, 359, 700), fill=(190, 40, 30))
        draw.rectangle((340, 570, 470, 750), fill=(235, 235, 235))
        probability = np.zeros((800, 600), np.float32)
        probability[170:701, 240:360] = .99
        objects = [{"class": 56, "confidence": .8, "box": [340, 570, 471, 751]}]
        return image, probability, objects

    def test_fills_gaps_and_clutter_while_preserving_person_seat_and_floor(self):
        image, probability, objects = self.scene()
        original = np.asarray(image)
        plan, alpha, _ = prepare_backdrop(original, probability, objects)
        rendered = np.asarray(render_backdrop(image, alpha, plan))
        np.testing.assert_array_equal(rendered[probability > .9], original[probability > .9])
        np.testing.assert_array_equal(rendered[570:751, 340:471], original[570:751, 340:471])
        floor = round(plan["floor"] * image.height)
        np.testing.assert_array_equal(rendered[floor:], original[floor:])
        self.assertLess(np.abs(rendered[15, 180].astype(float) - [150, 155, 160]).max(), 8)
        self.assertLess(np.abs(rendered[140, 100].astype(float) - [150, 155, 160]).max(), 8)
        np.testing.assert_array_equal(np.asarray(image), original)

    def test_floor_option_cleans_edges_and_preserves_contact_corridor(self):
        image, probability, objects = self.scene()
        original = np.asarray(image)
        plan, alpha, _ = prepare_backdrop(original, probability, objects, clean_floor=True)
        rendered = np.asarray(render_backdrop(image, alpha, plan))
        self.assertGreater(rendered[780, 10, 0], 130)
        np.testing.assert_array_equal(rendered[570:751, 340:471], original[570:751, 340:471])
        np.testing.assert_array_equal(rendered[760:, 240:360], original[760:, 240:360])

    def test_unrecognised_supporting_box_is_preserved(self):
        image, probability, _ = self.scene()
        original = np.asarray(image)
        plan, alpha, _ = prepare_backdrop(original, probability, [], clean_floor=True)
        rendered = np.asarray(render_backdrop(image, alpha, plan))
        np.testing.assert_array_equal(rendered[570:751, 340:471], original[570:751, 340:471])

    def test_dark_boot_missed_by_person_mask_is_preserved(self):
        image, probability, objects = self.scene()
        ImageDraw.Draw(image).polygon(
            [(350, 530), (430, 535), (560, 615), (555, 640), (355, 630)],
            fill=(25, 25, 25))
        original = np.asarray(image)
        plan, alpha, _ = prepare_backdrop(original, probability, objects)
        rendered = np.asarray(render_backdrop(image, alpha, plan))
        boot = np.zeros(probability.shape, np.uint8)
        boot_image = Image.fromarray(boot)
        ImageDraw.Draw(boot_image).polygon(
            [(350, 530), (430, 535), (560, 615), (555, 640), (355, 630)], fill=1)
        np.testing.assert_array_equal(rendered[np.asarray(boot_image) > 0],
                                      original[np.asarray(boot_image) > 0])

    def test_seat_connected_to_trim_above_hip_is_preserved(self):
        image, probability, _ = self.scene()
        draw = ImageDraw.Draw(image)
        draw.rectangle((215, 280, 238, 585), fill=(235, 235, 235))
        draw.rectangle((215, 570, 470, 590), fill=(235, 235, 235))
        original = np.asarray(image)
        plan, alpha, _ = prepare_backdrop(original, probability, [])
        rendered = np.asarray(render_backdrop(image, alpha, plan))
        np.testing.assert_array_equal(rendered[570:751, 340:471], original[570:751, 340:471])

    def test_saved_backdrop_settings_replay_exactly(self):
        image, probability, objects = self.scene()
        original = np.asarray(image)
        plan, alpha, _ = prepare_backdrop(original, probability, objects)
        saved, replay, _ = prepare_backdrop(original, probability, objects, plan=plan)
        np.testing.assert_array_equal(alpha, replay)
        self.assertEqual(render_backdrop(image, alpha, plan).tobytes(),
                         render_backdrop(image, replay, saved).tobytes())

    def test_detailed_cleanup_fills_floor_and_side_gaps_with_foreground_protected(self):
        image, probability, _ = self.scene()
        draw = ImageDraw.Draw(image)
        boot = [(350, 530), (430, 535), (560, 615), (555, 640), (355, 630)]
        draw.polygon(boot, fill=(25, 25, 25))
        detailed = Image.fromarray((probability * 255).astype(np.uint8))
        ImageDraw.Draw(detailed).polygon(boot, fill=220)
        foreground = np.asarray(detailed).astype(np.float32) / 255
        original = np.asarray(image)
        plan, alpha, _ = prepare_backdrop(original, probability, [], clean_floor=True,
                                          foreground_probability=foreground)
        self.assertEqual(plan["version"], 3)
        rendered = np.asarray(render_backdrop(image, alpha, plan))
        np.testing.assert_array_equal(rendered[foreground > .5], original[foreground > .5])
        np.testing.assert_array_equal(rendered[575:745, 370:465], original[575:745, 370:465])
        self.assertGreater(rendered[780, 300, 0], 130)
        self.assertGreater(rendered[500, 10, 0], 130)
        _, replay, _ = prepare_backdrop(original, probability, [], plan=plan,
                                        foreground_probability=foreground)
        np.testing.assert_array_equal(alpha, replay)

    def test_detailed_cleanup_skips_disagreement_on_person(self):
        image, probability, objects = self.scene()
        with self.assertRaisesRegex(ValueError, "disagree"):
            prepare_backdrop(np.asarray(image), probability, objects,
                             foreground_probability=np.zeros_like(probability))

    def test_edge_refinement_removes_background_rim_without_changing_subject(self):
        image = Image.new("RGB", (400, 600), (230, 230, 230))
        ImageDraw.Draw(image).rectangle((140, 120, 259, 529), fill=(40, 45, 50))
        probability = np.zeros((600, 400), np.float32)
        probability[120:530, 140:260] = .99
        foreground = np.zeros_like(probability)
        foreground[110:540, 130:270] = .5
        foreground[120:530, 140:260] = .99
        plan = {"version": 3, "coefficients": [[160, 160, 160]] + [[0, 0, 0]] * 5,
                "low": [160, 160, 160], "high": [160, 160, 160],
                "floor": .8, "cleanFloor": False}
        saved, alpha, _ = prepare_backdrop(np.asarray(image), probability, [], plan=plan,
                                           foreground_probability=foreground)
        rendered = np.asarray(render_backdrop(image, alpha, saved))
        np.testing.assert_array_equal(rendered[120:530, 140:260], np.asarray(image)[120:530, 140:260])
        np.testing.assert_array_equal(rendered[250, 131], [160, 160, 160])

    def test_detailed_cleanup_skips_unresolved_seated_support(self):
        image = Image.new("RGB", (600, 800), (150, 155, 160))
        ImageDraw.Draw(image).rectangle((180, 170, 420, 600), fill=(35, 40, 45))
        probability = np.zeros((800, 600), np.float32)
        probability[170:601, 180:421] = .99
        people = [{"class": 0, "confidence": .9, "box": [170, 160, 440, 620]}]
        with self.assertRaisesRegex(ValueError, "supporting seat"):
            prepare_backdrop(np.asarray(image), probability, people, clean_floor=True,
                             foreground_probability=probability)

    def test_adaptive_seat_uncertainty_does_not_turn_background_into_foreground(self):
        from backdrop_finish import unpack
        image=Image.new('RGB',(600,800),(150,155,160))
        ImageDraw.Draw(image).rectangle((180,170,420,600),fill=(35,40,45))
        probability=np.zeros((800,600),np.float32);probability[170:601,180:421]=.99
        people=[{'class':0,'confidence':.9,'box':[170,160,440,620]}]
        plan,_,_=prepare_backdrop(np.asarray(image),probability,people,foreground_probability=probability,floor_mode='patch')
        self.assertTrue(plan['supportUncertain'])
        self.assertFalse(unpack(plan['layers']['support'],[800,600]).any())

    def test_unconfident_subject_is_skipped(self):
        image, probability, objects = self.scene()
        with self.assertRaisesRegex(ValueError, "person"):
            prepare_backdrop(np.asarray(image), np.zeros_like(probability), objects)

    def test_isolated_uncertain_background_blob_is_cleaned(self):
        image, probability, objects = self.scene()
        probability[110:181, 65:136] = .06
        plan, alpha, _ = prepare_backdrop(np.asarray(image), probability, objects)
        rendered = np.asarray(render_backdrop(image, alpha, plan))
        self.assertLess(np.abs(rendered[140, 100].astype(float) - [150, 155, 160]).max(), 8)
