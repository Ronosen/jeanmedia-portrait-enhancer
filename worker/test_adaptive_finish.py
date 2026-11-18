import unittest
from unittest.mock import patch
import numpy as np
import cv2
from PIL import Image
from photo import correction
from backdrop_finish import finish,pack,unpack
from service import change_mask
from support_geometry import box_seats


class AdaptiveFinishTests(unittest.TestCase):
    def test_seamless_white_paper_has_no_artificial_horizon_band(self):
        rgb=np.full((180,140,3),210,np.uint8);body=np.zeros((180,140),np.uint8);body[35:135,45:95]=255
        rgb[body>0]=(120,60,40)
        plan={'adaptiveRevision':3,'separation':{'version':1},'floor':.65,
              'layers':{'shape':[180,140],'protected':pack(body),'backdrop':pack(np.zeros_like(body))}}
        out=np.asarray(finish(Image.fromarray(rgb),plan))
        self.assertLess(abs(int(out[100,20,0])-int(out[140,20,0])),4)
        np.testing.assert_array_equal(out[body>0],rgb[body>0])
    def test_separated_white_backdrop_is_finished_up_to_subject_edge(self):
        rgb=np.full((180,140,3),210,np.uint8);body=np.zeros((180,140),np.uint8);body[35:135,45:95]=255
        rgb[body>0]=(120,60,40)
        plan={'adaptiveRevision':3,'separation':{'version':1},'floor':.95,
              'layers':{'shape':[180,140],'protected':pack(body),'backdrop':pack(np.zeros_like(body))}}
        out=np.asarray(finish(Image.fromarray(rgb),plan))
        np.testing.assert_array_equal(out[body>0],rgb[body>0])
        self.assertLess(abs(int(out[80,42,0])-int(out[80,20,0])),4)
        self.assertGreater(out[80,42,0],225)
    def test_background_inside_oversized_guard_is_refined_without_cutting_subject(self):
        rgb=np.full((180,140,3),40,np.uint8);body=np.zeros((180,140),np.uint8);body[35:135,45:95]=255
        rgb[body>0]=(160,100,70)
        guard=cv2.dilate(body,np.ones((15,15),np.uint8))
        plan={'adaptiveRevision':3,'floor':.95,'layers':{'shape':[180,140],'source':pack(rgb),'protected':pack(guard),'backdrop':pack(np.zeros_like(body))}}
        out=np.asarray(finish(Image.fromarray(rgb),plan))
        np.testing.assert_array_equal(out[body>0],rgb[body>0])
        self.assertLess(out[80,42,0],32)
    def test_dark_backdrop_has_no_untreated_bright_band_beside_subject(self):
        rgb=np.full((180,140,3),40,np.uint8)
        subject=np.zeros((180,140),np.uint8);subject[30:140,50:90]=255
        rgb[subject>0]=(150,80,45)
        plan={'adaptiveRevision':3,'floor':.9,'layers':{'shape':[180,140],
              'protected':pack(subject),'backdrop':pack(np.zeros_like(subject))}}
        out=np.asarray(finish(Image.fromarray(rgb),plan))
        np.testing.assert_array_equal(out[subject>0],rgb[subject>0])
        self.assertLess(out[80,48,0],32)
        self.assertLess(abs(int(out[80,48,0])-int(out[80,25,0])),3)
    def test_implausibly_dark_floor_fill_is_rejected_and_cached_replay_keeps_source(self):
        from neural import repair
        from floor import encode_mask
        rgb=np.full((300,220,3),190,np.uint8)
        extension=np.zeros((300,220),np.uint8);extension[278:]=255
        selected=extension>0;protected=np.zeros_like(selected)
        plan={'adaptiveRevision':2,'floor':.75,'floorRepairs':{'extension':encode_mask(extension),
            'defects':encode_mask(np.zeros_like(extension)),'review':'edge','count':1}}
        class BadFloor:
            def run(self,*args):return [np.full((1,3,512,512),30,np.float32)]
        with patch('neural.model_session',return_value=BadFloor()):out=repair(Image.fromarray(rgb),selected,protected,plan)
        self.assertTrue(plan['floorFillRejected'])
        np.testing.assert_array_equal(np.asarray(out),rgb)
        self.assertEqual(plan['floorRepairs']['count'],0)
        self.assertIn('rejected',plan['floorRepairs']['review'])

    def test_floor_detects_dark_mat_tape_between_similar_light_materials(self):
        from floor import plan_floor_patches,decode_mask
        rgb=np.full((600,400,3),190,np.uint8);rgb[565:574]=80;rgb[580:]=(205,190,175)
        p=plan_floor_patches(rgb,np.zeros((600,400),bool),.82,context=True)
        mask=decode_mask(p['extension'],(600,400))>0
        self.assertTrue(mask[570,30]);self.assertTrue(mask[595,30]);self.assertFalse(mask[520,30])

    def test_scene_colours_cannot_trigger_independent_white_balance_gains(self):
        rgb=np.full((180,140,3),(140,150,160),np.uint8)
        lut=correction(Image.fromarray(rgb))
        np.testing.assert_array_equal(lut[:,0],lut[:,1])
        np.testing.assert_array_equal(lut[:,1],lut[:,2])

    def test_finish_reduces_wrinkle_preserves_foreground_floor_grain_and_replays(self):
        rng=np.random.default_rng(8);h,w=300,220
        yy,xx=np.mgrid[:h,:w]
        base=110+xx*.05+12*np.sin(xx/9)
        rgb=np.repeat(base[...,None],3,2)+rng.normal(0,1,(h,w,1))
        rgb=np.rint(np.clip(rgb,0,255)).astype(np.uint8)
        protected=np.zeros((h,w),bool);protected[80:240,80:140]=True
        rgb[protected]=(190,45,35);rgb[250:]=(50,55,60)
        plan={'adaptiveRevision':2,'floor':.83,'layers':{'shape':[h,w],
            'protected':pack(protected.astype(np.uint8)*255),'backdrop':pack(np.zeros((h,w),np.uint8))}}
        image=Image.fromarray(rgb);out=np.asarray(finish(image,plan))
        np.testing.assert_array_equal(out[protected],rgb[protected])
        np.testing.assert_array_equal(out[250:],rgb[250:])
        mask=np.asarray(change_mask(plan))>0
        np.testing.assert_array_equal(out[~mask],rgb[~mask])
        self.assertGreater(np.count_nonzero(out!=rgb),0)
        self.assertLess(out[30:60,30:190,0].std(),rgb[30:60,30:190,0].std())
        self.assertEqual(finish(image,plan).tobytes(),out.tobytes())

    def test_box_geometry_recovers_bright_seat_not_a_frame_panel(self):
        import cv2
        rgb=np.full((600,400,3),180,np.uint8)
        person=np.zeros((600,400),bool);person[100:400,150:220]=True
        rgb[person]=(200,50,30)
        cv2.rectangle(rgb,(160,380),(360,530),(225,225,225),-1)
        cv2.rectangle(rgb,(160,380),(360,530),(140,140,140),2)
        mask=box_seats(rgb,person)
        self.assertTrue(mask[480,330])
        self.assertFalse(mask[200,30])


