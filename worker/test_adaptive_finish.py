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
    def test_dark_saturated_backdrop_does_not_receive_neutral_black_fold_treatment(self):
        import copy
        rgb=np.full((180,140,3),(15,20,150),np.uint8);empty=np.zeros((180,140),np.uint8)
        plan={'adaptiveRevision':3,'separation':{'version':1},'floor':.8,
              'layers':{'shape':[180,140],'protected':pack(empty),'backdrop':pack(empty)}}
        updated=copy.deepcopy(plan);updated['separation']['backdropMethod']='multiscale-v1'
        expected=finish(Image.fromarray(rgb),plan);actual=finish(Image.fromarray(rgb),updated)
        self.assertEqual(actual.tobytes(),expected.tobytes());self.assertEqual(updated['backdropFinish']['foldFinish'],0)

    def test_broad_lower_black_backdrop_folds_are_removed_without_touching_floor(self):
        h,w=480,320;yy,xx=np.mgrid[:h,:w];rng=np.random.default_rng(9)
        light=26+8*np.exp(-((xx-w*.5)/(w*.4))**2)
        folds=12*np.sin(xx/17)*np.clip((yy/h-.58)/.16,0,1)
        rgb=np.repeat((light+folds+rng.normal(0,.8,(h,w)))[...,None],3,axis=2)
        rgb=np.rint(np.clip(rgb,0,255)).astype(np.uint8);row=round(h*.86)
        rgb[row:]=(45,49,55);body=np.zeros((h,w),np.uint8);body[80:450,135:185]=255;rgb[body>0]=(150,80,45)
        plan={'adaptiveRevision':3,'separation':{'version':1,'backdropMethod':'multiscale-v1'},'floor':.86,
              'layers':{'shape':[h,w],'protected':pack(body),'backdrop':pack(np.zeros_like(body))}}
        out=np.asarray(finish(Image.fromarray(rgb),plan))
        np.testing.assert_array_equal(out[body>0],rgb[body>0]);np.testing.assert_array_equal(out[row:],rgb[row:])
        # The last wall rows used to be excluded by the floor feather, leaving
        # the most visible folds untouched. Measure both broad ridges and grain.
        band=np.s_[row-18:row-4,20:110]
        self.assertLess(out[band][:,:,0].std(),rgb[band][:,:,0].std()*.3)
        original_grain=rgb[200:250,20:110,0].astype(float)-cv2.GaussianBlur(rgb[200:250,20:110,0].astype(float),(0,0),1)
        final_grain=out[200:250,20:110,0].astype(float)-cv2.GaussianBlur(out[200:250,20:110,0].astype(float),(0,0),1)
        self.assertGreater(final_grain.std(),original_grain.std()*.6)
        self.assertEqual(finish(Image.fromarray(rgb),plan).tobytes(),out.tobytes())

    def test_low_contrast_dark_floor_junction_is_not_the_fixed_eighty_percent_fallback(self):
        from studio import floor_boundary
        rgb=np.full((600,400,3),20,np.uint8);rgb[516:]=28
        subject=np.zeros((600,400),bool);subject[100:575,170:230]=True
        self.assertAlmostEqual(floor_boundary(rgb,subject),.86,delta=.025)

    def test_gradual_dark_illumination_does_not_create_a_new_material_edge(self):
        from studio import floor_boundary
        rgb=np.repeat(np.repeat(np.linspace(15,55,600)[:,None,None],400,axis=1),3,axis=2).astype(np.uint8)
        subject=np.zeros((600,400),bool);subject[100:575,170:230]=True
        self.assertEqual(floor_boundary(rgb,subject),1.0)

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


