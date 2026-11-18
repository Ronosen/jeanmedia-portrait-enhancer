import base64
import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from neural import repair
from service import dispatch
from floor import encode_mask, decode_mask, plan_floor_patches, apply_floor_patches


class FakePrediction:
    def run(self,outputs,inputs):
        assert inputs["image"].shape==(1,3,512,512)
        assert inputs["mask"].shape==(1,1,512,512)
        assert inputs["image"].min()>=0 and inputs["image"].max()<=1
        result=np.zeros((1,3,512,512),np.float32)
        result[:,0]=200;result[:,1]=50;result[:,2]=80
        return [result]


class ReconstructionTests(unittest.TestCase):
    def test_narrow_bottom_mat_lip_is_repaired_without_selecting_healthy_floor(self):
        rgb=np.full((600,400,3),70,np.uint8)
        rgb[594:,220:]=220
        protected=np.zeros((600,400),bool)
        plan=plan_floor_patches(rgb,protected,.7,context=True)
        selected=decode_mask(plan['extension'],protected.shape)>0
        self.assertTrue(selected[598,300])
        self.assertFalse(selected[550,300])
        self.assertLess(selected.mean(),.025)

    def test_even_a_bad_prediction_cannot_change_unselected_people_or_props(self):
        rgb=np.full((80,60,3),90,np.uint8)
        rgb[20:70,25:50]=(40,100,180)
        protected=np.zeros((80,60),bool);protected[20:70,25:50]=True
        mask=np.zeros_like(protected);mask[10:45,:15]=True
        settings={}
        with patch("neural.model_session",return_value=FakePrediction()):
            result=repair(Image.fromarray(rgb),mask,protected,settings)
        out=np.asarray(result)
        np.testing.assert_array_equal(out[~mask],rgb[~mask])
        np.testing.assert_array_equal(out[protected],rgb[protected])
        np.testing.assert_array_equal(out[mask],np.tile([200,50,80],(np.count_nonzero(mask),1)))
        with patch("neural.model_session",side_effect=AssertionError("Cached replay must not regenerate")):
            replay=repair(Image.fromarray(rgb),mask,protected,settings)
        self.assertEqual(result.tobytes(),replay.tobytes())
        mask[30,30]=True
        with self.assertRaisesRegex(ValueError,"protected"):
            repair(Image.fromarray(rgb),mask,protected,{})

    def test_change_view_is_an_outline_not_a_coloured_cover(self):
        mask=np.zeros((80,60),np.uint8);mask[10:60,5:30]=255
        plan={"layers":{"shape":[80,60],"backdrop":encode_mask(mask)}}
        result=dispatch("mask",{"studioPlan":plan})
        image=Image.open(io.BytesIO(base64.b64decode(result["url"].split(",")[1])))
        self.assertEqual(image.mode,"RGBA")
        self.assertEqual(image.getpixel((15,30))[3],0)
        self.assertGreater(image.getpixel((5,30))[3],0)

    def test_large_mat_edge_is_local_and_keeps_subject_and_healthy_floor(self):
        rng=np.random.default_rng(4)
        rgb=np.full((300,240,3),90,np.uint8)+rng.integers(0,4,(300,240,3),np.uint8)
        rgb[260:]=(205,180,140)
        protected=np.zeros((300,240),bool);protected[170:225,90:150]=True
        rgb[protected]=(220,220,220)
        plan=plan_floor_patches(rgb,protected,.55,context=True)
        extension=decode_mask(plan["extension"],protected.shape)>0
        defects=decode_mask(plan["defects"],protected.shape)>0
        self.assertGreater(np.count_nonzero(extension),0)
        out=np.asarray(apply_floor_patches(Image.fromarray(rgb),.55,plan,protected))
        np.testing.assert_array_equal(out[~(extension|defects)],rgb[~(extension|defects)])
        np.testing.assert_array_equal(out[protected],rgb[protected])
        self.assertLess(out[290,20,0],120)
