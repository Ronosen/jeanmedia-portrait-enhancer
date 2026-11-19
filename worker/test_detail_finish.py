import unittest
import numpy as np
from PIL import Image
from unittest.mock import patch
from detail_finish import apply,measure
from portrait_enhance import prepare,apply as portrait_apply,color_tile,whiten


class DetailTests(unittest.TestCase):
    def test_noise_reduces_without_colour_cast(self):
        rng=np.random.default_rng(12)
        gray=np.clip(100+rng.normal(0,4,(100,100)),0,255).astype(np.uint8)
        rgb=np.repeat(gray[:,:,None],3,axis=2)
        out=apply(rgb,{'noise':5,'denoise':.45,'sharpen':.1},np.zeros(gray.shape))
        self.assertLess(out[:,:,0].std(),gray.std())
        self.assertEqual(np.ptp(out,axis=2).max(),0)

    def test_flat_planes_and_hard_edges_have_no_halo(self):
        rgb=np.full((60,90,3),40,np.uint8);rgb[:,45:]=200
        out=apply(rgb,{'noise':0,'denoise':0,'sharpen':.35},np.zeros((60,90)))
        np.testing.assert_array_equal(out,rgb)

    def test_noise_measurement_adapts(self):
        plain=Image.new('RGB',(400,400),(120,120,120))
        rng=np.random.default_rng(5)
        noisy=Image.fromarray(np.clip(120+rng.normal(0,8,(400,400,3)),0,255).astype(np.uint8))
        clean=measure(plain);dirty=measure(noisy)
        self.assertEqual(clean['denoise'],0)
        self.assertGreater(dirty['denoise'],clean['denoise'])
        self.assertLess(dirty['sharpen'],clean['sharpen'])

    @patch('face_areas.detect',return_value=[])
    def test_tiled_finish_matches_whole_image_and_legacy_plan(self,_):
        rng=np.random.default_rng(9)
        rgb=rng.integers(20,210,(530,80,3),dtype=np.uint8)
        image=Image.fromarray(rgb);plan=prepare(image,image)
        zeros=np.zeros(rgb.shape[:2],np.float32)
        expected=whiten(apply(color_tile(rgb,plan['tone'],zeros),plan['detail'],zeros),zeros,zeros)
        np.testing.assert_array_equal(np.asarray(portrait_apply(image,plan)),expected)
        del plan['detail']
        np.testing.assert_array_equal(np.asarray(portrait_apply(image,plan)),color_tile(rgb,plan['tone'],zeros))


if __name__=='__main__':unittest.main()
