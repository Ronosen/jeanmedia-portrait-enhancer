import unittest
import numpy as np
from edge_refinement import refine,foreground_colours

class EdgeRefinementTests(unittest.TestCase):
 def test_solver_only_changes_uncertain_pixels_and_is_bounded(self):
  a=np.zeros((32,32),np.float32);a[8:24,8:24]=1;a[7,8:24]=.5
  rgb=np.full((32,32,3),80,np.uint8);before=a.copy()
  def solver(colour,trimap,**kwargs):return np.ones(trimap.shape)
  out,report=refine(rgb,a,solver)
  np.testing.assert_array_equal(a,before)
  np.testing.assert_array_equal(out[a==0],a[a==0]);np.testing.assert_array_equal(out[a==1],a[a==1])
  self.assertAlmostEqual(float(out[7,10]),.75);self.assertGreater(report['changedPixels'],0)
 def test_solid_matte_never_calls_solver(self):
  def forbidden(*args,**kwargs):raise AssertionError('Should not refine an opaque mask')
  a=np.ones((24,24),np.float32)
  out,report=refine(np.zeros((24,24,3),np.uint8),a,forbidden)
  np.testing.assert_array_equal(out,a);self.assertEqual(report['tiles'],0)
 def test_nonfinite_solver_keeps_original_edge_and_flags_review(self):
  a=np.zeros((32,32),np.float32);a[8:24,8:24]=1;a[7,8:24]=.5
  out,report=refine(np.zeros((32,32,3),np.uint8),a,lambda rgb,trimap,**kw:np.full(trimap.shape,np.nan))
  np.testing.assert_array_equal(out,a);self.assertTrue(report['review'])
 def test_invalid_foreground_estimate_is_rejected(self):
  with self.assertRaises(ValueError):foreground_colours(np.zeros((10,10,3),np.uint8),np.ones((10,10)),lambda rgb,a:np.zeros((2,2)))
 def test_foreground_estimation_has_validated_clipped_range(self):
  result=foreground_colours(np.zeros((10,10,3),np.uint8),np.ones((10,10)),lambda rgb,a:np.full(rgb.shape,1.2))
  self.assertEqual(result.min(),255);self.assertEqual(result.max(),255)

if __name__=='__main__':unittest.main()
