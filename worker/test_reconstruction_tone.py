import unittest
import numpy as np
from PIL import Image
from backdrop_finish import pack
from reconstruction_tone import match

class ReconstructionToneTests(unittest.TestCase):
 def fixture(self):
  h,w=160,140;yy,xx=np.mgrid[:h,:w]
  src=np.repeat((25+np.sin(xx/3))[...,None],3,axis=2).astype(np.uint8)
  repaired=src.copy();clutter=np.zeros((h,w),np.uint8);clutter[:45,:35]=255;repaired[:45,:35]=np.repeat((110+8*np.sin(xx[:45,:35]/3))[...,None],3,axis=2)
  body=np.zeros((h,w),np.uint8);body[40:150,65:100]=255
  plan={'sceneIntentSkill':1,'sceneCleanupRevision':22,'coefficients':[[25]*3]+[[0]*3]*5,'low':[25]*3,'high':[25]*3,'layers':{'shape':[h,w],'protected':pack(body),'anchoredForegroundAlpha':pack(body),'clutter':pack(clutter)}}
  return src,repaired,plan,clutter,body
 def test_drifted_repair_matches_source_donors_and_retains_texture(self):
  src,repaired,plan,mask,body=self.fixture();out=np.array(match(Image.fromarray(src),Image.fromarray(repaired),plan))
  self.assertLess(out[mask>0].mean(),40);self.assertGreater(out[mask>0].std(),.3)
  np.testing.assert_array_equal(out[mask==0],repaired[mask==0]);self.assertEqual(plan['reconstructionTone']['components'],1)
 def test_floor_and_protected_source_do_not_receive_tone_correction(self):
  src,repaired,plan,mask,body=self.fixture();out=np.array(match(Image.fromarray(src),Image.fromarray(repaired),plan))
  np.testing.assert_array_equal(out[120:],repaired[120:]);np.testing.assert_array_equal(out[body>0],repaired[body>0])
 def test_existing_good_repair_is_not_retoned(self):
  src,_,plan,mask,body=self.fixture();out=match(Image.fromarray(src),Image.fromarray(src),plan)
  np.testing.assert_array_equal(np.array(out),src);self.assertEqual(plan['reconstructionTone']['components'],0)

if __name__=='__main__':unittest.main()
