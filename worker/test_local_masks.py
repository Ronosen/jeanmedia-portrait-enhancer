import unittest
import numpy as np
from PIL import Image
from backdrop_finish import pack,finish
from local_masks import selection,apply

class MaskTests(unittest.TestCase):
 def test_floor_surface_fill_preserves_foreground_and_healthy_floor(self):
  from floor import encode_mask,apply_floor_patches
  rgb=np.full((300,240,3),190,np.uint8);rgb[280:287]=50;rgb[287:]=(210,195,170)
  protected=np.zeros((300,240),bool);protected[230:290,90:130]=True;rgb[protected]=(40,30,20)
  mask=np.zeros((300,240),np.uint8);mask[270:]=255;mask[protected]=0
  plan={'version':1,'shape':[300,240],'engine':'surface-v1','extension':encode_mask(mask),'defects':encode_mask(np.zeros_like(mask))}
  out=np.asarray(apply_floor_patches(Image.fromarray(rgb),.75,plan,protected))
  np.testing.assert_array_equal(out[mask==0],rgb[mask==0]);np.testing.assert_array_equal(out[protected],rgb[protected])
  self.assertGreater(out[295,20].mean(),150)
 def plan(self):
  person=np.zeros((120,100),np.uint8);person[20:100,35:65]=255
  rgb=np.full((120,100,3),180,np.uint8);rgb[person>0]=(180,50,30)
  return rgb,{'adaptiveRevision':3,'floor':.85,'layers':{'shape':[120,100],'person':pack(person),'protected':pack(person),'source':pack(rgb)}}
 def test_local_subject_adjustment_never_changes_unselected_pixels(self):
  rgb,p=self.plan();layer={'target':'subject','exposure':.5,'feather':0};p['localMasks']=[layer]
  out=np.asarray(apply(Image.fromarray(rgb),p));m=selection(p,layer)>0
  np.testing.assert_array_equal(out[~m],rgb[~m]);self.assertGreater(out[m].mean(),rgb[m].mean())
 def test_brush_subtract_and_invert_and_range(self):
  rgb,p=self.plan();layer={'target':'brush','strokes':[[.5,.5,.15,'add'],[.5,.5,.04,'subtract']],'feather':0}
  m=selection(p,layer);self.assertEqual(m[60,50],0);self.assertGreater(m.sum(),0)
  layer['invert']=True;np.testing.assert_allclose(selection(p,layer),1-m)
  c=selection(p,{'target':'colour','colour':[180,50,30],'tolerance':.1,'feather':0});self.assertEqual(c[50,50],1);self.assertEqual(c[5,5],0)
 def test_white_pop_is_bounded_and_preserves_subject(self):
  rgb,p=self.plan();out=np.asarray(finish(Image.fromarray(rgb),p));self.assertGreater(out[5,5,0],rgb[5,5,0]);np.testing.assert_array_equal(out[30:90,40:60],rgb[30:90,40:60])
