import unittest
from unittest.mock import patch
import numpy as np
from groups import protect_group
from studio import initial_protection

class GroupProtectionTests(unittest.TestCase):
 def test_black_gowns_are_not_rejected_when_global_saliency_picks_one_person(self):
  rgb=np.full((240,300,3),120,np.uint8)
  probability=np.zeros((240,300),np.float32);probability[50:230,30:100]=.99;probability[50:230,180:250]=.99
  foreground=np.zeros_like(probability);foreground[50:230,30:100]=.9
  people=[{'class':0,'confidence':.9,'box':[20,40,110,235]},{'class':0,'confidence':.9,'box':[170,40,260,235]}]
  def local_p(crop):
   p=np.zeros(crop.shape[:2],np.float32);p[15:-10,15:-15]=.99;return p
  with patch('groups.person_probability',side_effect=local_p),patch('groups.foreground_probability',side_effect=lambda crop:np.zeros(crop.shape[:2],np.float32)):
   p,f,info=protect_group(rgb,probability,foreground,people)
  self.assertEqual(info['peopleDetected'],2)
  self.assertGreater(f[150,200],.03)
 def test_cap_corners_and_thin_gold_tassels_are_recovered(self):
  rgb=np.full((240,300,3),120,np.uint8);rgb[42:52,18:110]=20;rgb[43:65,107:109]=[200,150,10]
  p=np.zeros((240,300),np.float32);p[50:230,30:100]=.99;p[50:230,180:250]=.99
  people=[{'class':0,'confidence':.9,'box':[20,40,110,235]},{'class':0,'confidence':.9,'box':[170,40,260,235]}]
  with patch('groups.person_probability',side_effect=lambda crop:np.zeros(crop.shape[:2],np.float32)),patch('groups.foreground_probability',side_effect=lambda crop:np.zeros(crop.shape[:2],np.float32)):
   fused,foreground,info=protect_group(rgb,p,p,people)
  self.assertGreater(fused[46,20],.9)
  self.assertGreater(foreground[58,108],.3)
  self.assertEqual(info['peopleDetected'],2)
  self.assertEqual(foreground[10,150],0)
 def test_local_saliency_cannot_protect_an_entire_background_crop(self):
  rgb=np.full((240,300,3),120,np.uint8)
  p=np.zeros((240,300),np.float32);p[50:230,30:100]=.99;p[50:230,180:250]=.99
  people=[{'class':0,'confidence':.9,'box':[20,40,110,235]},{'class':0,'confidence':.9,'box':[170,40,260,235]}]
  def local_p(crop):
   result=np.zeros(crop.shape[:2],np.float32);result[15:-10,15:-15]=.99;return result
  with patch('groups.person_probability',side_effect=local_p),patch('groups.foreground_probability',side_effect=lambda crop:np.ones(crop.shape[:2],np.float32)):
   _,f,_=protect_group(rgb,p,np.ones_like(p),people)
  self.assertEqual(f[5,5],0)
  self.assertEqual(f[35,12],0)
  self.assertGreater(f[150,200],.03)
 def test_single_person_keeps_existing_path(self):
  rgb=np.zeros((60,50,3),np.uint8);p=np.zeros((60,50),np.float32);f=p.copy()
  a,b,info=protect_group(rgb,p,f,[])
  self.assertIs(a,p);self.assertIs(b,f);self.assertIsNone(info)
