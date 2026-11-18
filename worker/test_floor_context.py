import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from floor_context import reconstruct
from studio import floor_boundary

class FloorContextTests(unittest.TestCase):
 def test_local_crop_excludes_removed_wall_clutter_from_its_donor_context(self):
  source=Image.new('RGB',(100,150),(90,95,100));empty=np.zeros((150,100),bool)
  clutter=empty.copy();clutter[120:140,0:8]=True;removed=clutter.copy();removed[110:120,0:8]=True
  settings={'version':2,'blendRevision':1}
  def fill(image,holes,guard,options):
   self.assertTrue(holes[4,3]);return Image.new('RGB',image.size,(70,75,80))
  with patch('floor_context.repair',side_effect=fill):result,mask=reconstruct(source,empty,empty,clutter,.82,settings,removed)
  final=np.asarray(Image.composite(result,source,mask))
  self.assertGreater(int(final[112,3,0]),70);self.assertLess(int(final[112,3,0]),90)
  np.testing.assert_array_equal(final[112,50],[90,95,100])
 def test_crop_repairs_only_missing_floor_and_excludes_subjects(self):
  source=Image.new('RGB',(100,150),(90,95,100));extension=np.zeros((150,100),bool);extension[135:]=True
  foreground=np.zeros_like(extension);foreground[100:145,40:60]=True
  clutter=np.zeros_like(extension);clutter[120:140,0:8]=True;settings={}
  def fill(image,holes,guard,options):
   self.assertEqual(image.height,34);self.assertTrue(holes[20,50]);self.assertFalse(guard.any())
   return Image.new('RGB',image.size,(70,75,80))
  with patch('floor_context.repair',side_effect=fill):result,mask=reconstruct(source,extension,foreground,clutter,.8,settings)
  final=np.asarray(Image.composite(result,source,mask));original=np.asarray(source)
  np.testing.assert_array_equal(final[foreground],original[foreground])
  np.testing.assert_array_equal(final[:115],original[:115])
  np.testing.assert_array_equal(final[148,20],[70,75,80]);np.testing.assert_array_equal(final[125,3],[70,75,80])
  self.assertEqual(settings['row'],116)
 def test_saved_floor_crop_rejects_a_changed_horizon(self):
  empty=np.zeros((100,100),bool)
  with self.assertRaisesRegex(ValueError,'different floor'):
   reconstruct(Image.new('RGB',(100,100)),empty,empty,empty,.8,{'row':1})
 def test_material_edge_below_shoes_does_not_become_the_floor_horizon(self):
  rgb=np.full((200,120,3),220,np.uint8);rgb[180:]=(160,120,90)
  subject=np.zeros((200,120),bool);subject[30:174,40:80]=True
  self.assertLess(floor_boundary(rgb,subject),.82)
