import unittest
from unittest.mock import patch
import numpy as np
from scene_context import backdrop_boundary,exposed_room,object_relationship
from scene_intent import assess


class SceneContextTests(unittest.TestCase):
 def fixture(self,h=240,w=320,start=45,slope=.05):
  rgb=np.full((h,w,3),25,np.uint8);ref=rgb.copy();protected=np.zeros((h,w),bool)
  protected[h//3:h-10,w//2:w*3//4]=True
  for x in range(w):
   y=round(start+x*slope);rgb[:y,x]=(160,120,65);rgb[y:y+4,x]=225
  return rgb,ref,protected
 def test_detects_observed_sloped_rail(self):
  rgb,ref,p=self.fixture();context=backdrop_boundary(rgb,ref,p)
  self.assertTrue(context['detected']);self.assertGreater(context['rows'][-1],context['rows'][0]+8)
  room=exposed_room(p.shape,context,p)
  self.assertTrue(room[10,10]);self.assertFalse(room[130,10]);self.assertFalse((room&p).any())
 def test_boundary_moves_with_source_not_fixed_frame_row(self):
  first=backdrop_boundary(*self.fixture(start=30,slope=0))
  second=backdrop_boundary(*self.fixture(start=60,slope=0))
  self.assertTrue(first['detected'] and second['detected'])
  self.assertGreater(np.median(second['rows'])-np.median(first['rows']),25)
 def test_uniform_and_patternless_gradient_have_no_invented_rail(self):
  rgb,ref,p=self.fixture();rgb[:]=25
  self.assertFalse(backdrop_boundary(rgb,ref,p)['detected'])
 def test_above_rail_decoration_and_crossing_prop_are_different(self):
  shape=(200,240);p=np.zeros(shape,bool);p[80:190,140:200]=True
  context={'detected':True,'rows':[60]*240};part=np.zeros(shape,bool);part[5:30,20:60]=True
  self.assertTrue(object_relationship(part,p,context)['roomDecoration'])
  part[30:90,20:60]=True
  self.assertFalse(object_relationship(part,p,context)['roomDecoration'])
 def test_near_subject_prop_is_protected(self):
  shape=(200,240);p=np.zeros(shape,bool);p[30:190,100:180]=True
  part=np.zeros(shape,bool);part[10:29,90:110]=True
  self.assertFalse(object_relationship(part,p,{'detected':True,'rows':[60]*240})['roomDecoration'])
 def test_balloon_identity_does_not_override_measured_room_context(self):
  shape=(200,240);rgb=np.full((*shape,3),25,np.uint8);part=np.zeros(shape,bool);part[5:35,20:60]=True
  rgb[part]=(220,160,20);person=np.zeros(shape,bool);person[80:190,140:200]=True
  raw=part.astype(np.float32);empty=np.zeros(shape,np.uint8)
  decision={'label':'balloon','action':'protect','similarity':.32,'margin':.07}
  with patch('scene_intent.classify',return_value=decision):
   keep,remove,report=assess(rgb,raw,empty,empty,person,context={'detected':True,'rows':[60]*240})
  self.assertTrue(remove[15,30]);self.assertEqual(keep[15,30],0)
  self.assertEqual(report['objects'][0]['reason'],'detached-decoration-above-measured-backdrop')
 def test_no_scene_boundary_keeps_existing_balloon_protection(self):
  shape=(200,240);rgb=np.full((*shape,3),25,np.uint8);raw=np.zeros(shape,np.float32);raw[5:35,20:60]=1
  rgb[raw>0]=(220,160,20);person=np.zeros(shape,bool);person[80:190,140:200]=True;empty=np.zeros(shape,np.uint8)
  with patch('scene_intent.classify',return_value={'label':'balloon','action':'protect','similarity':.32,'margin':.07}):
   keep,remove,_=assess(rgb,raw,empty,empty,person,context={'detected':False})
  self.assertEqual(keep[15,30],1);self.assertFalse(remove.any())

if __name__=='__main__':unittest.main()
