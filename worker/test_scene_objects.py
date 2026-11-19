import unittest
import numpy as np
from scene_objects import detached_clutter,thin_supports,corner_equipment,frame_panel_leaks
from scene_evidence import texture_finish


class SceneObjectsTests(unittest.TestCase):
 def test_frame_cleanup_never_removes_a_detailed_hand_missed_by_the_person_model(self):
  rgb=np.full((400,300,3),25,np.uint8);rgb[:350,:30]=160
  pp=np.zeros((400,300),np.float32);pp[100:330,60:110]=1
  raw=np.zeros_like(pp);raw[100:330,60:110]=1;raw[180:200,25:70]=1
  rgb[180:200,25:70]=(137,80,60);pp[180:200,25:60]=.1
  mask=frame_panel_leaks(rgb,np.full_like(rgb,25,dtype=float),pp,np.zeros_like(pp,bool),raw)
  self.assertFalse(mask[190,27]);self.assertTrue(mask[30,10])

 def test_panel_extent_excludes_all_donor_fragments_but_keeps_detailed_clothing(self):
  rgb=np.full((400,300,3),25,np.uint8);rgb[:350,:30]=160
  pp=np.zeros((400,300),np.float32);pp[100:330,40:110]=1
  pp[180:230,15:50]=.7;rgb[180:230,15:50]=160
  raw=np.zeros_like(pp);raw[180:230,30:50]=1
  mask,extent=frame_panel_leaks(rgb,np.full_like(rgb,25,dtype=float),pp,np.zeros_like(pp,bool),raw,with_extent=True)
  self.assertTrue(mask[200,20]);self.assertFalse(mask[200,35]);self.assertTrue(extent[10,5]);self.assertTrue(extent[340,5])

 def test_frame_wall_is_removed_but_white_clothing_and_support_are_protected(self):
  rgb=np.full((400,300,3),25,np.uint8);rgb[:350,:30]=160
  rgb[:350,:8]=(180,120,70);rgb[270:300,:8]=35
  pp=np.zeros((400,300),np.float32);pp[100:330,40:110]=1
  rgb[180:230,15:50]=160;pp[180:230,15:50]=1
  support=np.zeros((400,300),bool);support[300:360,20:80]=True;rgb[support]=220
  mask=frame_panel_leaks(rgb,np.full_like(rgb,25,dtype=float),pp,support)
  self.assertTrue(mask[30,10]);self.assertTrue(mask[30,2]);self.assertFalse(mask[280,2]);self.assertFalse(mask[200,20]);self.assertFalse(mask[support].any())

 def test_light_backdrop_is_not_selected_as_a_frame_wall(self):
  rgb=np.full((400,300,3),200,np.uint8);pp=np.zeros((400,300),np.float32);pp[100:300,100:200]=1
  self.assertFalse(frame_panel_leaks(rgb,np.full_like(rgb,195,dtype=float),pp,np.zeros_like(pp,bool)).any())

 def test_overhead_corner_gear_uses_its_edge_and_preserves_a_cap(self):
  import cv2
  rgb=np.full((400,300,3),60,np.uint8)
  cv2.fillPoly(rgb,[np.array([[0,0],[115,0],[0,65]])],(5,5,5))
  protected=np.zeros((400,300),bool)
  mask=corner_equipment(rgb,protected)
  self.assertTrue(mask[10,10]);self.assertFalse(mask[100,100])
  protected[5:45,5:70]=True
  self.assertFalse(corner_equipment(rgb,protected).any())

 def test_gradual_dark_corner_is_not_selected_as_equipment(self):
  yy,xx=np.mgrid[:400,:300];gray=np.clip(10+(xx+yy)*.15,0,80).astype(np.uint8)
  self.assertFalse(corner_equipment(np.repeat(gray[...,None],3,2),np.zeros(gray.shape,bool)).any())

 def test_clutter_shadow_is_removed_without_extending_into_a_bright_floor(self):
  raw=np.zeros((320,280),np.float32);raw[20:190,30:110]=1
  rgb=np.full((320,280,3),50,np.uint8);rgb[20:190,30:110]=15;rgb[30:185,110:117]=25;rgb[190:]=80
  protected=np.zeros((320,280),bool);protected[80:290,160:240]=True
  mask,count=detached_clutter(raw,protected,[{'class':0,'confidence':.9,'box':[150,70,250,300]}],.85,rgb,np.full_like(rgb,50,dtype=float))
  self.assertEqual(count,1);self.assertTrue(mask[100,114]);self.assertFalse(mask[200,114]);self.assertFalse(mask[protected].any())

 def test_detached_garment_is_selected_whole_but_people_and_floor_reflection_are_kept(self):
  raw=np.zeros((320,280),np.float32);raw[10:190,20:90]=1;raw[80:295,150:240]=1;raw[300:310,150:210]=1
  protected=raw>.5;protected[:200,:100]=False;protected[300:]=False
  people=[{'class':0,'confidence':.9,'box':[140,70,250,300]}]
  mask,count=detached_clutter(raw,protected,people,.85)
  self.assertEqual(count,1);self.assertTrue(mask[10:190,20:90].all());self.assertFalse(mask[protected].any());self.assertFalse(mask[300:].any())

 def test_possible_extra_person_is_not_automatically_removed(self):
  raw=np.zeros((320,280),np.float32);raw[10:190,20:90]=1
  people=[{'class':0,'confidence':.9,'box':[15,5,95,200]}]
  mask,count=detached_clutter(raw,np.zeros_like(raw,bool),people,.85)
  self.assertEqual(count,0);self.assertFalse(mask.any())

 def test_held_garment_is_kept_when_upper_edge_contacts_a_person(self):
  raw=np.zeros((320,280),np.float32);raw[30:230,20:90]=1
  protected=np.zeros_like(raw,bool);protected[28:40,92:110]=True
  people=[{'class':0,'confidence':.9,'box':[100,10,250,300]}]
  mask,count=detached_clutter(raw,protected,people,.85)
  self.assertEqual(count,0);self.assertFalse(mask.any())

 def test_paired_chrome_post_is_recovered_under_the_person(self):
  rgb=np.full((320,280,3),25,np.uint8);alpha=np.zeros((320,280),np.float32)
  alpha[40:160,100:180]=1;rgb[alpha>0]=(90,60,50);rgb[159:250,135:143]=220
  rgb[170:280,25:32]=220
  mask=thin_supports(rgb,alpha,[{'class':0,'confidence':.9,'box':[60,30,220,300]}])
  self.assertTrue(mask[200,139]);self.assertFalse(mask[200,28])

 def test_floor_material_edge_and_gradual_lighting_are_not_flattened(self):
  h,w=400,280;yy,xx=np.mgrid[:h,:w];gray=(25+xx/w*8).astype(np.float32);gray[290:]+=35
  delta,_,_=texture_finish(gray,np.ones((h,w),bool))
  self.assertEqual(float(np.max(np.abs(delta[280:310,40:240]))),0)
  self.assertLess(float(np.max(np.abs(delta[:180]))),1)


if __name__=='__main__':unittest.main()
