import unittest
import numpy as np
import cv2
from cropped_backdrop import assess,frame_intrusions

class CroppedBackdropTests(unittest.TestCase):
 def fixture(self):
  rgb=np.full((400,300,3),190,np.uint8);body=np.zeros((400,300),bool);body[80:,110:200]=True;rgb[body]=(140,80,50)
  cv2.fillPoly(rgb,[np.array([[260,0],[299,0],[299,180],[275,130]])],(15,15,15))
  rgb[:,:20]=(140,110,65)
  return rgb,body,np.full(rgb.shape,190,float)
 def test_cropped_scene_removes_measured_gear_and_wall_without_person_overlap(self):
  rgb,body,ref=self.fixture();wall,gear,info=assess(rgb,body,ref)
  self.assertTrue(info['detected']);self.assertTrue(gear[50,285]);self.assertTrue(gear[200,10]);self.assertFalse((gear&body).any());self.assertTrue(wall[150,70])
 def test_full_body_unknown_floor_keeps_original_safety_fallback(self):
  rgb,body,ref=self.fixture();body[350:]=False
  wall,gear,info=assess(rgb,body,ref);self.assertFalse(info['detected']);self.assertFalse(wall.any());self.assertFalse(gear.any())
 def test_dark_backdrop_with_floor_is_not_promoted_to_white_wall(self):
  rgb,body,ref=self.fixture();ref[:]=25
  wall,gear,info=assess(rgb,body,ref);self.assertFalse(wall.any());self.assertFalse(gear.any())
 def test_subject_connected_to_frame_is_not_removed_as_equipment(self):
  rgb,body,ref=self.fixture();body[:,250:]=True
  wall,gear,info=assess(rgb,body,ref);self.assertFalse(gear[:,250:].any())
 def test_exposed_top_border_does_not_need_a_visible_floor(self):
  rgb=np.full((200,180,3),25,np.uint8);rgb[:18]=(150,120,80);body=np.zeros((200,180),bool);body[:40,75:100]=True
  gear=frame_intrusions(rgb,body,np.full(rgb.shape,25,float))
  self.assertTrue(gear[8,20]);self.assertFalse((gear&body).any());self.assertFalse(gear[100:].any())
 def test_side_panel_below_corner_equipment_is_not_required_to_touch_top(self):
  rgb=np.full((300,200,3),25,np.uint8);rgb[60:260,:16]=190;body=np.zeros((300,200),bool)
  gear=frame_intrusions(rgb,body,np.full(rgb.shape,25,float))
  self.assertTrue(gear[150,8]);self.assertFalse(gear[280:].any())

if __name__=='__main__':unittest.main()
