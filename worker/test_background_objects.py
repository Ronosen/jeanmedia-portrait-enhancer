import unittest
import cv2
import numpy as np
from background_objects import edge_clutter

class EdgeClutterTests(unittest.TestCase):
 def test_thin_wall_panel_at_bottom_border_does_not_survive_on_dark_floor(self):
  rgb=np.full((300,240,3),30,np.uint8);protected=np.zeros((300,240),bool)
  rgb[30:265,0:3]=(225,225,225)
  self.assertTrue(edge_clutter(rgb,protected,.8)[250,1])
 def test_short_tripod_tip_cut_off_by_frame_is_removed(self):
  rgb=np.full((400,240,3),150,np.uint8);protected=np.zeros((400,240),bool)
  cv2.line(rgb,(0,363),(20,391),(15,15,15),7)
  self.assertTrue(edge_clutter(rgb,protected,.8)[380,12])
 def test_spare_box_cut_off_by_frame_is_still_detected(self):
  rgb=np.full((300,240,3),30,np.uint8);protected=np.zeros((300,240),bool)
  rgb[210:250,220:]=(225,225,225)
  self.assertTrue(edge_clutter(rgb,protected,.8)[230,235])
 def test_white_edge_box_on_dark_floor_is_not_treated_as_a_tile(self):
  rgb=np.full((300,240,3),30,np.uint8);protected=np.zeros((300,240),bool)
  rgb[250:290,215:]=(225,225,225)
  self.assertTrue(edge_clutter(rgb,protected,.8)[270,230])
 def test_spare_box_joined_to_a_white_wall_strip_is_detected(self):
  rgb=np.full((300,240,3),30,np.uint8);protected=np.zeros((300,240),bool)
  rgb[20:260,235:]=(225,225,225);rgb[245:290,215:]=(225,225,225)
  self.assertTrue(edge_clutter(rgb,protected,.8)[270,230])
 def test_spare_box_is_removed_but_protected_seat_is_kept(self):
  rgb=np.full((300,240,3),180,np.uint8);protected=np.zeros((300,240),bool)
  cv2.rectangle(rgb,(190,215),(230,275),(230,230,230),-1);cv2.rectangle(rgb,(190,215),(230,275),(120,120,120),2)
  cv2.rectangle(rgb,(85,200),(155,280),(230,230,230),-1);protected[198:283,83:158]=True
  mask=edge_clutter(rgb,protected,.75)
  self.assertTrue(mask[240,210]);self.assertFalse(mask[240,120])
 def test_removed_clutter_reflection_is_cleaned_without_erasing_clothing_reflection(self):
  rgb=np.full((300,240,3),180,np.uint8);protected=np.zeros((300,240),bool)
  protected[50:200,80:150]=True;rgb[protected]=(220,15,15)
  rgb[260:280,2:12]=(230,190,20);rgb[285:295,95:105]=(180,20,20)
  mask=edge_clutter(rgb,protected,.75)
  self.assertTrue(mask[270,6]);self.assertFalse(mask[290,100])
 def test_blue_cloth_at_frame_edge_is_not_mistaken_for_distant_blue_clothing_reflection(self):
  rgb=np.full((300,240,3),(25,32,50),np.uint8);protected=np.zeros((300,240),bool)
  protected[40:190,150:200]=True;rgb[protected]=(20,30,230)
  rgb[250:285,0:20]=(20,30,230)
  self.assertTrue(edge_clutter(rgb,protected,.75)[265,8])
 def test_floor_tiles_are_not_misclassified_as_spare_seats(self):
  rgb=np.full((300,240,3),180,np.uint8);protected=np.zeros((300,240),bool)
  cv2.rectangle(rgb,(185,260),(230,295),(230,230,230),-1);cv2.rectangle(rgb,(185,260),(230,295),(100,100,100),2)
  self.assertFalse(edge_clutter(rgb,protected,.75)[275,210])
 def test_removed_edge_object_reflection_is_not_saved_by_similar_subject_hue(self):
  rgb=np.full((300,240,3),180,np.uint8);protected=np.zeros((300,240),bool);removed=np.zeros_like(protected)
  protected[50:200,80:150]=True;rgb[protected]=(230,160,30)
  rgb[180:220,0:20]=(230,160,30);removed[180:220,0:20]=True
  rgb[260:280,2:12]=(230,160,30)
  self.assertTrue(edge_clutter(rgb,protected,.75,removed)[270,6])
