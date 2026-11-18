import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from separation import anchored_alpha,composite,prepare,render,replay,trim_support_edges
from backdrop_finish import pack,unpack

class SeparationTests(unittest.TestCase):
 def test_books_touching_hand_and_pedestal_keep_their_printed_source_pixels(self):
  from support_geometry import attached_props
  rgb=np.full((240,180,3),30,np.uint8);support=np.zeros((240,180),bool);support[120:220,70:145]=True;rgb[support]=230
  rgb[95:120,75:140]=(190,140,150);rgb[102:105,80:130]=15
  alpha=np.zeros((240,180),np.float32);alpha[80:98,85:110]=1;rgb[alpha>0]=(140,80,55)
  rgb[95:120,5:40]=200
  props=attached_props(rgb,alpha,support,np.full_like(rgb,30,dtype=float))
  self.assertTrue(props[103,100]);self.assertTrue(props[112,135]);self.assertFalse(props[110,20])

 def test_white_seat_stops_at_object_edge_before_cast_shadow(self):
  rgb=np.full((200,160,3),210,np.uint8);rgb[90:180,50:105]=235;rgb[90:180,105:115]=180
  support=np.zeros((200,160),bool);support[90:180,50:115]=True
  fixed=trim_support_edges(rgb,support)
  self.assertTrue(fixed[140,100]);self.assertFalse(fixed[140,110])
 def test_unrelated_salient_object_is_not_part_of_subject(self):
  a=np.zeros((100,100),np.float32);a[20:90,40:70]=1;a[10:60,0:15]=1
  p=np.zeros_like(a);p[30:70,45:65]=1
  result=anchored_alpha(a,p,[])
  self.assertEqual(result[30,5],0);self.assertEqual(result[50,50],1)
 def test_unrelated_vase_detection_does_not_protect_background_clutter(self):
  a=np.zeros((100,100),np.float32);a[20:90,40:70]=1;a[10:60,0:15]=1
  p=np.zeros_like(a);p[30:70,45:65]=1
  result=anchored_alpha(a,p,[{'class':75,'confidence':.9,'box':[0,10,15,60]}])
  self.assertEqual(result[30,5],0);self.assertEqual(result[50,50],1)
 def test_missing_group_member_is_rejected(self):
  a=np.zeros((100,100),np.float32);a[20:90,10:35]=1
  independent=a.copy();independent[20:90,65:90]=1
  people=[{'class':0,'confidence':.9,'box':[60,10,95,95]}]
  with self.assertRaisesRegex(ValueError,'person is missing'):anchored_alpha(a,independent,people)
 def test_softbox_false_person_detection_does_not_reject_real_subject(self):
  a=np.zeros((100,100),np.float32);a[20:90,60:90]=1
  result=anchored_alpha(a,a,[{'class':0,'confidence':.8,'box':[0,10,30,95]}])
  self.assertEqual(result[50,70],1)
 def test_composite_preserves_subject_and_cleans_background(self):
  source=Image.new('RGB',(30,30),(170,80,30));background=Image.new('RGB',(30,30),(20,25,30))
  alpha=np.zeros((30,30),np.uint8);alpha[8:22,8:22]=255;alpha[7,8:22]=128
  result=np.asarray(composite(source,background,alpha))
  np.testing.assert_array_equal(result[10:20,10:20],np.asarray(source)[10:20,10:20])
  np.testing.assert_array_equal(result[0,0],[20,25,30])
  self.assertGreater(int(result[7,10,0]),20);self.assertLess(int(result[7,10,0]),170)
 def test_old_background_colour_is_removed_from_soft_edge(self):
  pixels=np.full((30,30,3),100,np.uint8);pixels[10:20,10:20]=200;pixels[9,10:20]=150
  a=np.zeros((30,30),np.uint8);a[10:20,10:20]=255;a[9,10:20]=128
  out=np.asarray(composite(Image.fromarray(pixels),Image.new('RGB',(30,30),(20,20,20)),a))
  self.assertLessEqual(abs(int(out[9,15,0])-110),1)
  np.testing.assert_array_equal(out[12,12],[200,200,200])
 def test_confirmed_pedestal_is_included_in_internal_subject_layer(self):
  rgb=np.full((100,100,3),100,np.uint8);a=np.zeros((100,100),np.float32);a[15:65,30:65]=1
  support=np.zeros((100,100),np.uint8);support[60:90,50:80]=255
  plan={'coefficients':[[100,100,100]]+[[0,0,0]]*5,'low':[100]*3,'high':[100]*3,'floor':.8,
        'layers':{'shape':[100,100],'support':pack(support)}}
  prepare(plan,rgb,a,a,[])
  self.assertEqual(unpack(plan['layers']['foregroundAlpha'],[100,100])[80,70],255)
  self.assertEqual(unpack(plan['layers']['foregroundAlpha'],[100,100])[95,95],0)
 def test_support_shadow_rectangle_is_not_composited_onto_white_backdrop(self):
  rgb=np.full((180,140,3),180,np.uint8);rgb[90:160,50:90]=230
  a=np.zeros((180,140),np.float32);a[30:95,45:65]=1
  support=np.zeros((180,140),np.uint8);support[90:160,50:100]=255
  plan={'coefficients':[[180]*3]+[[0]*3]*5,'low':[180]*3,'high':[180]*3,'floor':.9,
        'layers':{'shape':[180,140],'support':pack(support)}}
  prepare(plan,rgb,a,a,[]);alpha=unpack(plan['layers']['foregroundAlpha'],[180,140])
  self.assertEqual(alpha[95,96],0);self.assertEqual(alpha[140,70],255)
 def test_subject_is_removed_from_plate_and_original_is_composited_back(self):
  a=np.zeros((40,40),np.uint8);a[10:30,10:30]=255
  plan={'layers':{'shape':[40,40],'foregroundAlpha':pack(a)}}
  source=Image.new('RGB',(40,40),(180,80,40));background=Image.new('RGB',(40,40),(15,15,15))
  with patch('neural.repair',return_value=background) as repair:
   result=np.asarray(render(source,np.zeros((40,40),np.uint8),plan))
  self.assertTrue(repair.call_args.args[1][20,20]);self.assertFalse(repair.call_args.args[2].any())
  np.testing.assert_array_equal(result[20,20],[180,80,40]);np.testing.assert_array_equal(result[0,0],[15]*3)
 def test_equipment_tip_below_floor_horizon_is_not_left_in_photo(self):
  rgb=np.full((100,100,3),180,np.uint8);a=np.zeros((100,100),np.float32);a[20:80,50:70]=1
  plan={'coefficients':[[180]*3]+[[0]*3]*5,'low':[180]*3,'high':[180]*3,'floor':.8,
        'equipmentPlan':{'repairs':[{'polygon':[[0,.4],[.15,.4],[.1,1],[0,1]]}]},
        'layers':{'shape':[100,100]}}
  prepare(plan,rgb,a,a,[])
  self.assertEqual(unpack(plan['layers']['backdrop'],[100,100])[95,5],255)
  self.assertEqual(unpack(plan['layers']['foregroundAlpha'],[100,100])[40,60],255)
 def test_background_repair_selection_does_not_disqualify_clean_floor_donors(self):
  from floor import encode_mask
  rgb=np.full((100,100,3),90,np.uint8);rgb[90:]=220
  empty=np.zeros((100,100),np.uint8);extension=empty.copy();extension[90:]=255
  selected=empty.copy();selected[70:]=255
  plan={'floor':.7,'separation':{'version':1,'floorMethod':'source-texture-v1'},
        'layers':{'shape':[100,100],'foregroundAlpha':pack(empty),'clutter':pack(empty)},
        'floorRepairs':{'version':1,'shape':[100,100],'defects':encode_mask(empty),'extension':encode_mask(extension),'engine':'context-v1'}}
  with patch('neural.repair',return_value=Image.new('RGB',(100,100),'black')):
   out=np.asarray(render(Image.fromarray(rgb),selected,plan))
  np.testing.assert_array_equal(out[95,30],[90,90,90])
 def test_saved_matte_replay_checks_source_without_rerunning_models(self):
  source=Image.new('RGB',(40,40),(180,80,40));a=np.zeros((40,40),np.uint8);a[10:30,10:30]=255
  plan={'separation':{'version':1},'layers':{'shape':[40,40],'source':pack(np.asarray(source)),
        'foregroundAlpha':pack(a),'backdrop':pack(np.zeros_like(a))}}
  with patch('separation.render',return_value=source) as renderer:
   full,out,_=replay(source,source,plan,True)
  self.assertEqual(renderer.call_count,2);self.assertEqual(full.tobytes(),source.tobytes())
  with self.assertRaisesRegex(ValueError,'different source'):replay(source,Image.new('RGB',(40,40),'white'),plan,True)
