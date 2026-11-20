import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from separation import anchored_alpha,composite,prepare,render,replay,trim_support_edges
from backdrop_finish import pack,unpack

class SeparationTests(unittest.TestCase):
 def test_semantic_clutter_below_uncertain_wall_enters_visible_repair_mask(self):
  rgb=np.full((220,180,3),25,np.uint8);alpha=np.zeros((220,180),np.float32);alpha[40:200,80:130]=1
  raw=alpha.copy();raw[180:210,10:40]=1;rgb[180:210,10:40]=(30,40,200)
  removal=np.zeros(alpha.shape,bool);removal[180:210,10:40]=True
  empty=np.zeros(alpha.shape,np.uint8)
  plan={'sceneIntentSkill':1,'coefficients':[[25]*3]+[[0]*3]*5,'low':[25]*3,'high':[25]*3,'floor':.7,
        'floorAnalysis':{'detected':False},'layers':{'shape':[220,180]}}
  with patch('scene_intent.assess',return_value=(np.zeros_like(alpha),removal,{'version':1,'objects':[],'uncertain':0})):
   selected=prepare(plan,rgb,alpha,alpha,[],raw,alpha)
  self.assertTrue((selected[removal]>0).all())
  self.assertTrue((unpack(plan['layers']['clutter'],[220,180])[removal]>0).all())
  self.assertFalse((unpack(plan['layers']['protected'],[220,180])[removal]>0).any())
  self.assertTrue((unpack(plan['layers']['protected'],[220,180])[alpha>.5]>0).all())

 def test_confirmed_front_panel_recovers_shaded_side_but_not_detached_wall(self):
  from support_geometry import complete_support_surfaces
  rgb=np.full((300,240,3),50,np.uint8);rgb[180:250,60:100]=160;rgb[170:250,100:170]=230
  rgb[:,10:25]=220
  # A thin bright reflection reaches the frame but is not a prop panel.
  rgb[250:,135:136]=200
  seed=np.zeros((300,240),bool);seed[185:245,110:165]=True
  full=complete_support_surfaces(rgb,seed,np.full_like(rgb,30,dtype=float))
  self.assertTrue(full[220,70]);self.assertTrue(full[175,120])
  self.assertFalse(full[:,15].any());self.assertFalse(full[260,120]);self.assertFalse(full[290,135])

 def test_support_contact_shadow_has_a_local_feather_and_no_wall_extension(self):
  from support_geometry import support_source_masks
  rgb=np.full((300,240,3),30,np.uint8);support=np.zeros((300,240),bool)
  support[190:260,80:160]=True;rgb[support]=220;rgb[260:264,80:160]=8
  exact,blend=support_source_masks(rgb,np.full_like(rgb,30,dtype=float),np.zeros((300,240),np.float32),support,np.zeros_like(support),.6)
  self.assertTrue(exact[260,110]);self.assertGreater(blend[262,110],0)
  self.assertEqual(blend[280,110],0);self.assertEqual(blend[175,110],0)
  self.assertEqual(blend[261,40],0)

 def test_saved_floor_fringe_is_preserved_and_opaque_overlap_still_rejected(self):
  source=Image.new('RGB',(40,40),(30,30,30));a=np.zeros((40,40),np.uint8);a[32,20]=4
  hole=np.zeros_like(a);hole[32,20]=255
  plan={'floor':.5,'separation':{'version':1},'layers':{'shape':[40,40],'source':pack(np.asarray(source)),
        'foregroundAlpha':pack(a),'backdrop':pack(np.zeros_like(a))},
        'floorRepairs':{'version':1,'shape':[40,40],'defects':pack(np.zeros_like(a)),'extension':pack(hole)}}
  with patch('separation.render',return_value=source):replay(source,source,plan,True)
  self.assertFalse(unpack(plan['floorRepairs']['extension'],[40,40]).any())
  a[32,20]=255;plan['layers']['foregroundAlpha']=pack(a);plan['floorRepairs']['extension']=pack(hole)
  with self.assertRaisesRegex(ValueError,'overlaps'):replay(source,source,plan,True)

 def test_shaded_top_can_match_the_fitted_background_but_still_belong_to_the_seat(self):
  from support_geometry import attached_props
  rgb=np.full((300,240,3),25,np.uint8);rgb[:,:16]=160
  support=np.zeros((300,240),bool);support[160:280,10:180]=True;rgb[support]=230;rgb[140:160,12:170]=170
  alpha=np.zeros((300,240),np.float32);alpha[110:145,90:120]=1;rgb[alpha>0]=(140,80,60)
  props=attached_props(rgb,alpha,support,np.full_like(rgb,170,dtype=float))
  self.assertTrue(props[150,50]);self.assertFalse(props[120,12])

 def test_inset_prop_search_recovers_a_shaded_top_joined_to_an_edge_wall(self):
  from support_geometry import attached_props
  rgb=np.full((300,240,3),25,np.uint8);rgb[:,:16]=160
  support=np.zeros((300,240),bool);support[160:280,10:180]=True;rgb[support]=230
  rgb[140:160,12:170]=170
  alpha=np.zeros((300,240),np.float32);alpha[110:145,90:120]=1;rgb[alpha>0]=(140,80,60)
  props=attached_props(rgb,alpha,support,np.full_like(rgb,25,dtype=float))
  self.assertTrue(props[150,50]);self.assertFalse(props[120,12])

 def test_frame_cleaning_keeps_a_confirmed_seat_back_above_its_measured_front(self):
  rgb=np.full((360,300,3),25,np.uint8);rgb[:,:40]=160;rgb[200:310,20:110]=230;rgb[180:200,30:110]=230
  alpha=np.zeros((360,300),np.float32);alpha[80:190,100:170]=1;rgb[alpha>0]=70
  rough=np.zeros((360,300),np.uint8);rough[200:310,20:110]=255;rough[180:200,30:110]=255
  measured=np.zeros_like(rough);measured[200:310,20:110]=255
  plan={'coefficients':[[25]*3]+[[0]*3]*5,'low':[25]*3,'high':[25]*3,'floor':1,
        'layers':{'shape':[360,300],'support':pack(rough),'geometrySupport':pack(measured)}}
  prepare(plan,rgb,alpha,alpha,[{'class':0,'confidence':.9,'box':[100,80,170,190]}],alpha,alpha)
  self.assertEqual(unpack(plan['layers']['foregroundAlpha'],[360,300])[190,35],255)

 def test_seat_left_edge_excludes_a_nearby_bright_wall_fragment(self):
  rgb=np.full((240,200,3),25,np.uint8);rgb[:,:60]=130;rgb[110:220,60:170]=230
  support=np.zeros((240,200),bool);support[110:220,40:170]=True
  actual=trim_support_edges(rgb,support)
  self.assertFalse(actual[170,45]);self.assertTrue(actual[170,65])

 def test_hand_seat_contact_does_not_promote_nearby_wall_to_foreground(self):
  from support_geometry import support_contacts
  support=np.zeros((300,240),bool);support[150:260,100:180]=True
  alpha=np.zeros((300,240),np.float32);alpha[130:155,94:130]=1
  contact=support_contacts(alpha,support)
  self.assertFalse(contact[158,95]);self.assertTrue(contact[149,110])

 def test_support_rim_does_not_copy_a_nearby_detached_wall_strip(self):
  from support_geometry import support_source_masks
  rgb=np.full((300,240,3),30,np.uint8);support=np.zeros((300,240),bool)
  support[90:260,105:170]=True;rgb[support]=220;rgb[90:260,99:102]=200
  exact,_=support_source_masks(rgb,np.full_like(rgb,30,dtype=float),np.zeros((300,240),np.float32),support,np.zeros_like(support))
  self.assertFalse(exact[180,100]);self.assertTrue(exact[180,110])

 def test_geometry_panel_does_not_erase_the_rest_of_a_bright_seat(self):
  rgb=np.full((240,200,3),25,np.uint8);rgb[140:220,65:180]=220
  alpha=np.zeros((240,200),np.float32);alpha[50:145,60:95]=1;rgb[alpha>0]=80
  rough=np.zeros((240,200),np.uint8);rough[140:220,65:180]=255
  measured=np.zeros_like(rough);measured[140:220,65:140]=255
  plan={'coefficients':[[25]*3]+[[0]*3]*5,'low':[25]*3,'high':[25]*3,'floor':1,
        'layers':{'shape':[240,200],'support':pack(rough),'geometrySupport':pack(measured)}}
  prepare(plan,rgb,alpha,alpha,[])
  actual=unpack(plan['layers']['foregroundAlpha'],[240,200])
  self.assertEqual(actual[180,170],255);self.assertEqual(actual[180,190],0)

 def test_truncated_background_patch_is_not_promoted_to_contact_prop(self):
  from support_geometry import attached_props
  rgb=np.full((300,240,3),210,np.uint8);support=np.zeros((300,240),bool);support[160:280,80:180]=True;rgb[support]=230
  # Wall shadow crosses the search boundary; it has no measured object top.
  rgb[80:160,100:180]=170
  alpha=np.zeros((300,240),np.float32);alpha[80:165,175:190]=1;rgb[alpha>0]=70
  props=attached_props(rgb,alpha,support,np.full_like(rgb,210,dtype=float))
  self.assertFalse(props[140,130])

 def test_measured_box_replaces_oversized_shadow_support(self):
  rgb=np.full((220,180,3),210,np.uint8);rgb[125:200,75:135]=235;rgb[80:125,75:135]=170
  alpha=np.zeros((220,180),np.float32);alpha[30:140,45:65]=1;rgb[alpha>0]=60
  rough=np.zeros((220,180),np.uint8);rough[80:200,75:135]=255
  measured=np.zeros_like(rough);measured[125:200,75:135]=255
  plan={'coefficients':[[210]*3]+[[0]*3]*5,'low':[210]*3,'high':[210]*3,'floor':1,
        'layers':{'shape':[220,180],'support':pack(rough),'geometrySupport':pack(measured)}}
  prepare(plan,rgb,alpha,alpha,[])
  result=unpack(plan['layers']['foregroundAlpha'],[220,180])
  self.assertEqual(result[100,100],0);self.assertEqual(result[150,100],255)

 def test_shaded_support_rim_is_recovered_without_protecting_external_background(self):
  from support_geometry import support_source_masks
  rgb=np.full((300,240,3),30,np.uint8);support=np.zeros((300,240),bool)
  support[90:260,105:170]=True;rgb[90:260,100:170]=220;rgb[90:100,100:170]=110;rgb[100:260,100:105]=85
  exact,blend=support_source_masks(rgb,np.full_like(rgb,30,dtype=float),np.zeros((300,240),np.float32),support,np.zeros_like(support))
  self.assertTrue(exact[95,102]);self.assertTrue(exact[180,102])
  self.assertFalse(exact[180,95]);self.assertEqual(blend[180,95],0)

 def test_native_prop_boundary_pixels_are_copied_without_edge_rematting(self):
  rng=np.random.default_rng(8);source=Image.fromarray(rng.integers(0,256,(141,123,3),dtype=np.uint8))
  alpha=np.zeros((40,35),np.uint8);alpha[8:33,10:26]=128
  contact=np.zeros_like(alpha);contact[9:32,10:12]=255;contact[9:11,10:26]=255
  plan={'layers':{'shape':[40,35],'foregroundAlpha':pack(alpha),'sourceContact':pack(contact)}}
  with patch('neural.repair',return_value=Image.new('RGB',source.size,(20,25,30))):
   actual=np.asarray(render(source,np.zeros_like(alpha),plan))
  boundary=np.asarray(Image.fromarray(contact).resize(source.size,Image.Resampling.NEAREST))>0
  np.testing.assert_array_equal(actual[boundary],np.asarray(source)[boundary])
  np.testing.assert_array_equal(actual[0,0],[20,25,30])

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
