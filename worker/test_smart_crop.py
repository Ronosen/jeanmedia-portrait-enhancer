import copy,unittest
import numpy as np
from backdrop_finish import pack,unpack
from smart_crop import propose,prepare,validated_box,retained_mask,output_box,output_size,protection,continuous_surface


class SmartCropTests(unittest.TestCase):
 def fixture(self):
  protected=np.zeros((200,240),bool);protected[60:170,85:160]=True
  defects=np.zeros_like(protected);defects[:,:30]=True
  return protected,defects
 def plan(self,p,m):
  zero=np.zeros(p.shape,np.uint8)
  return {'layers':{'shape':list(p.shape),'foregroundAlpha':pack(p.astype(np.uint8)*255),
   'clutter':pack(m.astype(np.uint8)*255),'backdrop':pack(m.astype(np.uint8)*255),
   'wallSurface':pack(np.full(p.shape,255,np.uint8))},'floorRepairs':{'defects':pack(zero),'extension':pack(zero),'count':0},'review':[]}
 def test_only_a_safe_unwanted_outer_strip_is_trimmed(self):
  p,m=self.fixture();crop=propose(p,m)
  self.assertTrue(crop['enabled']);self.assertEqual(crop['box'],[30,0,240,200])
 def test_subject_at_edge_vetoes_that_crop(self):
  p,m=self.fixture();p[90:120,:20]=True
  self.assertFalse(propose(p,m)['enabled'])
 def test_detached_prop_at_edge_is_retained(self):
  p,m=self.fixture();p[30:45,:18]=True
  self.assertFalse(propose(p,m)['enabled'])
 def test_clear_background_is_not_cropped_for_composition(self):
  p,m=self.fixture();m[:]=False
  self.assertFalse(propose(p,m)['enabled'])
 def test_interior_defects_use_normal_repair(self):
  p,m=self.fixture();m[:]=False;m[50:80,40:60]=True
  self.assertFalse(propose(p,m)['enabled'])
 def test_a_tiny_corner_mark_does_not_justify_losing_canvas(self):
  p,m=self.fixture();m[:]=False;m[:3,:3]=True
  self.assertFalse(propose(p,m)['enabled'])
 def test_crop_cannot_be_proposed_without_reliable_protection(self):
  p,m=self.fixture();p[:]=False
  self.assertFalse(propose(p,m)['enabled'])
 def test_unknown_prop_is_a_crop_veto(self):
  p,m=self.fixture();plan=self.plan(p,m)
  plan['sceneIntent']={'objects':[{'action':'uncertain','label':'other','box':[0,20,40,45]}]}
  self.assertTrue(protection(plan)[20,0]);prepare(plan,(4000,3333))
  self.assertFalse(plan['smartCrop']['enabled'])
 def test_crop_removes_discarded_area_from_reconstruction_masks(self):
  p,m=self.fixture();plan=self.plan(p,m);prepare(plan,(4000,3333))
  self.assertTrue(plan['smartCrop']['enabled']);self.assertFalse((unpack(plan['layers']['clutter'],p.shape)>0).any())
  self.assertFalse((unpack(plan['layers']['backdrop'],p.shape)>0).any())
  np.testing.assert_array_equal(unpack(plan['layers']['foregroundAlpha'],p.shape)>0,p)
 def test_native_coordinates_and_dimensions_are_scaled_consistently(self):
  p,m=self.fixture();plan=self.plan(p,m);prepare(plan,(4000,3333))
  self.assertEqual(output_box((4000,3333),plan),(500,0,4000,3333))
  self.assertEqual(output_size((4000,3333),plan),(3500,3333))
  self.assertEqual(output_box((240,200),plan),(30,0,240,200))
 def test_saved_crop_replays_the_same_box(self):
  p,m=self.fixture();plan=self.plan(p,m);prepare(plan,(4000,3333))
  self.assertEqual(validated_box(copy.deepcopy(plan)),validated_box(plan))
 def test_tampering_cannot_move_crop_into_protection(self):
  p,m=self.fixture();p[80:100,25:50]=True;plan=self.plan(p,m)
  plan['layers']['cropProtection']=pack(p.astype(np.uint8)*255)
  plan['smartCrop']={'version':1,'enabled':True,'previewSize':[240,200],'box':[25,0,240,200]}
  with self.assertRaises(ValueError):validated_box(plan)
 def test_incomplete_crop_protection_is_rejected(self):
  p,m=self.fixture();plan=self.plan(p,m);prepare(plan,(4000,3333))
  plan['layers']['cropProtection']=pack(np.zeros(p.shape,np.uint8))
  with self.assertRaises(ValueError):validated_box(plan)
 def test_combined_trims_retain_at_least_three_quarters(self):
  p,m=self.fixture();m[:,:42]=True;m[:,-42:]=True;m[:30]=True
  crop=propose(p,m);l,t,r,b=crop['box']
  self.assertGreaterEqual((r-l)*(b-t),p.size*.75)
  self.assertFalse(p[:t].any() or p[b:].any() or p[:,:l].any() or p[:,r:].any())
 def test_bottom_floor_gap_can_be_cropped_but_shoes_prevent_it(self):
  p,m=self.fixture();p[:]=False;p[60:150,85:160]=True;m[:]=False;m[-20:]=True
  self.assertEqual(propose(p,m)['box'],[0,0,240,180])
  p[170:200,85:125]=True
  self.assertFalse(propose(p,m)['enabled'])
 def test_discarded_canvas_does_not_become_a_blend_boundary(self):
  p,m=self.fixture();plan=self.plan(p,m);prepare(plan,(4000,3333))
  kept=retained_mask(plan);surface=continuous_surface(np.zeros(p.shape,bool),plan)
  np.testing.assert_array_equal(surface,~kept)
 def test_old_plans_and_imports_keep_original_dimensions(self):
  self.assertEqual(output_size((4000,6000),None),(4000,6000))
  self.assertEqual(output_box((4000,6000),{}),(0,0,4000,6000))
 def test_feet_and_supports_reserve_ground_for_contact_shadows(self):
  p,m=self.fixture();m[:]=False;m[-20:]=True
  plan=self.plan(p,m);plan['floor']=.7;plan['floorAnalysis']={'detected':True}
  guard=protection(plan)
  self.assertGreater(np.nonzero(guard)[0].max(),np.nonzero(p)[0].max())
  decision=propose(guard,m)
  self.assertGreaterEqual(decision['box'][3],185)
 def test_discarded_floor_extension_is_not_repaired_or_counted(self):
  p,m=self.fixture();plan=self.plan(p,m)
  plan['floorRepairs']['extension']=pack(m.astype(np.uint8)*255);plan['floorRepairs']['count']=1
  prepare(plan,(4000,3333))
  self.assertEqual(plan['floorRepairs']['count'],0)
  self.assertFalse((unpack(plan['floorRepairs']['extension'],p.shape)>0).any())
 def test_inspection_donors_cannot_include_the_discarded_canvas(self):
  from repair_quality import masks
  p,m=self.fixture();plan=self.plan(p,m);prepare(plan,(4000,3333))
  keep=retained_mask(plan);repair,protected,wall,surface=masks(plan)
  self.assertFalse(repair[~keep].any());self.assertFalse(wall[~keep].any());self.assertFalse(surface[~keep].any())
  np.testing.assert_array_equal(protected,p)

if __name__=='__main__':unittest.main()
