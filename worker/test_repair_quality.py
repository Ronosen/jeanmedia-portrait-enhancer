import unittest
from unittest.mock import patch
import numpy as np
from repair_quality import inspect_arrays,harmonize_arrays,inspect_and_retry,harmonize,generic_repair_mask
from backdrop_finish import pack,unpack

class ArrayFrame:
 def __init__(self,a):self.a=a
 def resize(self,*args):return self.a

class RepairQualityTests(unittest.TestCase):
 def fixture(self):
  src=np.full((100,120,3),45,np.uint8);repair=np.zeros((100,120),bool);repair[20:55,20:50]=True
  protected=np.zeros(repair.shape,bool);protected[60:95,65:90]=True
  return src,repair,protected,np.ones(repair.shape,bool)
 def test_good_repair_does_not_claim_client_acceptance(self):
  src,m,p,s=self.fixture();report,suspect=inspect_arrays(src,src.copy(),m,p,s)
  self.assertFalse(report['issues']);self.assertFalse(suspect.any());self.assertTrue(report['visualReviewRequired'])
 def test_drifted_patch_is_flagged_without_selecting_people(self):
  src,m,p,s=self.fixture();bad=src.copy();bad[m]=180
  report,suspect=inspect_arrays(src,bad,m,p,s)
  self.assertTrue(report['issues']);self.assertTrue(suspect.any());self.assertFalse((suspect&p).any())
 def test_low_frequency_harmonization_is_bounded_and_local(self):
  src,m,p,s=self.fixture();bad=src.copy();bad[m]=110
  out=harmonize_arrays(src,bad,m,p,s)
  np.testing.assert_array_equal(out[~m],bad[~m]);np.testing.assert_array_equal(out[p],bad[p])
  self.assertLess(float(out[m].mean()),float(bad[m].mean()))
  self.assertLessEqual(np.max(np.abs(out.astype(float)-bad)),28)
 def test_patterned_donor_material_is_not_flattened(self):
  src,m,p,s=self.fixture();src[::2]=180;out=harmonize_arrays(src,src.copy(),m,p,s)
  np.testing.assert_array_equal(out,src)
 def plan(self,m,p):
  return {'layers':{'shape':list(m.shape),'clutter':pack(m.astype(np.uint8)*255),'wallSurface':pack(np.full(m.shape,255,np.uint8)),'foregroundAlpha':pack(p.astype(np.uint8)*255)}}
 def test_retry_is_bounded_and_only_accepts_an_improvement(self):
  src,m,p,s=self.fixture();plan=self.plan(m,p);bad={'version':1,'issues':[{'kind':'repair-seam'}],'score':1.0,'visualReviewRequired':True}
  good={'version':1,'issues':[],'score':0.0,'visualReviewRequired':True};frame=ArrayFrame(src)
  with patch('repair_quality.inspect_arrays',side_effect=[(bad,m),(good,np.zeros_like(m))]),patch('repair_quality.harmonize',side_effect=lambda source,bg,*args:bg):
   calls=[]
   def repairer(background,mask,protected,settings):calls.append(1);return frame
   inspect_and_retry(frame,frame,plan,repairer)
  self.assertEqual(len(calls),1);self.assertTrue(plan['repairInspection']['retryAccepted']);self.assertEqual(plan['repairInspection']['attempts'],2)
 def test_worse_retry_is_rejected_and_review_remains(self):
  src,m,p,s=self.fixture();plan=self.plan(m,p);frame=ArrayFrame(src)
  first={'issues':[{'kind':'seam'}],'score':1.0};worse={'issues':[{'kind':'seam'}],'score':2.0}
  with patch('repair_quality.inspect_arrays',side_effect=[(first,m),(worse,m)]),patch('repair_quality.harmonize',side_effect=lambda source,bg,*args:bg):
   out=inspect_and_retry(frame,frame,plan,lambda *args:frame)
  self.assertIs(out,frame);self.assertFalse(plan['repairInspection']['retryAccepted']);self.assertTrue(plan['review'])
 def test_rejected_retry_is_not_repeated_during_saved_export(self):
  src,m,p,s=self.fixture();plan=self.plan(m,p);frame=ArrayFrame(src)
  plan['repairInspection']={'retryAccepted':False,'issues':[{'kind':'seam'}]}
  with patch('repair_quality.inspect_arrays',side_effect=AssertionError('No new inspection decision on replay')):
   self.assertIs(inspect_and_retry(frame,frame,plan,lambda *args: self.fail('No new inference on replay')),frame)
 def test_accepted_retry_uses_recorded_mask_without_a_new_attempt(self):
  src,m,p,s=self.fixture();plan=self.plan(m,p);frame=ArrayFrame(src)
  plan['repairInspection']={'retryAccepted':True}
  plan['repairRetry']={'mask':pack(m.astype(np.uint8)*255),'settings':{'inpaintPatch':'cached'}}
  with patch('repair_quality.inspect_arrays',side_effect=AssertionError('No new decision on replay')),patch('repair_quality.harmonize',side_effect=lambda source,bg,*args:bg):
   calls=[]
   def replay(background,mask,protected,settings):
    np.testing.assert_array_equal(mask,m);self.assertEqual(settings['inpaintPatch'],'cached');calls.append(1);return frame
   self.assertIs(inspect_and_retry(frame,frame,plan,replay),frame)
  self.assertEqual(len(calls),1)
 def test_generic_harmonizer_cannot_overwrite_specialist_transition(self):
  src,m,p,s=self.fixture();dark=np.full(src.shape,20,np.uint8)
  # This is the conflicting second-pass correction introduced in r24.
  general=harmonize_arrays(src,dark,m,p,s)
  self.assertTrue(np.any(general[m]!=dark[m]))
  plan=self.plan(m,p);plan['sceneCleanupRevision']=25
  plan['reconstructionTone']={'ownedMask':pack(m.astype(np.uint8)*255)}
  frame=ArrayFrame(dark)
  self.assertIs(harmonize(ArrayFrame(src),frame,plan),frame)
 def test_generic_retry_cannot_redraw_a_specialist_owned_region(self):
  src,m,p,s=self.fixture();plan=self.plan(m,p);plan['sceneCleanupRevision']=25
  plan['reconstructionTone']={'ownedMask':pack(m.astype(np.uint8)*255)}
  report={'issues':[{'kind':'repair-seam'}],'score':1.0}
  frame=ArrayFrame(src)
  with patch('repair_quality.inspect_arrays',return_value=(report,m)):
   inspect_and_retry(frame,frame,plan,lambda *args:self.fail('Specialist region must not be redrawn'))
  self.assertEqual(plan['repairInspection']['attempts'],1);self.assertTrue(plan['review'])
 def test_other_repair_regions_and_old_plans_keep_their_behavior(self):
  _,m,p,_=self.fixture();owned=np.zeros(m.shape,bool);owned[20:35,20:50]=True
  plan=self.plan(m,p);plan['sceneCleanupRevision']=25
  plan['reconstructionTone']={'ownedMask':pack(owned.astype(np.uint8)*255)}
  out=generic_repair_mask(plan,m)
  np.testing.assert_array_equal(out,m&~owned)
  plan['sceneCleanupRevision']=24
  np.testing.assert_array_equal(generic_repair_mask(plan,m),m)
 def test_specialist_records_ownership_even_when_repair_is_already_good(self):
  from reconstruction_tone import match
  src,m,p,_=self.fixture();plan=self.plan(m,p);plan.update(sceneCleanupRevision=25,
    coefficients=[[45]*3]+[[0]*3]*5,low=[45]*3,high=[45]*3)
  plan['layers']['protected']=pack(p.astype(np.uint8)*255)
  plan['layers']['anchoredForegroundAlpha']=plan['layers']['foregroundAlpha']
  frame=ArrayFrame(src)
  self.assertIs(match(frame,frame,plan),frame)
  self.assertEqual(plan['reconstructionTone']['components'],0)
  np.testing.assert_array_equal(unpack(plan['reconstructionTone']['ownedMask'],m.shape)>0,m)

if __name__=='__main__':unittest.main()
