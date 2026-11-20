import unittest
from unittest.mock import patch
import numpy as np
import cv2
from scene_intent import assess,proposals

class SceneIntentTests(unittest.TestCase):
 def fixture(self):
  rgb=np.full((200,240,3),25,np.uint8);raw=np.zeros((200,240),np.float32);raw[20:100,15:65]=1
  rgb[20:100,15:65]=(200,155,30);person=np.zeros(raw.shape,bool);person[60:190,140:210]=1
  return rgb,raw,np.zeros(raw.shape,np.uint8),np.zeros(raw.shape,bool),person
 def test_protected_object_uses_measured_mask_not_padded_box(self):
  args=self.fixture()
  with patch('scene_intent.classify',return_value={'label':'balloon','action':'protect','similarity':.32,'margin':.07}):keep,remove,report=assess(*args)
  self.assertTrue(keep[30:90,20:60].all());self.assertEqual(keep[10,10],0);self.assertFalse(remove.any());self.assertGreater(len(report['objects']),0)
 def test_confident_clutter_is_selected_without_touching_people(self):
  args=self.fixture()
  with patch('scene_intent.classify',return_value={'label':'loose bag','action':'remove','similarity':.32,'margin':.08}):keep,remove,_=assess(*args)
  self.assertTrue(remove[40,30]);self.assertFalse((remove&args[-1]).any());self.assertFalse(keep.any())
 def test_ambiguous_classification_never_authorizes_removal(self):
  with patch('scene_intent.classify',return_value={'label':'other','action':'uncertain','similarity':.24,'margin':.005}):keep,remove,report=assess(*self.fixture())
  self.assertFalse(remove.any());self.assertGreater(report['uncertain'],0)
 def test_object_touching_person_is_not_a_removal_candidate(self):
  rgb,raw,d,e,person=self.fixture();person[20:100,15:65]=True
  with patch('scene_intent.classify') as classify:keep,remove,_=assess(rgb,raw,d,e,person)
  classify.assert_not_called();self.assertFalse(remove.any())
 def test_colour_stripes_do_not_create_whole_frame_proposals(self):
  rgb,raw,d,e,person=self.fixture();raw[:]=0;rgb[:]=(20,20,20);rgb[:,10:12]=(230,180,20)
  self.assertEqual(proposals(rgb,raw,d,e,person),[])
 def test_removal_is_withheld_when_context_disagrees(self):
  first={'label':'loose bag','action':'remove','similarity':.29,'margin':.04}
  second={'label':'other','action':'uncertain','similarity':.26,'margin':.02}
  with patch('scene_intent.classify',side_effect=lambda crop: first if crop.shape[0]<100 else second):
   keep,remove,_=assess(*self.fixture())
  self.assertFalse(remove.any())
 def test_two_balloon_views_can_protect_compact_neutral_material(self):
  rgb=np.full((180,220,3),160,np.uint8);cv2.ellipse(rgb,(65,40),(35,30),0,0,360,(35,35,35),-1)
  empty=np.zeros((180,220),np.uint8)
  result={'label':'balloon','action':'uncertain','similarity':.28,'margin':.018}
  with patch('scene_intent.classify',return_value=result):keep,remove,_=assess(rgb,empty.astype(float),empty,empty,empty.astype(bool))
  self.assertEqual(keep[40,65],1);self.assertEqual(keep[100,65],0);self.assertFalse(remove.any())
 def test_plausible_uncertain_prop_is_preserved_for_review(self):
  result={'label':'graduation gown','action':'uncertain','similarity':.27,'margin':.012}
  with patch('scene_intent.classify',return_value=result):keep,remove,report=assess(*self.fixture())
  self.assertTrue(keep[40,30]);self.assertFalse(remove.any())
  self.assertTrue(report['objects'][0]['preservedForReview']);self.assertGreater(report['uncertain'],0)

if __name__=='__main__':unittest.main()
