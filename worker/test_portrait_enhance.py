import copy
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image
from face_areas import MOUTH,EYES,feature_masks,polygon
from portrait_enhance import prepare,validate,apply,color_tile,whiten


class PortraitEnhancementTests(unittest.TestCase):
    tone={'contrast':1.12,'gamma':1.,'vibrance':.14}

    def test_neutral_blacks_deepen_without_losing_shadow_steps(self):
        ramp=np.tile(np.arange(15,95,dtype=np.uint8)[None,:,None],(10,1,3))
        out=color_tile(ramp,self.tone,np.zeros(ramp.shape[:2]))
        self.assertLess(out[0,35,0],ramp[0,35,0])
        self.assertGreater(len(np.unique(out[:,:,0])),55)
        self.assertTrue(np.all(np.diff(out[0,:,0].astype(int))>=0))
        self.assertLessEqual(np.ptp(out,axis=2).max(),1)

    def test_faded_colour_gains_chroma_without_turning_neutrals_warm(self):
        rgb=np.array([[[150,95,100],[80,125,165],[160,160,160]]],np.uint8)
        out=color_tile(rgb,self.tone,np.zeros((1,3)))
        before=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(float)
        after=cv2.cvtColor(out,cv2.COLOR_RGB2LAB).astype(float)
        for i in (0,1):
            self.assertGreater(np.linalg.norm(after[0,i,1:]-128),np.linalg.norm(before[0,i,1:]-128))
            old=np.arctan2(*(before[0,i,1:]-128)[::-1]);new=np.arctan2(*(after[0,i,1:]-128)[::-1])
            self.assertLess(abs(old-new),.12)
        self.assertLessEqual(np.ptp(out[0,2]),1)

    def test_face_guard_limits_skin_changes(self):
        rgb=np.full((12,12,3),(135,85,65),np.uint8)
        out=color_tile(rgb,self.tone,np.ones((12,12)))
        self.assertLessEqual(np.abs(out.astype(int)-rgb).max(),4)

    def test_face_guard_cannot_leave_an_oval_on_neutral_shadows(self):
        rgb=np.full((60,80,3),(45,45,45),np.uint8)
        guard=np.zeros((60,80));guard[:,40:]=1
        out=color_tile(rgb,self.tone,guard)
        np.testing.assert_array_equal(out[:,:40],out[:,40:])

    def test_whitening_changes_only_selected_pixels_and_preserves_gaps(self):
        rgb=np.full((20,30,3),(120,70,50),np.uint8)
        rgb[5:10,5:15]=(210,195,145);rgb[6:8,8:10]=(20,20,20)
        teeth=np.zeros((20,30));teeth[5:10,5:15]=1;teeth[6:8,8:10]=0
        out=whiten(rgb,teeth,np.zeros_like(teeth))
        np.testing.assert_array_equal(out[teeth==0],rgb[teeth==0])
        self.assertGreater(out[5,5,2],rgb[5,5,2])
        self.assertLess(np.ptp(out[5,5]),np.ptp(rgb[5,5]))

    def test_semantic_eye_mask_excludes_irises_skin_and_lips(self):
        rgb=np.full((110,180,3),(125,80,60),np.uint8);p=np.zeros((478,2))
        for indices,center in EYES:
            cx=45 if center==468 else 130
            theta=np.linspace(np.pi,3*np.pi,len(indices),endpoint=False)
            p[indices]=np.column_stack([cx+24*np.cos(theta),30+7*np.sin(theta)])
            p[center]=[cx,30]
            p[center+1:center+5]=np.array([[cx+8,30],[cx,38],[cx-8,30],[cx,22]])
            region=polygon(rgb.shape[:2],p,indices);rgb[region]=(185,180,155)
            cv2.circle(rgb,(cx,30),9,(20,25,30),-1)
        theta=np.linspace(np.pi,3*np.pi,len(MOUTH),endpoint=False)
        p[MOUTH]=np.column_stack([90+35*np.cos(theta),80+11*np.sin(theta)])
        mouth=polygon(rgb.shape[:2],p,MOUTH);rgb[mouth]=(210,195,145)
        rgb[84:90,80:100]=(180,65,85)
        teeth,eyes,_=feature_masks(rgb,p)
        self.assertGreater(eyes.sum(),10);self.assertGreater(teeth.sum(),10)
        self.assertEqual(eyes[30,45],0);self.assertEqual(eyes[30,130],0)
        self.assertEqual(teeth[86,90],0)
        allowed=mouth.copy()
        for indices,_ in EYES:allowed|=polygon(rgb.shape[:2],p,indices)
        self.assertFalse(((teeth+eyes>0)&~allowed).any())
        p[14]=p[13]
        closed,_,_=feature_masks(rgb,p)
        self.assertEqual(closed.sum(),0)

    @patch('face_areas.detect',return_value=[])
    def test_saved_finish_replays_preserves_alpha_and_rejects_other_source(self,_):
        image=Image.new('RGBA',(80,100),(65,70,75,128));plan=prepare(image,image)
        validate(plan,image);one=apply(image,plan);two=apply(image,plan)
        self.assertEqual(one.tobytes(),two.tobytes())
        self.assertEqual(one.getchannel('A').tobytes(),image.getchannel('A').tobytes())
        with self.assertRaisesRegex(ValueError,'different photo'):validate(plan,Image.new('RGBA',image.size,(66,70,75,128)))
        bad=copy.deepcopy(plan);bad['tone']['vibrance']=float('nan')
        with self.assertRaises(ValueError):validate(bad,image)


if __name__=='__main__':unittest.main()
