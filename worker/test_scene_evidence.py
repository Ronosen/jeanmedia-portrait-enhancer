import unittest
import cv2
import numpy as np
from PIL import Image
from scene_evidence import floor_evidence
from backdrop_finish import finish, pack, unpack


class SceneEvidenceTests(unittest.TestCase):
    def test_floor_follows_material_at_different_positions(self):
        h,w=400,240
        for row in (132,204,284,356):
            with self.subTest(row=row):
                rgb=np.full((h,w,3),20,np.uint8);rgb[row:]=(30,33,35)
                subject=np.zeros((h,w),bool);subject[40:385,100:140]=True
                rgb[subject]=(160,80,50)
                floor,analysis=floor_evidence(rgb,subject)
                self.assertTrue(analysis['detected'])
                self.assertAlmostEqual(floor*h,row,delta=5)

    def test_one_edge_object_does_not_become_a_full_width_floor(self):
        rgb=np.full((400,240,3),25,np.uint8);rgb[150:,:50]=180
        subject=np.zeros((400,240),bool);subject[40:350,100:140]=True
        self.assertEqual(floor_evidence(rgb,subject)[0],1)

    def test_clean_lighting_is_preserved_and_no_floor_is_invented(self):
        h,w=400,240;yy,xx=np.mgrid[:h,:w]
        light=20+10*xx/w+16*(yy/h)**2
        rgb=np.repeat(np.rint(light)[...,None],3,axis=2).astype(np.uint8)
        subject=np.zeros((h,w),bool);subject[40:350,100:140]=True;rgb[subject]=(140,70,40)
        self.assertEqual(floor_evidence(rgb,subject)[0],1)
        plan=self.plan(subject,1)
        out=finish(Image.fromarray(rgb),plan)
        self.assertEqual(out.tobytes(),rgb.tobytes())
        self.assertEqual(plan['backdropFinish']['action'],'preserve')

    def plan(self,subject,floor):
        return {'adaptiveRevision':3,'floor':floor,'separation':{'version':1,'backdropMethod':'local-evidence-v2'},
                'layers':{'shape':list(subject.shape),'protected':pack(subject.astype(np.uint8)*255)}}

    def test_local_folds_move_with_photo_and_leave_other_regions_untouched(self):
        h,w=480,320;yy,xx=np.mgrid[:h,:w]
        for centre in (120,270,380):
            with self.subTest(centre=centre):
                fold=12*np.sin(xx/7)*np.exp(-((yy-centre)/25)**8)
                rgb=np.repeat(np.rint(30+fold)[...,None],3,axis=2).astype(np.uint8)
                rgb[430:]=(55,60,65)
                subject=np.zeros((h,w),bool);subject[50:460,140:180]=True;rgb[subject]=(140,70,40)
                plan=self.plan(subject,430/h);out=np.asarray(finish(Image.fromarray(rgb),plan))
                region=np.s_[centre-10:centre+10,20:110,0]
                self.assertLess(out[region].std(),rgb[region].std()*.45)
                np.testing.assert_array_equal(out[subject],rgb[subject])
                np.testing.assert_array_equal(out[430:],rgb[430:])
                active=unpack(plan['backdropFinish']['mask'],[h,w])>0
                rows=np.flatnonzero(active.any(axis=1))
                self.assertLess(abs(np.median(rows)-centre),15)
                self.assertLess(len(rows),150)
                self.assertEqual(finish(Image.fromarray(rgb),plan).tobytes(),out.tobytes())


if __name__=='__main__':unittest.main()
