import unittest
import cv2
import numpy as np
from PIL import Image
from scene_evidence import floor_evidence, wall_surface
from backdrop_finish import finish, pack, unpack


class SceneEvidenceTests(unittest.TestCase):
    def test_weaker_material_witness_protects_surface_without_moving_floor(self):
        rgb=np.full((300,240,3),25,np.uint8)
        for a,b in [(0,30),(90,120),(210,240)]:rgb[185:210,a:b]=(38,40,48)
        rgb[210:]=(65,70,85)
        floor,analysis=floor_evidence(rgb,np.zeros((300,240),bool),True)
        self.assertTrue(analysis['detected']);self.assertAlmostEqual(floor,210/300,delta=.04)
        self.assertLess(analysis['surfaceSafetyRow'],analysis['row'])
    def test_deep_black_folds_are_smoothed_without_touching_foreground(self):
        from scene_evidence import local_finish
        h,w=240,180;yy,xx=np.mgrid[:h,:w]
        gray=(8+2*np.sin(xx/5)).astype(np.float32)
        safe=np.ones((h,w),bool);safe[40:210,70:100]=False
        delta,_,_=local_finish(gray,safe,bounded=True)
        self.assertLess(np.std((gray+delta)[safe]),np.std(gray[safe])*.65)
        self.assertTrue((delta[~safe]==0).all())
        light=(8+3*xx/w+2*(yy/h)**2).astype(np.float32)
        correction,_,_=local_finish(light,safe,bounded=True)
        self.assertLess(np.max(np.abs(correction)),.8)

    def test_unknown_dark_floor_is_not_exposed_to_wall_smoothing(self):
        h,w=240,180;yy,xx=np.mgrid[:h,:w]
        rgb=np.full((h,w,3),8,np.uint8);rgb[180:]=(14,16,18)
        rgb[210:212,::8]=40
        subject=np.zeros((h,w),bool);subject[40:220,70:110]=True
        wall,analysis=wall_surface(rgb,subject,1.,{'detected':False})
        self.assertFalse(wall.any());self.assertTrue(analysis['preserveUncertainSurface'])
        plan=self.plan(subject,1.);plan['separation']['backdropMethod']='local-evidence-v5'
        plan['layers']['wallSurface']=pack(wall.astype(np.uint8)*255)
        np.testing.assert_array_equal(np.asarray(finish(Image.fromarray(rgb),plan)),rgb)

    def test_opposite_local_edges_are_not_one_material_transition(self):
        rgb=np.full((400,240,3),20,np.uint8);rgb[:100,120:]=40;rgb[100:300,:120]=40;rgb[300:]=100
        protected=np.zeros((400,240),bool);protected[40:380,100:140]=True
        floor,analysis=floor_evidence(rgb,protected,require_coherence=True)
        self.assertTrue(analysis['detected']);self.assertAlmostEqual(floor,.75,delta=.025)

    def test_sloping_uneven_join_is_preserved_instead_of_a_horizontal_cutoff(self):
        h,w=400,280;yy,xx=np.mgrid[:h,:w]
        for direction in (-1,1):
            with self.subTest(direction=direction):
                edge=np.rint(285+direction*(np.arange(w)-w/2)*.23+6*np.sin(np.arange(w)/35)).astype(int)
                floor=yy>=edge[None,:]
                rgb=np.full((h,w,3),30,np.uint8);rgb[floor]=(65,75,85)
                subject=np.zeros((h,w),bool);subject[40:380,120:160]=True;rgb[subject]=(140,70,40)
                wall,contour=wall_surface(rgb,subject,.71,{'detected':True})
                self.assertTrue(contour['detected'])
                self.assertTrue(wall[30].all())
                # No uncleaned vertical strips above an occluded join, and no
                # stair-step cutoff where neighbouring detector strips meet.
                last_wall=wall.sum(axis=0)
                self.assertLessEqual(np.abs(np.diff(last_wall)).max(),1)
                for x0,x1,row in contour['segments']:
                    self.assertLess(abs(row-np.median(edge[x0:x1])),4)
                self.assertFalse(wall[floor].any())
                # A horizontal .71 cutoff would include healthy floor at one
                # side and leave a dark wall strip at the other.
                plan=self.plan(subject,.71);plan['separation']['backdropMethod']='local-evidence-v3'
                plan['layers']['wallSurface']=pack(wall.astype(np.uint8)*255)
                out=np.asarray(finish(Image.fromarray(rgb),plan))
                join=np.abs(yy-edge[None,:])<=4
                np.testing.assert_array_equal(out[join],rgb[join])
                np.testing.assert_array_equal(out[floor],rgb[floor])
                np.testing.assert_array_equal(out[subject],rgb[subject])

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

    def test_new_smoothing_follows_wall_mask_and_preserves_contacting_prop(self):
        h,w=400,300;yy,xx=np.mgrid[:h,:w]
        light=35+8*xx/w+15*np.sin(xx/12)*np.exp(-((yy-180)/100)**4)
        rgb=np.repeat(np.rint(light)[...,None],3,axis=2).astype(np.uint8)
        protected=np.zeros((h,w),bool);protected[70:350,130:170]=True
        protected[300:340,90:140]=True;rgb[protected]=190
        rgb[350:]=65
        plan=self.plan(protected,.875);plan['separation']['backdropMethod']='local-evidence-v5'
        plan['layers']['wallSurface']=pack((yy<345).astype(np.uint8)*255)
        out=np.asarray(finish(Image.fromarray(rgb),plan))
        self.assertLess(out[130:220,20:100,0].std(),rgb[130:220,20:100,0].std()*.5)
        np.testing.assert_array_equal(out[protected],rgb[protected])
        np.testing.assert_array_equal(out[345:],rgb[345:])

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
