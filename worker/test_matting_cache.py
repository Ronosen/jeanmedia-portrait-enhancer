import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import matting

class MattingCacheTests(unittest.TestCase):
 def test_exact_pixels_reuse_inference_and_corrupt_cache_is_recomputed(self):
  with tempfile.TemporaryDirectory() as folder:
   rgb=np.full((24,18,3),30,np.uint8);alpha=np.full((24,18),.75,np.float32)
   digest='5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333'
   with patch.object(matting,'__file__',str(Path(folder)/'matting.py')),patch('runtime.sha256',return_value=digest),patch.object(matting,'_infer_alpha',return_value=alpha) as infer:
    np.testing.assert_array_equal(matting.foreground_alpha(rgb),alpha)
    np.testing.assert_array_equal(matting.foreground_alpha(rgb),alpha)
    self.assertEqual(infer.call_count,1)
    rgb[0,0]=31;matting.foreground_alpha(rgb);self.assertEqual(infer.call_count,2)
    for target in (Path(folder)/'.cache'/'matting').glob('*.npy'):target.write_bytes(b'broken')
    matting.foreground_alpha(rgb);self.assertEqual(infer.call_count,3)
