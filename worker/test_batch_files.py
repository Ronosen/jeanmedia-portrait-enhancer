import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from batch_files import dispatch

class BatchFilesTests(unittest.TestCase):
 def test_lists_only_top_level_supported_photos_in_natural_order(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for name in ['IMG_10.JPG','IMG_2.jpg','résumé 1.png','notes.txt']:(root/name).write_bytes(b'original')
   (root/'edited').mkdir();(root/'edited'/'old.jpg').write_bytes(b'old')
   result=dispatch('batch-list',{'folder':tmp})
   self.assertEqual([Path(p).name for p in result['sources']],['IMG_2.jpg','IMG_10.JPG','résumé 1.png'])
 def test_outputs_are_beside_sources_and_existing_files_are_preserved(self):
  with tempfile.TemporaryDirectory() as tmp:
   source=Path(tmp)/'portrait.JPG';source.write_bytes(b'original')
   first=Path(dispatch('batch-output',{'source':str(source)})['output']);first.write_bytes(b'edited')
   second=Path(dispatch('batch-output',{'source':str(source)})['output'])
   self.assertEqual(first.parent,source.parent/'edited');self.assertEqual(second.name,'portrait-2.jpg')
   self.assertEqual(source.read_bytes(),b'original');self.assertEqual(first.read_bytes(),b'edited')
 def test_report_is_atomic_and_a_failed_update_preserves_previous_report(self):
  with tempfile.TemporaryDirectory() as tmp:
   data={'folder':tmp,'runId':'12345678-1234-1234-1234-123456789abc','items':[{'source':'one.jpg','status':'saved'}]}
   target=Path(dispatch('batch-report',data)['report']);before=target.read_bytes()
   with patch('batch_files.os.replace',side_effect=OSError('disk full')):
    with self.assertRaises(OSError):dispatch('batch-report',{**data,'items':[]})
   self.assertEqual(target.read_bytes(),before);self.assertFalse(list(target.parent.glob('*.tmp')))
 def test_report_path_traversal_and_non_photo_sources_are_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   source=Path(tmp)/'notes.txt';source.write_text('keep')
   with self.assertRaises(ValueError):dispatch('batch-output',{'source':str(source)})
   with self.assertRaises(ValueError):dispatch('batch-report',{'folder':tmp,'runId':'../escape','items':[]})
 def test_linked_edited_folder_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'source').mkdir();(root/'other').mkdir();source=root/'source'/'one.jpg';source.write_bytes(b'original')
   try:(source.parent/'edited').symlink_to(root/'other',target_is_directory=True)
   except OSError:self.skipTest('symlinks unavailable')
   with self.assertRaises(ValueError):dispatch('batch-output',{'source':str(source)})
