"""Regression for historical hash aliases and irreversible eval assignments."""
import tempfile
from pathlib import Path
import shutil
import unittest
import dataset_store


class StoreTests(unittest.TestCase):
    def test_reencoding_and_hash_alias_cannot_escape_eval(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'source'; source.write_bytes(b'fixture')
            def copy(a,b):
                b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
            def row(key,digest):
                return {'id':key,'origin':key,'sha256':digest,'revision':'rev-'+digest,
                        'image':key+'.png','mask':key+'.png','_image':str(source),'_mask':str(source)}
            initial=dataset_store.merge(root/'data',[row(str(i),'hash-'+str(i)) for i in range(10)],copy)
            ev=next(r for r in initial['entries'].values() if r['group']=='eval')
            previous_hash=ev['sha256']
            changed=row(ev['origin'],'reencoded')
            reencoded=dataset_store.merge(root/'data',[changed],copy)
            self.assertEqual(reencoded['entries'][ev['origin']]['group'],'eval')
            aliased=dataset_store.merge(root/'data',[row('new-name',previous_hash)],copy)
            self.assertEqual(aliased['entries']['new-name']['group'],'eval')
            before={k:r['group'] for k,r in aliased['entries'].items()}
            extended=dataset_store.merge(root/'data',[row('more-'+str(i),'new-hash-'+str(i)) for i in range(10)],copy)
            self.assertTrue(all(extended['entries'][k]['group']==group for k,group in before.items()))

if __name__=='__main__':unittest.main()
