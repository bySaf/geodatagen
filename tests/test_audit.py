import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from src.validation import check_signal


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.path=Path(self.tmp.name)/'bscan.h5'
        self.tx=np.array([[1.,1.,.96],[1.1,1.,.96]])
        self.rx=self.tx+[.04,0,0]
        with h5py.File(self.path,'w') as f:
            f.attrs.update(config_sha256='test',dt=1e-10,units='V/m',axis_order='scan,time')
            f['tx_m']=self.tx;f['rx_m']=self.rx;f['complete']=[True,True]
            f['time_s']=np.arange(5)*1e-10;f['Ey']=np.ones((2,5));f['seconds']=[.1,.2]

    def tearDown(self): self.tmp.cleanup()

    def test_completion_mask_is_required(self):
        self.assertTrue(check_signal(self.path,self.tx,self.rx,'test')['valid'])
        with h5py.File(self.path,'r+') as f: f['complete'][1]=False
        result=check_signal(self.path,self.tx,self.rx,'test')
        self.assertFalse(result['valid']);self.assertEqual(result['completed'],1)

    def test_reject_corrupt_signal_coordinates_and_hash(self):
        with self.assertRaises(ValueError): check_signal(self.path,self.tx,self.rx,'changed')
        with self.assertRaises(ValueError): check_signal(self.path,self.tx+.1,self.rx,'test')
        with h5py.File(self.path,'r+') as f: f['Ey'][1,2]=np.nan
        with self.assertRaises(ValueError): check_signal(self.path,self.tx,self.rx,'test')
