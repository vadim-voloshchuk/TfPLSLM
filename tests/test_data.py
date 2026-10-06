import numpy as np
from tfplslm.data import Writer, Documents, SequentialBatcher


def test_document_boundaries_padding_and_resume(tmp_path):
    w=Writer(tmp_path,'test')
    w.add([1,5,6,2]); w.add([1,8,9,10,11,12,13,2]); w.close()
    batcher=SequentialBatcher([Documents(tmp_path,'test')],4,4,seed=12)
    x,y,reset,_=batcher.next()
    assert reset.all()
    assert (x[:,0]==1).all()
    for row in range(4):
        if x[row,1]==5:
            assert list(y[row])==[5,6,2,-100]
    saved=batcher.state_dict()
    clone=SequentialBatcher([Documents(tmp_path,'test')],4,4,seed=99)
    clone.load_state_dict(saved)
    for a,b in zip(batcher.next(),clone.next()): np.testing.assert_array_equal(a,b)
