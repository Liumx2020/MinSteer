"""Regress the multilingual direction sign using matching formal/casual records."""
import ast
from pathlib import Path
import tempfile
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
path=ROOT/'politeness/analyze_pca.py'
names={'load_jp_vector','load_en_vector','load_cn_vector','cosine_sim','pca_analysis'}
tree=ast.parse(path.read_text())
nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
assert len(nodes)==len(names)
ns={'torch':torch,'F':F}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)

with tempfile.TemporaryDirectory() as folder:
    folder=Path(folder)
    formal=torch.tensor([[[1.,2.,3.]],[[2.,3.,4.]]])
    delta=torch.tensor([1.,-2.,.5])
    casual=formal+delta
    torch.save({'generation_1':{'result':{'residuals':{'resid_pre':formal}}},
                'generation_2':{'result':{'residuals':{'resid_pre':casual}}}},folder/'jp.pt')
    torch.save({'direction':'casual_minus_formal','vectors':{'resid_pre':delta[None,:]}},folder/'en.pt')
    jp=ns['load_jp_vector'](folder/'jp.pt',0)
    en=ns['load_en_vector'](folder/'en.pt',0)
    cn=ns['load_cn_vector'](folder/'en.pt',0)
    assert torch.equal(jp,delta)
    assert torch.equal(en,jp) and torch.equal(cn,jp)
    assert ns['cosine_sim'](jp,en)>.999999
    assert ns['cosine_sim'](jp,cn)>.999999
print('JP, EN and CN preserve the same formal-to-casual direction.')
