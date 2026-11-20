"""Typed immutable packet I/O and exact state equality for the new Supra profile."""
from dataclasses import dataclass
import hashlib,json,os,tempfile
from pathlib import Path

@dataclass(frozen=True)
class Checkpoint:
    path: str
    bytes: int
    sha256: str

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def source_record(path):
    p=Path(path).absolute(); b=p.read_bytes()
    return dict(file=p.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def file_pin(path):
    p=Path(path).absolute()
    with p.open('rb') as f:
        size=os.fstat(f.fileno()).st_size; sha=hashlib.file_digest(f,'sha256').hexdigest()
    return Checkpoint(str(p),size,sha)

def read_packet(pin):
    import torch
    if (not isinstance(pin,Checkpoint) or type(pin.bytes) is not int or pin.bytes<1
            or not isinstance(pin.sha256,str) or len(pin.sha256)!=64
            or any(c not in '0123456789abcdef' for c in pin.sha256)):
        raise ValueError('typed whole-file packet pin required')
    with Path(pin.path).absolute().open('rb') as f:
        before=os.fstat(f.fileno())
        if before.st_size!=pin.bytes or hashlib.file_digest(f,'sha256').hexdigest()!=pin.sha256:
            raise ValueError('whole packet bytes/SHA differs')
        f.seek(0); packet=torch.load(f,map_location='cpu',weights_only=True); after=os.fstat(f.fileno())
        fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if fields(before)!=fields(after):raise ValueError('packet changed during read')
    return packet

def write_new(path,writer):
    p=Path(path); fd,tmp=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name)
    try:
        with os.fdopen(fd,'wb') as f:writer(f);f.flush();os.fsync(f.fileno())
        os.link(tmp,p);os.unlink(tmp)
        d=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(d)
        finally:os.close(d)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return file_pin(p)

def save_packet(path,value):
    import torch
    return write_new(path,lambda f:torch.save(value,f))

def write_json(path,value):
    raw=(canonical(value)+'\n').encode();return write_new(path,lambda f:f.write(raw))

def new_directory(path):
    p=Path(path).absolute();p.mkdir(parents=True,exist_ok=False)
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return p

def tree(value,device):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().to(device=device).clone()
    if isinstance(value,dict):return {k:tree(v,device)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(tree(v,device)for v in value)
    return value

def equal(left,right):
    import torch
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor)and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left.detach().cpu(),right.detach().cpu())
    if type(left)is not type(right):return False
    if isinstance(left,dict):return left.keys()==right.keys()and all(equal(left[k],right[k])for k in left)
    if isinstance(left,(list,tuple)):return len(left)==len(right)and all(equal(a,b)for a,b in zip(left,right))
    return left==right
