import sys
import os
import glob

def list_folder(path,pattern=""):
  filt = build_path(path,pattern)
  return glob.glob(filt)

def split_path(path):
  return {"base":os.path.dirname(path), "file":os.path.basename(path)}

def build_path(dirpath,fn):
  if dirpath.endswith('/'):
    path=dirpath+fn
  else:
    path=dirpath+"/"+fn
  return path

def sub_folder(fp, subfolder):
  spp=split_path(fp)
  return build_path(build_path(spp["base"],subfolder),spp["file"])

def replace_filename(fp,matchstr,repstr):
  spp=split_path(fp)
  return build_path(spp["base"],spp["file"].replace(matchstr,repstr))

def append_suffix(fp,suffix):
  spp=split_path(fp)
  fn,ext = os.path.splitext(spp["file"])
  return build_path(spp["base"],fn+suffix+ext)

def create_folders(fp):
  os.makedirs(fp,exist_ok=True)
  return
