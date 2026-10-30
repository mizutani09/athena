#!/usr/bin/env python3
"""Extend an OPTAB rho-T table in temperature with audited log-slope tails.

No new microphysics is invented inside the source domain.  Above its maximum
temperature, each density row is continued with the median terminal log-log
slope.  The same controlled continuation is available below the supplied
range for cold numerical atmosphere cells.  Rosseland values in the
low-density fully-ionized limit are bounded below by electron scattering for
X=0.70; Planck absorption is not given that floor.  The generated HDF5 records
the operation and source checksum.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--metadata",type=Path)
    p.add_argument("--logt-min",type=float,default=2.5)
    p.add_argument("--logt-max",type=float,default=7.0)
    p.add_argument("--nt",type=int,default=192)
    p.add_argument("--tail-points",type=int,default=8)
    p.add_argument("--hydrogen",type=float,default=0.70)
    return p.parse_args()


def main():
    a=parse_args()
    with h5py.File(a.input,"r") as h:
        lr=np.asarray(h["axes/log10_density"]);lt=np.asarray(h["axes/log10_temperature"])
        lp=np.asarray(h["kappa/log10_planck"]);lross=np.asarray(h["kappa/log10_rosseland"])
        source_attrs={k:(v.item() if isinstance(v,np.generic) else v) for k,v in h.attrs.items()}
    if a.logt_min>=lt[0] or a.logt_max<=lt[-1] or a.nt<=lt.size:
        raise ValueError("extension must add both lower- and higher-T tails")
    newt=np.linspace(a.logt_min,a.logt_max,a.nt)
    result=[]
    for values in (lp,lross):
        out=np.empty((lr.size,newt.size))
        for i,row in enumerate(values):
            slope_lo=np.median(np.diff(row[:a.tail_points])/np.diff(lt[:a.tail_points]))
            slope_hi=np.median(np.diff(row[-a.tail_points:])/np.diff(lt[-a.tail_points:]))
            inside=(newt>=lt[0])&(newt<=lt[-1])
            out[i,inside]=np.interp(newt[inside],lt,row)
            cold=newt<lt[0];hot=newt>lt[-1]
            out[i,cold]=row[0]+slope_lo*(newt[cold]-lt[0])
            out[i,hot]=row[-1]+slope_hi*(newt[hot]-lt[-1])
        result.append(np.clip(out,-30.0,8.0))
    newp,newr=result
    # In the dilute, fully ionized regime the transport opacity cannot fall
    # below Thomson scattering.  Do not apply this to dense rows where the
    # source table contains additional physics and its own continuation.
    es=0.2*(1.0+a.hydrogen)
    dilute=lr<=-7.0
    hot=newt>=lt[-1]
    newr[np.ix_(dilute,hot)]=np.maximum(newr[np.ix_(dilute,hot)],np.log10(es))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(a.output,"w") as h:
        ga=h.create_group("axes");gk=h.create_group("kappa")
        ga.create_dataset("log10_density",data=lr).attrs["unit"]="g cm^-3"
        ga.create_dataset("log10_temperature",data=newt).attrs["unit"]="K"
        for name,val in (("log10_planck",newp),("log10_rosseland",newr)):
            gk.create_dataset(name,data=val).attrs["unit"]="log10(cm^2 g^-1)"
        gk.create_dataset("planck",data=10.0**newp).attrs["unit"]="cm^2 g^-1"
        gk.create_dataset("rosseland",data=10.0**newr).attrs["unit"]="cm^2 g^-1"
        h.attrs.update(source_attrs)
        h.attrs["generator"]="tools/rsg/extend_rsg_opacity.py"
        h.attrs["source_sha256"]=hashlib.sha256(a.input.read_bytes()).hexdigest()
        h.attrs["extension"]=("terminal log-log slope; dilute Rosseland floor "
                              f"kappa_es={es:.8g} cm^2/g for logRho<=-7")
        h.attrs["log10T_min"]=a.logt_min
        h.attrs["log10T_max"]=a.logt_max
    meta={"source":str(a.input.resolve()),"source_sha256":hashlib.sha256(a.input.read_bytes()).hexdigest(),
          "output":str(a.output.resolve()),"output_sha256":hashlib.sha256(a.output.read_bytes()).hexdigest(),
          "logrho":[float(lr[0]),float(lr[-1]),int(lr.size)],
          "logT":[float(newt[0]),float(newt[-1]),int(newt.size)],
          "tail_points":a.tail_points,"electron_scattering_floor":es,
          "source_attributes":source_attrs,
          "provenance_limitation":"source composition and license are not embedded in the supplied OPTAB snapshot"}
    path=a.metadata or a.output.with_suffix(a.output.suffix+".json")
    path.write_text(json.dumps(meta,indent=2)+"\n");print(json.dumps(meta,indent=2))


if __name__=="__main__":main()
