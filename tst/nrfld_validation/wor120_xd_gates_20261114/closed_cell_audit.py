import json
from pathlib import Path
import h5py
import numpy as np
root=Path('/work/mizutaniks/wor120/lte256_closed_probe')
p=next(root.glob('*.prim.*.athdf'))
u=next(root.glob('*.uov.*.athdf'))
with h5py.File(p) as f, h5py.File(u) as v:
    names=[x.decode() for x in f.attrs['VariableNames']]
    rho=np.asarray(f['prim'][names.index('rho')],dtype=float)
    press=np.asarray(f['prim'][names.index('press')],dtype=float)
    vel=[np.asarray(f['prim'][names.index('vel'+str(i))],dtype=float) for i in (1,2,3)]
    er=np.asarray(v['user_out_var'][2],dtype=float)
    temp=np.asarray(v['user_out_var'][0],dtype=float)
    x=np.asarray(f['x1v'])[:,None,None,:]
    y=np.asarray(f['x2v'])[:,None,:,None]
    z=np.asarray(f['x3v'])[:,:,None,None]
    radius=np.sqrt(x*x+y*y+z*z)
    speed=np.sqrt(sum(w*w for w in vel))
    ratio=er/rho
    proxy=np.sqrt((5.0/3.0)*press/rho)
    floor=rho<=1.01e-13
    def record(index):
        return {key:float(field[index]) for key,field in {'radius_Rsun':radius,'rho':rho,'press':press,'Er':er,'Tgas_K':temp,'speed':speed,'ideal_sound_proxy':proxy,'Er_over_rho':ratio}.items()}
    records={}
    for name,field in [('max_speed',speed),('max_Er_over_rho',ratio),('max_ideal_signal',speed+proxy),('min_temperature',-temp)]:
        records[name]=record(np.unravel_index(np.nanargmax(field),field.shape))
    if np.any(floor):
        idx=np.unravel_index(np.nanargmax(np.where(floor,ratio,-np.inf)),ratio.shape)
        records['max_Er_over_rho_on_force_inactive_floor']=record(idx)
    records['floor_cells']=int(floor.sum())
    records['time_code']=float(f.attrs['Time'])
out=Path('/work/mizutaniks/wor120/results/closed_stall_cells.json')
out.write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
