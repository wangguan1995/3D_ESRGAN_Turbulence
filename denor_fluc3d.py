import numpy as np

data=np.load(file="")
data.shape

print(data.shape)
u,v,w=np.split(data,3,axis=-1)





def denor(x,maxn, minn):
    xx=x*(maxn-minn)+minn
    return xx


uu=denor(u,1.1833968612118588,-0.0005995676493508181)
vv=denor(v,0.2780295508749168,-0.2311295648860864)
ww=denor(w,0.3014445281247533,-0.3043292534651477)


s=np.concatenate((uu,vv,ww),axis=-1)
print(s.shape)
np.save(file="denor_.npy",arr=s)














