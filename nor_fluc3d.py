import numpy as np

# data=np.load(file="./channelflow_3d_100.npy")# lr_channelflow_3d_100.npy
# save_name = "./nor_channelflow_3d_100.npy"
data=np.load(file="./lr_channelflow_3d_100.npy")# lr_channelflow_3d_100.npy
save_name = "./nor_lr_channelflow_3d_100.npy"

data.shape

print(data.shape)
u,v,w=np.split(data,3,axis=-1)

ss=100



def nor(x,maxn, minn):
    xx=(x-minn)/(maxn-minn)
    return xx


uu=nor(u,1.1833968612118588,-0.0005995676493508181)
vv=nor(v,0.2780295508749168,-0.2311295648860864)
ww=nor(w,0.3014445281247533,-0.3043292534651477)


s=np.concatenate((uu,vv,ww),axis=-1)
print(s.shape)
np.save(file=save_name,arr=s)














