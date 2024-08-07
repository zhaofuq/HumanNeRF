import numpy as np

def camparams_to_nerf(Ks, Ts, output_path):
    
    with open(os.path.join(output_path,'Intrinsic.inf'), 'w') as f:
        for i in range(Ks.shape[0]):
            f.write('%d\n'%i)
            f.write('%f %f %f\n %f %f %f\n %f %f %f\n' % tuple(Ks[i].reshape(9).tolist()))
            f.write('\n')

    with open(os.path.join(output_path,'CamPose.inf'), 'w') as f:
        for i in range(Ts.shape[0]):
            A = Ts[i,0:3,:]
            tmp = np.concatenate( [A[0:3,2].T, A[0:3,0].T,A[0:3,1].T,A[0:3,3].T])
            f.write('%f %f %f %f %f %f %f %f %f %f %f %f\n' % tuple(tmp.tolist()))

def campose_to_extrinsic(camposes):
    if camposes.shape[1]!=12:
        raise Exception(" wrong campose data structure!")
        return
    
    res = np.zeros((camposes.shape[0],4,4))
    
    res[:,0:3,2] = camposes[:,0:3]
    res[:,0:3,0] = camposes[:,3:6]
    res[:,0:3,1] = camposes[:,6:9]
    res[:,0:3,3] = camposes[:,9:12]
    res[:,3,3] = 1.0
    
    return res


def read_intrinsics(fn_instrinsic):
    fo = open(fn_instrinsic)
    data= fo.readlines()
    i = 0
    Ks = []
    while i<len(data):
        if len(data[i])>5:
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            a = np.array(tmp)
            i = i+1
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            b = np.array(tmp)
            i = i+1
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            c = np.array(tmp)
            res = np.vstack([a,b,c])
            Ks.append(res)

        i = i+1
    Ks = np.stack(Ks)
    fo.close()

    return Ks
