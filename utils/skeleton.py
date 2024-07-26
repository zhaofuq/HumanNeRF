from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F


def lbs(J, parents, rot_mats ,v_template = None, lbs_weights = None,  posedirs = None, num_joints=24, dtype=torch.float32):

    batch_size = 1
    device = 'cpu'

    # Add shape contribution
    #v_shaped = v_template + blend_shapes(betas, shapedirs)
    v_shaped = v_template
    
    # Get the joints
    # NxJx3 array
    #J = vertices2joints(J_regressor, v_shaped)

    rot_mats = rot_mats.reshape([batch_size, -1, 3, 3])
    #rot_mats = batch_rodrigues(rot_mats.reshape(-1, 3)).reshape([batch_size, -1, 3, 3])
     
    if posedirs is not None:
        # 3. Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)
        pose_feature = (rot_mats[:, 1:, :, :] - ident).reshape([batch_size, -1])

        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs).reshape(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped
    else:
        v_posed = v_shaped

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

#     # 5. Do skinning:
#     # W is N x V x (J + 1)
   
#     W = lbs_weights.repeat([batch_size, 1, 1])

#     # (N x V x (J + 1)) x (N x (J + 1) x 16)
#     T = torch.matmul(W, A.reshape(batch_size, num_joints, 16)).reshape(batch_size, -1, 4, 4)

#     homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
#     v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
#     v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

#     verts = v_homo[:, :, :3, 0]
    
#     return verts, J_transformed, T, A
    return J_transformed, A

def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])

def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape

def batch_rodrigues(aa_rots):
    '''
    convert batch of rotations in axis-angle representation to matrix representation
    :param aa_rots: Nx3
    :return: mat_rots: Nx3x3
    '''

    dtype = aa_rots.dtype
    device = aa_rots.device

    batch_size = aa_rots.shape[0]

    angle = torch.norm(aa_rots + 1e-8, dim=1, keepdim=True)
    rot_dir = aa_rots / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    batch_size = rot_mats.shape[0]
    num_joints = joints.shape[1]
    device = rot_mats.device

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)
        
    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = torch.cat([joints, torch.zeros([batch_size, num_joints, 1, 1], dtype=dtype, device=device)],dim=2)
    init_bone = torch.matmul(transforms, joints_homogen)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    rel_transforms = transforms - init_bone
    
    return posed_joints, rel_transforms

def make_rotate_xyz(rx, ry, rz,anlge = True):
    if anlge:
        rx,ry,rz = np.radians(rx),np.radians(ry),np.radians(rz)
    sinX,sinY,sinZ = np.sin(rx),np.sin(ry),np.sin(rz)
    cosX,cosY,cosZ = np.cos(rx),np.cos(ry),np.cos(rz)
    Rx,Ry,Rz = np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))
    Rx[0, 0] = 1.0;Rx[1, 1] = cosX;Rx[1, 2] = -sinX;Rx[2, 1] = sinX;Rx[2, 2] = cosX
    Ry[0, 0] = cosY;Ry[0, 2] = sinY;Ry[1, 1] = 1.0;Ry[2, 0] = -sinY;Ry[2, 2] = cosY
    Rz[0, 0] = cosZ;Rz[0, 1] = -sinZ;Rz[1, 0] = sinZ;Rz[1, 1] = cosZ;Rz[2, 2] = 1.0
    R = np.matmul(np.matmul(Rx,Ry),Rz)
    return R

def make_rotate_batch_temp_xyz(pose_cube):
    R = np.zeros((pose_cube.shape[0],3,3))
    for i in range(pose_cube.shape[0]):
        R[i] = make_rotate_xyz( pose_cube[i][0][0] , pose_cube[i][0][1] , pose_cube[i][0][2]  ,False)
    return R

def rodrigues(r):
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(np.float64).eps)
    r_hat = r / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R

def with_zeros(x):
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))


def pack(x):
    return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

class Skeleton():
    def __init__(self, root= None):
        super(Skeleton,self).__init__()
        self.root = root
        self.path = os.path.join(root,'test_origin.skeleton')
        self.sk_trans = None
        self.sk_bones = None
        self.sk_joint_names = None
        self.sk_joint_names_sel = None
        self.sk_J = None
        
        self.gen_kintree_table()
        
        
    def gen_kintree_table(self):
        kintree_table = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        kintree_table = [int(float(item)) for item in kintree_table]
        kintree_table = np.array(kintree_table).reshape(1,-1)
        self.kintree_table = np.concatenate([kintree_table,np.arange(kintree_table.shape[1]).reshape(1,-1)], axis= 0).astype('uint32')
        self.id_to_col = {self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])}
        self.parent = {i: self.id_to_col[self.kintree_table[0, i]] for i in range(1, self.kintree_table.shape[1])}


    def load_data(self):
        
        with open(self.path,'r') as f:
            data = f.readlines()
        sk_trans = data[3][40:90].split()

        sk_trans = [float(item) for item in sk_trans]
        self.sk_trans = np.array(sk_trans).reshape(1,3)
        
        sk_bones = []
        for i in range(24):
            sk_bone_temp = data[6+i*3].split()[3:6]
            sk_bone_temp = [float(item) for item in sk_bone_temp]
            sk_bones.append(sk_bone_temp)
        self.sk_bones = np.stack(sk_bones,axis=0)
        
        # skeleton joints names
        sk_joint_names = []
        for i in range(24*3):
            sk_joint_name_temp = data[6+i].split()[0]
            sk_joint_names.append(sk_joint_name_temp)
        self.sk_joint_names = sk_joint_names
        
        # skeleton joints names sel
        sk_joint_names_sel = []
        for i in range(35):
            sk_joint_names_sel_temp = data[105+i*3].split()[0]
            sk_joint_names_sel.append(sk_joint_names_sel_temp)
        self.sk_joint_names_sel = sk_joint_names_sel
        

        # skeleton joints
        sk_J = []
        sk_J.append(self.sk_bones[0])
        for i in range(1,24):
            sk_J.append(sk_J[self.parent[i]] + sk_bones[i])
        sk_J = np.vstack(sk_J)
        self.sk_J = sk_J
        
        motion_cur = self.motion_cur_list.split()[1:]
        motion_cur = [float(item) for item in motion_cur]
        motion_cur = np.array(motion_cur).reshape(-1,1)
        motion_trans_cur = motion_cur[:3].reshape(1,3)
        self.motion_trans_cur = motion_trans_cur
        
        joint_names = self.sk_joint_names
        joint_names_sel = self.sk_joint_names_sel
        dofs_cur = np.zeros((24*3,1))
        for i in range(3,35):
            for j in range(24*3):
                if joint_names_sel[i]==joint_names[j]:
                    dofs_cur[j] = np.fmod((motion_cur[i] + np.pi *2) , (np.pi *2 * 2) ) - np.pi *2
                    break
        dofs_cur = dofs_cur.reshape(-1,1,3)
        self.dofs_cur = dofs_cur
        
        pose_cube_cur = dofs_cur.reshape((-1, 1, 3))  # 24 ,1 ,3 

        R_cur = make_rotate_batch_temp_xyz(pose_cube_cur)
        self.R_cur = R_cur
        
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        
        G[0] = with_zeros(np.hstack((R_cur[0], sk_J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
              G[i] = G[self.parent[i]].dot(with_zeros(np.hstack([R_cur[i],((sk_J[i, :]-sk_J[self.parent[i],:]).reshape([3,1]))])))
        self.G = G
        
        J_new2 = np.zeros((24,3))
        for i in range(24):
            J_new2[i] = G[i][:3,3].reshape(1,3)
 
        self.J_new2 = J_new2 #+ self.motion_trans_cur
        
    def load_motion(self, idx, motion_dir):
            path = os.path.join(self.root,motion_dir)
            with open(os.path.join(path),'r') as f:
                data = f.readlines()
            self.frame_number = len(data) - 1
            self.motion_cur_list = data[idx+1]

# def make_rotate_xyz(rx, ry, rz,anlge = True):
#     if anlge:
#         rx,ry,rz = np.radians(rx),np.radians(ry),np.radians(rz)
#     sinX,sinY,sinZ = np.sin(rx),np.sin(ry),np.sin(rz)
#     cosX,cosY,cosZ = np.cos(rx),np.cos(ry),np.cos(rz)
#     Rx,Ry,Rz = np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))
#     Rx[0, 0] = 1.0;Rx[1, 1] = cosX;Rx[1, 2] = -sinX;Rx[2, 1] = sinX;Rx[2, 2] = cosX
#     Ry[0, 0] = cosY;Ry[0, 2] = sinY;Ry[1, 1] = 1.0;Ry[2, 0] = -sinY;Ry[2, 2] = cosY
#     Rz[0, 0] = cosZ;Rz[0, 1] = -sinZ;Rz[1, 0] = sinZ;Rz[1, 1] = cosZ;Rz[2, 2] = 1.0
#     R = np.matmul(np.matmul(Rx,Ry),Rz)
#     return R


# def make_rotate_batch_temp_xyz(pose_cube):
#     R = np.zeros((pose_cube.shape[0],3,3))
#     for i in range(pose_cube.shape[0]):
#         R[i] = make_rotate_xyz( pose_cube[i][0][0] , pose_cube[i][0][1] , pose_cube[i][0][2]  ,False)
#     return R

# def rodrigues(r):
#     theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
#     # avoid zero divide
#     theta = np.maximum(theta, np.finfo(np.float64).eps)
#     r_hat = r / theta
#     cos = np.cos(theta)
#     z_stick = np.zeros(theta.shape[0])
#     m = np.dstack([
#       z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
#       r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
#       -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
#     ).reshape([-1, 3, 3])
#     i_cube = np.broadcast_to(
#       np.expand_dims(np.eye(3), axis=0),
#       [theta.shape[0], 3, 3]
#     )
#     A = np.transpose(r_hat, axes=[0, 2, 1])
#     B = r_hat
#     dot = np.matmul(A, B)
#     R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
#     return R

# def with_zeros(x):
#     return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))


# def pack(x):
#     return np.dstack((np.zeros((x.shape[0], 4, 3)), x))


# class Skeleton():
#     def __init__(self, root= None):
#         super(Skeleton,self).__init__()
#         self.root = root
#         self.path = os.path.join(root,'test_origin.skeleton')
#         self.sk_trans = None
#         self.sk_bones = None
#         self.sk_joint_names = None
#         self.sk_joint_names_sel = None
#         self.sk_J = None
        
#     def load_data(self):

#         with open(os.path.join(self.root,'kintree.txt'),'r') as f:
#             data = f.readlines()
#         kintree_table = data[1].split()
#         kintree_table = [int(float(item)) for item in kintree_table]
#         kintree_table = np.array(kintree_table).reshape(1,-1)
#         self.kintree_table = np.concatenate([kintree_table,np.arange(kintree_table.shape[1]).reshape(1,-1)], axis= 0).astype('uint32')
#         self.id_to_col = {self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])}
#         self.parent = {i: self.id_to_col[self.kintree_table[0, i]] for i in range(1, self.kintree_table.shape[1])}
        
#         with open(self.path,'r') as f:
#             data = f.readlines()
#         sk_trans = data[3][40:90].split()
#         sk_trans = [float(item) for item in sk_trans]
#         self.sk_trans = np.array(sk_trans).reshape(1,3)
        
        
#         sk_bones = []
#         for i in range(24):
#             sk_bone_temp = data[6+i*3].split()[3:6]
#             sk_bone_temp = [float(item) for item in sk_bone_temp]
#             sk_bones.append(sk_bone_temp)
#         self.sk_bones = np.stack(sk_bones,axis=0)
        
#         # skeleton joints names
#         sk_joint_names = []
#         for i in range(24*3):
#             sk_joint_name_temp = data[6+i].split()[0]
#             sk_joint_names.append(sk_joint_name_temp)
#         self.sk_joint_names = sk_joint_names
        
#         # skeleton joints names sel
#         sk_joint_names_sel = []
#         for i in range(35):
#             sk_joint_names_sel_temp = data[105+i*3].split()[0]
#             sk_joint_names_sel.append(sk_joint_names_sel_temp)
#         self.sk_joint_names_sel = sk_joint_names_sel
        

#         # skeleton joints
#         sk_J = []
#         sk_J.append(self.sk_bones[0])
#         for i in range(1,24):
#             sk_J.append(sk_J[self.parent[i]] + sk_bones[i])
#         sk_J = np.vstack(sk_J)
#         self.sk_J = sk_J
        
#         motion_cur = self.motion_cur_list.split()[1:]
#         motion_cur = [float(item) for item in motion_cur]
#         motion_cur = np.array(motion_cur).reshape(-1,1)
#         motion_trans_cur = motion_cur[:3].reshape(1,3)
#         self.motion_trans_cur = motion_trans_cur
        
#         joint_names = self.sk_joint_names
#         joint_names_sel = self.sk_joint_names_sel
#         dofs_cur = np.zeros((24*3,1))
#         for i in range(3,35):
#             for j in range(24*3):
#                 if joint_names_sel[i]==joint_names[j]:
#                     dofs_cur[j] = np.fmod((motion_cur[i] + np.pi *2) , (np.pi *2 * 2) ) - np.pi *2
#                     break
#         dofs_cur = dofs_cur.reshape(-1,1,3)
#         pose_cube_cur = dofs_cur.reshape((-1, 1, 3))  # 24 ,1 ,3 

#         R_cur = make_rotate_batch_temp_xyz(pose_cube_cur)
#         self.R_cur = R_cur

#         G = np.empty((kintree_table.shape[1], 4, 4))
#         G[0] = with_zeros(np.hstack((R_cur[0], sk_J[0, :].reshape([3, 1]))))
#         for i in range(1, kintree_table.shape[1]):
#               G[i] = G[self.parent[i]].dot(with_zeros(np.hstack([R_cur[i],((sk_J[i, :]-sk_J[self.parent[i],:]).reshape([3,1]))])))
                
#         self.G = G
                
#         J_new2 = np.zeros((24,3))
#         for i in range(24):
#             J_new2[i] = G[i][:3,3].reshape(1,3)
#         self.J_new2 = J_new2
        
#     def load_motion(self,idx):
#             path = os.path.join(self.root,'framesInit.motion')
#             with open(os.path.join(path),'r') as f:
#                 data = f.readlines()
#             self.frame_number = len(data) - 1
#             self.motion_cur_list = data[idx+1]