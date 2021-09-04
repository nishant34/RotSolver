import numpy as np 


def create_self_supervised_gt(num_images):
    a = np.eye(3)
    b = a[None,:,:]
    b = np.repeat(b,num_images,axis=0)
    b = b[None,:,:,:]
    b = np.repeat(b,num_images,axis=0)
    return b

def get_full_connected_adj_matrix(num_nodes):
   adj = np.ones((num_nodes, num_nodes),int)
   np.fill_diagonal(adj,0)
   return adj

"some image and camera utilities......"

def get_directions(h,w,f,cam_rots):
    "generating rays"
    x, y = np.meshgrid(w,h,dtype=np.float32)
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(w, dtype=np.float32),  # X-Axis (columns)
        np.arange(h, dtype=np.float32),  # Y-Axis (rows)
        indexing='xy')
    camera_dirs = np.stack(
        [(x - w * 0.5 + 0.5) / f,
         -(y - h * 0.5 + 0.5) / f, -np.ones_like(x)],
        axis=-1)
    directions = ((camera_dirs[None, ..., None, :] *
                   cam_rots[:, None, None, :3, :3]).sum(axis=-1))
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    directions = directions.reshape(-1,directions.shape[-1])
    viewdirs = viewdirs.reshape(-1,viewdirs.shape[-1])
    return directions, viewdirs

    






