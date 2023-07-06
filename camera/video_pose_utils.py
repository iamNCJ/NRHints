import numpy as np


def trans_t(t): return np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]],
    dtype=np.float32)


def rot_phi(phi): return np.array([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]],
    dtype=np.float32)


def rot_theta(th): return np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]],
    dtype=np.float32)


def pose_spherical(theta, phi, radius, is_z_up: bool = False):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    if is_z_up:
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


def gen_fix_light_rot_view(num_views=60, radius=4.5, pl_pos=None,
                           pl_intensity=None, is_z_up=False):
    if pl_intensity is None:
        pl_intensity = [25.0, 25.0, 25.0]
    if pl_pos is None:
        pl_pos = [0, 0.5 * 4.5, 0.866 * 4.5]
    render_poses = np.stack(
        [pose_spherical(angle, -30.0, radius, is_z_up) for angle in np.linspace(-180, 180, num_views + 1)[:-1]], 0)
    pls = np.array([pl_pos + pl_intensity]
                   ).repeat(render_poses.shape[0], axis=0)
    return render_poses.astype(np.float32), pls.astype(np.float32)


def gen_fix_view_rot_light(num_lights=60, radius=4.5, pl_intensity=None, view_theta=-180, view_phi=-30,
                           view_radius=4.5, is_z_up=False):
    if pl_intensity is None:
        pl_intensity = [25.0, 25.0, 25.0]
    pls = np.stack(
        [np.concatenate((pose_spherical(angle, -30.0, radius, is_z_up)[..., 3][0:3], np.array(pl_intensity)), -1) for angle
         in np.linspace(-180, 180, num_lights + 1)[:-1]], 0)
    render_poses = np.array(pose_spherical(view_theta, view_phi, view_radius, is_z_up))[
        None, ...].repeat(pls.shape[0], axis=0)
    return render_poses.astype(np.float32), pls.astype(np.float32)
