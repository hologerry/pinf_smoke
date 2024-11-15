import json
import os
import sys

import cv2 as cv
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm, trange

from siren_basic import SineLayer


# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def set_rand_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, render_vel=False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i : i + chunk], render_vel=render_vel, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(
    H,
    W,
    K,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    ndc=True,
    near=0.0,
    far=1.0,
    use_viewdirs=False,
    c2w_staticcam=None,
    time_step=None,
    bkgd_color=None,
    render_vel=False,
    **kwargs,
):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    if time_step != None:
        time_step = time_step.expand(list(rays.shape[0:-1]) + [1])
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction, t)
        rays = torch.cat([rays, time_step], dim=-1)
    # Render and reshape
    all_ret = batchify_rays(rays, chunk, render_vel=render_vel, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    if bkgd_color is not None:
        torch_bkgd_color = torch.Tensor(bkgd_color).cuda()
        # rgb map for model: fine, coarse, merged, dynamic_fine, dynamic_coarse
        for _i in [
            "_map",
            "0",
            "h1",
            "h10",
            "h2",
            "h20",
        ]:  #  add background for synthetic scenes, for image-based supervision
            rgb_i, acc_i = "rgb" + _i, "acc" + _i
            if (rgb_i in all_ret) and (acc_i in all_ret):
                all_ret[rgb_i] = all_ret[rgb_i] + torch_bkgd_color * (1.0 - all_ret[acc_i][..., None])
    # if render_vel:
    #     import pdb
    #     pdb.set_trace()
    k_extract = ["rgb_map", "disp_map", "acc_map"]
    if "vel_map" in all_ret:
        k_extract += ["vel_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(
    render_poses,
    hwf,
    K,
    chunk,
    render_kwargs,
    bbox_model=None,
    gt_imgs=None,
    savedir=None,
    render_factor=0,
    render_steps=None,
    bkgd_color=None,
    render_vel=False,
):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    psnrs = []
    lpipss = []
    ssims = []
    lpips_net = LPIPS().cuda()  # input should be [-1, 1] or [0, 1] (normalize=True)

    rgbs = []
    disps = []
    vels = []
    cur_timestep = None
    # start = 0
    # print("len render_poses", render_poses.shape)
    for i, c2w in tqdm(enumerate((render_poses)), total=len(render_poses)):
        # i = i + start
        # print(i, time.time() - t)
        if render_steps is not None:
            cur_timestep = render_steps[i]
        if render_vel:
            rgb, disp, acc, vel_map, extras = render(
                H,
                W,
                K,
                chunk=chunk,
                c2w=c2w[:3, :4],
                bbox_model=bbox_model,
                time_step=cur_timestep,
                render_vel=render_vel,
                **render_kwargs,
            )
        else:
            rgb, disp, acc, extras = render(
                H,
                W,
                K,
                chunk=chunk,
                c2w=c2w[:3, :4],
                bbox_model=bbox_model,
                time_step=cur_timestep,
                render_vel=render_vel,
                **render_kwargs,
            )

        # vels.append(vel_map.cpu().numpy())
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        rgb8 = to8b(rgbs[-1])
        filename = os.path.join(savedir, "rgb_{:03d}.png".format(i))
        imageio.imwrite(filename, rgb8)
        # if savedir is not None:
        # vel_mask = (rgbs[-1] > 0.1).any(-1)
        # vel_map[450:] = 0
        # if render_vel:
        #     save_quiver_plot(
        #         vel_map[..., 0].cpu().numpy() * vel_mask,
        #         vel_map[..., 1].cpu().numpy() * vel_mask,
        #         64,
        #         os.path.join(savedir, "vel_{:03d}.png".format(i)),
        #         scale=0.05,
        #     )

        # for rgb_i in ["rgbh1", "rgbh2", "rgb0"]:
        #     if rgb_i in extras:
        #         _data = extras[rgb_i].cpu().numpy()
        #         other_rgbs.append(_data)
        # if len(other_rgbs) >= 1:
        #     other_rgb8 = np.concatenate(other_rgbs, axis=1)
        #     other_rgb8 = to8b(other_rgb8)
        #     filename = os.path.join(savedir, "_{:03d}.png".format(i))
        #     imageio.imwrite(filename, other_rgb8)

        # if gt_imgs is not None:
        # other_rgbs.append(gt_imgs[i])
        gt_img = torch.tensor(gt_imgs[i], dtype=torch.float32)  # [H, W, 3]
        lpips_value = lpips_net(rgb.permute(2, 0, 1), gt_img.permute(2, 0, 1), normalize=True).item()
        p = -10.0 * np.log10(np.mean(np.square(rgb.detach().cpu().numpy() - gt_img.cpu().numpy())))
        ssim_value = structural_similarity(gt_img.cpu().numpy(), rgb.cpu().numpy(), data_range=1.0, channel_axis=2)

        lpipss.append(lpips_value)
        psnrs.append(p)
        ssims.append(ssim_value)
        # print(f"RENDER: PSNR: {p:.5g}, SSIM: {ssim_value:.5g}, LPIPS: {lpips_value:.5g}")
        imageio.imsave(os.path.join(savedir, "test_rgb_{:03d}.png".format(i)), rgb8)
        gt8 = to8b(gt_img.cpu().numpy())
        imageio.imsave(os.path.join(savedir, "test_gt_{:03d}.png".format(i)), gt8)

    # if gt_imgs is not None:
    avg_psnr = sum(psnrs) / len(psnrs)
    print(f"RENDER: Avg PSNR: ", avg_psnr)
    avg_ssim = sum(ssims) / len(ssims)
    print(f"RENDER: Avg SSIM: ", avg_ssim)
    avg_lpips = sum(lpipss) / len(lpipss)
    print(f"RENDER: Avg LPIPS: ", avg_lpips)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def render_future_pred(
    render_poses,
    hwf,
    K,
    time_steps,
    savedir,
    gt_imgs,
    bbox_model,
    rx=128,
    ry=192,
    rz=128,
    save_fields=False,
    vort_particles=None,
    project_solver=None,
    get_vel_der_fn=None,
    **render_kwargs,
):
    H, W, focal = hwf
    dt = time_steps[1] - time_steps[0]
    render_kwargs.update(chunk=512 * 16)
    psnrs = []
    lpipss = []
    ssims = []
    lpips_net = LPIPS().cuda()  # input should be [-1, 1] or [0, 1] (normalize=True)

    # construct simulation domain grid
    xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, rx), torch.linspace(0, 1, ry), torch.linspace(0, 1, rz)])
    coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)  # [X, Y, Z, 3]
    coord_3d_world = bbox_model.sim2world(coord_3d_sim)  # [X, Y, Z, 3]

    # initialize density field
    starting_frame = 89
    n_pred = 30
    time_step = torch.ones_like(coord_3d_world[..., :1]) * time_steps[starting_frame]
    coord_4d_world = torch.cat([coord_3d_world, time_step], dim=-1)  # [X, Y, Z, 4]
    den = batchify_query(
        coord_4d_world, lambda pts: render_kwargs["network_query_fn"](pts, None, render_kwargs["network_fine"])
    )  # [X, Y, Z, 1]
    vel = batchify_query(coord_4d_world, render_kwargs["network_query_fn_vel"])  # [X, Y, Z, 3]

    source_height = 0.25
    # y_start = int(source_height * ry)
    # print("y_start: {}".format(y_start))
    # render_kwargs.update(y_start=y_start)
    # proj_y = render_kwargs['proj_y']
    for idx, i in enumerate(trange(starting_frame + 1, starting_frame + n_pred + 1, desc="Predicting future")):
        c2w = render_poses[0]
        mask_to_sim = coord_3d_sim[..., 1] > source_height
        n_substeps = 1
        use_reflect = False
        render_kwargs["bbox_model"] = bbox_model
        if vort_particles is not None:
            confinement_field = vort_particles(coord_3d_sim, i)
            # print(
            #     "Vortex energy over velocity: {:.2f}%".format(
            #         torch.norm(confinement_field, dim=-1).pow(2).sum() / torch.norm(vel, dim=-1).pow(2).sum() * 100
            #     )
            # )
        else:
            confinement_field = torch.zeros_like(vel)
        vel = vel + confinement_field

        for _ in range(n_substeps):
            dt_ = dt / n_substeps
            den, _ = advect_SL(den, vel, coord_3d_sim, dt_, **render_kwargs)
            if use_reflect:
                vel_half_step, _ = advect_SL(vel, vel, coord_3d_sim, dt_ / 2, **render_kwargs)
                vel_half_proj = vel_half_step.clone()
                # vel_half_proj[..., 2] *= -1
                # vel_half_proj[:, y_start:y_start + proj_y] = project_solver.Poisson(vel_half_proj[:, y_start:y_start + proj_y])
                # vel_half_proj[..., 2] *= -1
                vel_reflect = 2 * vel_half_proj - vel_half_step
                vel, _ = advect_SL(vel_reflect, vel_half_proj, coord_3d_sim, dt_ / 2, **render_kwargs)
                # vel[..., 2] *= -1
                # vel[:, y_start:y_start + proj_y] = project_solver.Poisson(vel[:, y_start:y_start + proj_y])
                # vel[..., 2] *= -1
            else:
                vel, _ = advect_SL(vel, vel, coord_3d_sim, dt_, **render_kwargs)
                # vel[..., 2] *= -1  # world coord is left handed, while solver assumes right handed
                # vel[:, y_start:y_start + proj_y] = project_solver.Poisson(vel[:, y_start:y_start + proj_y])
                # vel[..., 2] *= -1

        coord_4d_world[..., 3] = time_steps[i]  # sample density source at current moment
        den[~mask_to_sim] = batchify_query(
            coord_4d_world[~mask_to_sim],
            lambda pts: render_kwargs["network_query_fn"](pts, None, render_kwargs["network_fine"]),
        )
        vel[~mask_to_sim] = batchify_query(coord_4d_world[~mask_to_sim], render_kwargs["network_query_fn_vel"])

        # if save_fields:
        #     save_fields_to_vti(  # type: ignore
        #         vel.permute(2, 1, 0, 3).cpu().numpy(),
        #         den.permute(2, 1, 0, 3).cpu().numpy(),
        #         os.path.join(savedir, "fields_{:03d}.vti".format(idx)),
        #     )
        #     print("Saved fields to {}".format(os.path.join(savedir, "fields_{:03d}.vti".format(idx))))
        rgb, _, _, _ = render(
            H, W, K, c2w=c2w[:3, :4], time_step=time_steps[i][None], render_grid=True, den_grid=den, **render_kwargs
        )
        # rgb = rgb[90:960, 45:540]
        rgb8 = to8b(rgb.cpu().numpy())
        # [90:960, 45:540]
        gt_img = torch.tensor(gt_imgs[i].squeeze(), dtype=torch.float32)  # [H, W, 3]
        lpips_value = lpips_net(rgb.permute(2, 0, 1), gt_img.permute(2, 0, 1), normalize=True).item()

        p = -10.0 * np.log10(np.mean(np.square(rgb.detach().cpu().numpy() - gt_img.cpu().numpy())))
        ssim_value = structural_similarity(gt_img.cpu().numpy(), rgb.cpu().numpy(), data_range=1.0, channel_axis=2)
        lpipss.append(lpips_value)
        psnrs.append(p)
        ssims.append(ssim_value)
        # print(f"FUTURE: PSNR: {p:.5g}, SSIM: {ssim_value:.5g}, LPIPS: {lpips_value:.5g}")

        imageio.imsave(os.path.join(savedir, "fut_rgb_{:03d}.png".format(idx)), rgb8)
        gt8 = to8b(gt_img.cpu().numpy())
        imageio.imsave(os.path.join(savedir, "fut_gt_{:03d}.png".format(idx)), gt8)
        print("saved", os.path.join(savedir, "fut_rgb_{:03d}.png".format(idx)))

    if gt_imgs is not None:
        avg_psnr = sum(psnrs) / len(psnrs)
        print(f"FUTURE: Avg PSNR: ", avg_psnr)
        avg_ssim = sum(ssims) / len(ssims)
        print(f"FUTURE: Avg SSIM: ", avg_ssim)
        avg_lpips = sum(lpipss) / len(lpipss)
        print(f"FUTURE: Avg LPIPS: ", avg_lpips)
        with open(
            os.path.join(savedir, "psnrs_{:0.2f}_ssim_{:.2g}_lpips_{:.2g}.json".format(avg_psnr, avg_ssim, avg_lpips)),
            "w",
        ) as fp:
            json.dump(psnrs, fp)


def render_advect_den(
    render_poses,
    hwf,
    K,
    time_steps,
    savedir,
    gt_imgs,
    bbox_model,
    rx=128,
    ry=192,
    rz=128,
    save_fields=False,
    vort_particles=None,
    get_vel_der_fn=None,
    **render_kwargs,
):
    H, W, focal = hwf
    dt = time_steps[1] - time_steps[0]
    render_kwargs.update(chunk=512 * 32)
    psnrs = []

    psnrs = []
    lpipss = []
    ssims = []
    lpips_net = LPIPS().cuda()  # input should be [-1, 1] or [0, 1] (normalize=True)

    # construct simulation domain grid
    xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, 80), torch.linspace(0, 1, 142), torch.linspace(0, 1, 80)])
    coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)  # [X, Y, Z, 3]
    coord_3d_world = bbox_model.sim2world(coord_3d_sim)  # [X, Y, Z, 3]

    xs_100, ys_178, zs_100 = torch.meshgrid(
        [torch.linspace(0, 1, 100), torch.linspace(0, 1, 178), torch.linspace(0, 1, 100)]
    )
    coord_3d_sim_den = torch.stack([xs_100, ys_178, zs_100], dim=-1)  # [X, Y, Z, 3]
    coord_3d_world_den = bbox_model.sim2world(coord_3d_sim_den)  # [X, Y, Z, 3]

    # initialize density field
    time_step = torch.ones_like(coord_3d_world[..., :1]) * time_steps[0]
    coord_4d_world = torch.cat([coord_3d_world, time_step], dim=-1)  # [X, Y, Z, 4]
    # pdb.set_trace()
    den = batchify_query(
        coord_4d_world, lambda pts: render_kwargs["network_query_fn"](pts, None, render_kwargs["network_fine"])
    )  # [X, Y, Z, 1]
    den[..., -1:] = F.relu(den[..., -1:])
    vel = batchify_query(coord_4d_world, render_kwargs["network_query_fn_vel"])  # [X, Y, Z, 3]
    # grid_savedir =

    source_height = 0.25
    for i, c2w in tqdm(enumerate((render_poses)), total=len(render_poses), desc="Advecting density"):
        # gt_den = np.load("./data/ScalarReal/00000_syn_GT/_density/density_{:06}.npz".format(60 + i))["data"]
        # gt_vel = np.load("./data/ScalarReal/00000_syn_GT/_velocity/velocity_{:06}.npz".format(60 + i))["data"]
        # positive_den = den[den[...,-1]>0]
        # positive_den_position = coord_3d_world_den[gt_den[...,-1]>0.1]
        # positive_den_vel = gt_vel[gt_den[...,-1]>0.1]
        # np.save(f"./gt_particles_scene_0/position_{i}", positive_den_position.cpu().numpy())
        # np.save(f"./gt_particles_scene_0/velocity_{i}", positive_den_vel)
        # print(i, positive_den_position.shape)
        # v,f,n,val = measure.marching_cubes(den[...,-1].cpu().numpy(),0.9)
        # mesh = trimesh.Trimesh(vertices = v, faces = f)
        # update simulation den and source den
        mask_to_sim = coord_3d_sim[..., 1] > source_height
        if i > 0:
            coord_4d_world[..., 3] = time_steps[i - 1]  # sample velocity at previous moment

            # coord_4d_world.requires_grad = True
            vel = batchify_query(coord_4d_world, render_kwargs["network_query_fn_vel"])  # [X, Y, Z, 3]

            vel_confined = vel
            den, vel = advect_SL(den, vel_confined, coord_3d_sim, dt, RK=2, **render_kwargs)

            # zero grad for coord_4d_world
            # coord_4d_world.grad = None
            # coord_4d_world = coord_4d_world.detach()

            coord_4d_world[..., 3] = time_steps[i]  # source density at current moment
            # source_den = batchify_query(coord_4d_world, lambda pts: render_kwargs['network_query_fn'](pts, None, render_kwargs['network_fn']))
            den[~mask_to_sim] = batchify_query(
                coord_4d_world[~mask_to_sim],
                lambda pts: render_kwargs["network_query_fn"](pts, None, render_kwargs["network_fine"]),
            )
        den_reconstruction = batchify_query(
            coord_4d_world, lambda pts: render_kwargs["network_query_fn"](pts, None, render_kwargs["network_fine"])
        )
        vel_curr = batchify_query(coord_4d_world, render_kwargs["network_query_fn_vel"])
        # print(confinement_field.shape)
        np.save(os.path.join(savedir, f"density_advection_grid_{i}.npy"), den[..., -1].detach().cpu().numpy())
        np.save(
            os.path.join(savedir, f"density_reconstruction_grid_{i}.npy"),
            den_reconstruction[..., -1].detach().cpu().numpy(),
        )

        np.save(os.path.join(savedir, f"velocity_grid_{i}.npy"), vel_curr[...].detach().cpu().numpy())
        if save_fields:
            save_fields_to_vti(  # type: ignore
                vel.permute(2, 1, 0, 3).detach().cpu().numpy(),
                den.permute(2, 1, 0, 3).detach().cpu().numpy(),
                os.path.join(savedir, "fields_{:03d}.vti".format(i)),
            )

        render_kwargs["bbox_model"] = bbox_model
        rgb = render(
            H, W, K, c2w=c2w[:3, :4], time_step=time_steps[i][None], render_grid=True, den_grid=den, **render_kwargs
        )[0]
        rgb8 = to8b(rgb.detach().cpu().numpy())
        gt_img = torch.tensor(gt_imgs[i].squeeze(), dtype=torch.float32)  # [H, W, 3]

        p = -10.0 * np.log10(np.mean(np.square(rgb.detach().cpu().numpy() - gt_img.cpu().numpy())))
        ssim_value = structural_similarity(gt_img.cpu().numpy(), rgb.cpu().numpy(), data_range=1.0, channel_axis=2)
        lpips_value = lpips_net(rgb.permute(2, 0, 1), gt_img.permute(2, 0, 1), normalize=True).item()

        psnrs.append(p)
        ssims.append(ssim_value)
        lpipss.append(lpips_value)

        # print(f"ADVECT: PSNR: {p:.5g}, SSIM: {ssim_value:.5g}, LPIPS: {lpips_value:.5g}")
        imageio.imsave(os.path.join(savedir, "adv_rgb_{:03d}.png".format(i)), rgb8)
        gt8 = to8b(gt_img.detach().cpu().numpy())
        imageio.imsave(os.path.join(savedir, "adv_gt_{:03d}.png".format(i)), gt8)

        # imageio.imsave(os.path.join(savedir, "rgb_{:03d}.png".format(i)), rgb8)
        # imageio.imsave(os.path.join(savedir, "gt_{:03d}.png".format(i)), gt_img.detach().cpu())

    # if gt_imgs is not None:
    avg_psnr = sum(psnrs) / len(psnrs)
    print(f"ADVECT: Avg PSNR: ", avg_psnr)
    avg_ssim = sum(ssims) / len(ssims)
    print(f"ADVECT: Avg SSIM: ", avg_ssim)
    avg_lpips = sum(lpipss) / len(lpipss)
    print(f"ADVECT: Avg LPIPS: ", avg_lpips)
    with open(
        os.path.join(
            savedir, "advect_psnrs_{:0.2f}_ssim_{:.2g}_lpips_{:.2g}.json".format(avg_psnr, avg_ssim, avg_lpips)
        ),
        "w",
    ) as fp:
        json.dump(psnrs, fp)


def create_nerf(args, vel_model=None, bbox_model=None, ndim=3):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, ndim)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed, dim=ndim)
    output_ch = 4  # 5 if args.N_importance > 0 else 4
    skips = [4]

    my_model_dict = {
        "nerf": NeRF,
        "siren": SIREN_NeRFt,
        "hybrid": SIREN_Hybrid,
    }
    model_args = {}
    if args.fading_layers > 0:
        if args.net_model == "siren":
            model_args["fading_fin_step"] = args.fading_layers
        elif args.net_model == "hybrid":
            model_args["fading_fin_step_static"] = args.fading_layers
            model_args["fading_fin_step_dynamic"] = args.fading_layers
    if bbox_model is not None:
        model_args["bbox_model"] = bbox_model

    my_model = my_model_dict[args.net_model]

    model = my_model(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs,
        **model_args,
    )
    if args.net_model == "hybrid":
        model.toDevice()
    model = model.cuda()

    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = my_model(
            D=args.netdepth_fine,
            W=args.netwidth_fine,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
            **model_args,
        )
        if args.net_model == "hybrid":
            model_fine.toDevice()
        model_fine = model_fine.cuda()
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs, viewdirs, network_fn, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=args.netchunk
    )

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    vel_optimizer = None
    if vel_model is not None:
        vel_grad_vars = list(vel_model.parameters())
        vel_optimizer = torch.optim.Adam(params=vel_grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################
    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if "tar" in f
        ]

    # print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Load model
        if args.net_model == "hybrid":
            model.static_model.load_state_dict(ckpt["network_fn_state_dict_static"])
            if model_fine is not None:
                model_fine.static_model.load_state_dict(ckpt["network_fine_state_dict_static"])
            model.dynamic_model.load_state_dict(ckpt["network_fn_state_dict_dynamic"])
            if model_fine is not None:
                model_fine.dynamic_model.load_state_dict(ckpt["network_fine_state_dict_dynamic"])
        else:
            model.load_state_dict(ckpt["network_fn_state_dict"])
            if model_fine is not None:
                model_fine.load_state_dict(ckpt["network_fine_state_dict"])

        if vel_model is not None:
            if "network_vel_state_dict" in ckpt:
                vel_model.load_state_dict(ckpt["network_vel_state_dict"])
        if vel_optimizer is not None:
            if "vel_optimizer_state_dict" in ckpt:
                vel_optimizer.load_state_dict(ckpt["vel_optimizer_state_dict"])
    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": args.perturb,
        "N_importance": args.N_importance,
        "network_fine": model_fine,
        "N_samples": args.N_samples,
        "network_fn": model,
        "use_viewdirs": args.use_viewdirs,
        "raw_noise_std": args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != "llff" or args.no_ndc:
        # print("Not ndc!")
        render_kwargs_train["ndc"] = False
        render_kwargs_train["lindisp"] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, vel_optimizer


def raw2outputs(raw_list, z_vals, rays_d, raw_noise_std=0, pytest=False, remove99=False, render_vel=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw_list: a list of tensors in shape [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    noise = 0.0
    alpha_list = []
    color_list = []
    for raw in raw_list:
        if raw is None:
            continue
        if raw_noise_std > 0.0:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(42)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        if remove99:
            alpha = torch.where(alpha > 0.99, torch.zeros_like(alpha), alpha)
        if render_vel:
            rgb = raw[..., :3]
        else:
            rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        alpha_list += [alpha]
        color_list += [rgb]

    densTiStack = torch.stack([1.0 - alpha for alpha in alpha_list], dim=-1)
    # [N_rays, N_samples, N_raws]
    densTi = torch.prod(densTiStack, dim=-1, keepdim=True)
    # [N_rays, N_samples]
    densTi_all = torch.cat([densTiStack, densTi], dim=-1)
    # [N_rays, N_samples, N_raws + 1]
    Ti_all = torch.cumprod(densTi_all + 1e-10, dim=-2)  # accu along samples
    Ti_all = Ti_all / (densTi_all + 1e-10)
    # [N_rays, N_samples, N_raws + 1], exclusive
    weights_list = [alpha * Ti_all[..., -1] for alpha in alpha_list]  # a list of [N_rays, N_samples]
    self_weights_list = [
        alpha_list[alpha_i] * Ti_all[..., alpha_i] for alpha_i in range(len(alpha_list))
    ]  # a list of [N_rays, N_samples]

    def weighted_sum_of_samples(wei_list, content_list=None, content=None):
        content_map_list = []
        if content_list is not None:
            content_map_list = [
                torch.sum(weights[..., None] * ct, dim=-2)
                # [N_rays, N_content], weighted sum along samples
                for weights, ct in zip(wei_list, content_list)
            ]
        elif content is not None:
            content_map_list = [
                torch.sum(weights * content, dim=-1)
                # [N_rays], weighted sum along samples
                for weights in wei_list
            ]
        content_map = torch.stack(content_map_list, dim=-1)
        # [N_rays, (N_contentlist,) N_raws]
        content_sum = torch.sum(content_map, dim=-1)
        # [N_rays, (N_contentlist,)]
        return content_sum, content_map

    if not render_vel:
        rgb_map, _ = weighted_sum_of_samples(weights_list, color_list)  # [N_rays, 3]
    else:
        mask = color_list[0][..., -1] > 0.1
        rgb_map = color_list[0][
            :, int(color_list[0].shape[1] / 3.5), :3
        ]  # *mask[:, int(color_list[0].shape[1]/3.5), None]
    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map, _ = weighted_sum_of_samples(weights_list, None, 1)  # [N_rays]

    _, rgb_map_stack = weighted_sum_of_samples(self_weights_list, color_list)
    _, acc_map_stack = weighted_sum_of_samples(self_weights_list, None, 1)

    # Estimated depth map is expected distance.
    # Disparity map is inverse depth.
    depth_map, _ = weighted_sum_of_samples(weights_list, None, z_vals)  # [N_rays]
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)
    # alpha * Ti
    weights = (1.0 - densTi)[..., 0] * Ti_all[..., -1]  # [N_rays, N_samples]

    # weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    # rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    # depth_map = torch.sum(weights * z_vals, -1)
    # acc_map = torch.sum(weights, -1)

    return rgb_map, disp_map, acc_map, weights, depth_map, Ti_all[..., -1], rgb_map_stack, acc_map_stack


def save_quiver_plot(u, v, res, save_path, scale=0.00000002):
    """
    Args:
        u: [H, W], vel along x (W)
        v: [H, W], vel along y (H)
        res: resolution of the plot along the longest axis; if None, let step = 1
        save_path:
    """
    import matplotlib
    import matplotlib.pyplot as plt

    H, W = u.shape
    y, x = np.mgrid[0:H, 0:W]
    axis_len = max(H, W)
    step = 1 if res is None else axis_len // res
    xq = [i[::step] for i in x[::step]]
    yq = [i[::step] for i in y[::step]]
    uq = [i[::step] for i in u[::step]]
    vq = [i[::step] for i in v[::step]]

    uv_norm = np.sqrt(np.array(uq) ** 2 + np.array(vq) ** 2).max()
    short_len = min(H, W)
    matplotlib.rcParams["font.size"] = 10 / short_len * axis_len
    fig, ax = plt.subplots(figsize=(10 / short_len * W, 10 / short_len * H))
    q = ax.quiver(xq, yq, uq, vq, pivot="tail", angles="uv", scale_units="xy", scale=scale / step)
    ax.invert_yaxis()
    plt.quiverkey(q, X=0.6, Y=1.05, U=uv_norm, label=f"Max arrow length = {uv_norm:.2g}", labelpos="E")
    plt.savefig(save_path)
    plt.close()
    return


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    retraw=False,
    lindisp=False,
    perturb=0.0,
    N_importance=0,
    network_fine=None,
    raw_noise_std=0.0,
    verbose=False,
    pytest=False,
    render_vel=True,
    has_t=False,
    vel_model=None,
    netchunk=1024 * 64,
    warp_fading_dt=None,
    network_query_fn_vel=None,
    warp_mod="rand",
    render_grid=False,
    den_grid=None,
    bbox_model=None,
    remove99=False,
):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

      warp_fading_dt, to train nearby frames with flow-based warping, fading*delt_t
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    if render_grid:
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        time_step = ray_batch[0, -1]
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
        z_vals = near * (1.0 - t_vals) + far * (t_vals)

        z_vals = z_vals.expand([N_rays, N_samples])
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
        pts = torch.cat([pts, time_step * torch.ones((pts.shape[0], pts.shape[1], 1))], -1)  # [..., 4]
        pts_flat = torch.reshape(pts, [-1, 4])

        bbox_mask = bbox_model.insideMask(pts_flat[..., :3], to_float=False)
        if bbox_mask.sum() == 0:
            bbox_mask[0] = True  # in case zero rays are inside the bbox
        # bbox_mask = torch.ones_like(bbox_mask)
        pts = pts_flat[bbox_mask]

        ret = {}
        out_dim = 4
        raw_flat_den = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)

        pts_world = pts[..., :3]
        pts_sim = bbox_model.world2sim(pts_world)
        pts_sample = pts_sim * 2 - 1  # ranging [-1, 1]

        den_grid = den_grid
        den_grid_d = den_grid[..., [3]][None, ...].permute([0, 4, 3, 2, 1])
        #         den_grid_r = den_grid[...,[0]][None, ...].permute([0, 4, 3, 2, 1])
        #         den_grid_g = den_grid[...,[1]][None, ...].permute([0, 4, 3, 2, 1])
        #         den_grid_b = den_grid[...,[2]][None, ...].permute([0, 4, 3, 2, 1])

        den_sampled = F.grid_sample(den_grid_d, pts_sample[None, ..., None, None, :], align_corners=True).reshape(
            -1, 1
        )
        # r_sampled = F.grid_sample(den_grid_r, pts_sample[None, ..., None, None, :], align_corners=True).reshape(-1, 1)
        # g_sampled = F.grid_sample(den_grid_g, pts_sample[None, ..., None, None, :], align_corners=True).reshape(-1, 1)
        # b_sampled = F.grid_sample(den_grid_b, pts_sample[None, ..., None, None, :], align_corners=True).reshape(-1, 1)

        raw_flat_den[bbox_mask] = network_query_fn(pts, None, network_fn)
        # if den_sampled.shape[0]>1:
        #     pdb.set_trace()
        raw_flat_den[bbox_mask, 3] = den_sampled[..., 0]
        # raw_flat_den[bbox_mask,0] = r_sampled[...,0]
        # raw_flat_den[bbox_mask,1] = g_sampled[...,0]
        # raw_flat_den[bbox_mask,2] = b_sampled[...,0]
        raw_flat_den = raw_flat_den.reshape(-1, 64, 4)

        rgb_map, disp_map, acc_map, weights, depth_map, _, rgb_map_stack, acc_map_stack = raw2outputs(
            [raw_flat_den], z_vals, rays_d
        )

        ret["rgb_map"] = rgb_map
        ret["disp_map"] = disp_map
        ret["acc_map"] = acc_map
        return ret
    elif render_vel:
        ret = {}
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        time_step = ray_batch[0, -1]
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
        z_vals = near * (1.0 - t_vals) + far * (t_vals)

        z_vals = z_vals.expand([N_rays, N_samples])
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
        pts = torch.cat([pts, time_step * torch.ones((pts.shape[0], pts.shape[1], 1))], -1)  # [..., 4]
        pts_flat = torch.reshape(pts, [-1, 4])

        bbox_mask = bbox_model.insideMask(pts_flat[..., :3], to_float=False)
        if bbox_mask.sum() == 0:
            bbox_mask[0] = True  # in case zero rays are inside the bbox
        # bbox_mask = torch.ones_like(bbox_mask)
        pts = pts_flat[bbox_mask]

        out_dim = 3
        raw_flat_vel = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)
        raw_flat_vel[bbox_mask] = network_query_fn_vel(pts)  # raw_vel
        raw_vel = raw_flat_vel.reshape(N_rays, N_samples, out_dim)
        out_dim = 1
        raw_flat_den = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)
        raw_flat_den[bbox_mask] = network_query_fn(pts, None, network_fn)[..., [-1]]  # raw_den
        raw_den = raw_flat_den.reshape(N_rays, N_samples, out_dim)
        raw = torch.cat([raw_vel, raw_den], -1)
        rgb_map, disp_map, acc_map, weights, depth_map, _, rgb_map_stack, acc_map_stack = raw2outputs(
            [raw], z_vals, rays_d, render_vel=True
        )
        vel_map = rgb_map[..., :2]
        ret["vel_map"] = vel_map
    else:
        ret = {}
    # elif render_vel:
    #     out_dim = 3
    #     raw_flat_vel = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)
    #     raw_flat_vel[bbox_mask] = network_query_fn_vel(pts)[0]  # raw_vel
    #     raw_vel = raw_flat_vel.reshape(N_rays, N_samples, out_dim)
    #     out_dim = 1
    #     raw_flat_den = torch.zeros([N_rays, N_samples, out_dim]).reshape(-1, out_dim)
    #     raw_flat_den[bbox_mask] = network_query_fn(pts)  # raw_den
    #     raw_den = raw_flat_den.reshape(N_rays, N_samples, out_dim)
    #     raw = torch.cat([raw_vel, raw_den], -1)
    #     rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, render_vel=render_vel)
    #     vel_map = rgb_map[..., :2]
    #     ret['vel_map'] = vel_map
    #     return ret

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    rays_t, viewdirs = None, None
    if has_t:
        rays_t = ray_batch[:, -1:]  # [N_rays, 1]
        viewdirs = ray_batch[:, -4:-1] if ray_batch.shape[-1] > 9 else None
    elif ray_batch.shape[-1] > 8:
        viewdirs = ray_batch[:, -3:]

    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.0:
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(42)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    if rays_t is not None:
        rays_t_bc = torch.reshape(rays_t, [-1, 1, 1]).expand([N_rays, N_samples, 1])
        pts = torch.cat([pts, rays_t_bc], dim=-1)

    def warp_raw_random(orig_pts, orig_dir, fading, fn, mod="rand", has_t=has_t):
        # mod, "rand", "forw", "back", "none"
        if (not has_t) or (mod == "none") or (vel_model is None):
            orig_raw = network_query_fn(orig_pts, orig_dir, fn)  # [N_rays, N_samples, 4]
            return orig_raw

        orig_pos, orig_t = torch.split(orig_pts, [3, 1], -1)

        _vel = batchify(vel_model, netchunk)(orig_pts.view(-1, 4))
        _vel = torch.reshape(_vel, [N_rays, -1, 3])
        # _vel.shape, [N_rays, N_samples(+N_importance), 3]
        if mod == "rand":
            # random_warpT = np.random.normal(0.0, 0.6, orig_t.get_shape().as_list())
            # random_warpT = np.random.uniform(-3.0, 3.0, orig_t.shape)
            random_warpT = torch.rand(orig_t.shape) * 6.0 - 3.0  # [-3,3]
        else:
            random_warpT = 1.0 if mod == "back" else (-1.0)  # back
        # mean and standard deviation: 0.0, 0.6, so that 3sigma < 2, train +/- 2*delta_T
        random_warpT = random_warpT * fading
        random_warpT = torch.Tensor(random_warpT)

        warp_t = orig_t + random_warpT
        warp_pos = orig_pos + _vel * random_warpT
        warp_pts = torch.cat([warp_pos, warp_t], dim=-1)
        warp_pts = warp_pts.detach()  # stop gradiant

        warped_raw = network_query_fn(warp_pts, orig_dir, fn)  # [N_rays, N_samples, 4]

        return warped_raw

    def get_raw(fn, staticpts, staticdirs, has_t=has_t):
        static_raw, smoke_raw = None, None
        smoke_warp_mod = warp_mod
        if (None in [vel_model, warp_fading_dt]) or (not has_t):
            smoke_warp_mod = "none"

        smoke_raw = warp_raw_random(staticpts, staticdirs, warp_fading_dt, fn, mod=smoke_warp_mod, has_t=has_t)
        if has_t and (smoke_raw.shape[-1] > 4):  # hybrid mode
            if smoke_warp_mod == "none":
                static_raw = smoke_raw
            else:
                static_raw = warp_raw_random(staticpts, staticdirs, warp_fading_dt, fn, mod="none", has_t=True)

            static_raw = static_raw[..., :4]
            smoke_raw = smoke_raw[..., -4:]

        return smoke_raw, static_raw  # [N_rays, N_samples, 4], [N_rays, N_samples, 4]

    # raw = run_network(pts)
    C_smokeRaw, C_staticRaw = get_raw(network_fn, pts, viewdirs)
    raw = [C_smokeRaw, C_staticRaw]
    rgb_map, disp_map, acc_map, weights, depth_map, ti_map, rgb_map_stack, acc_map_stack = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, pytest=pytest, remove99=remove99
    )

    if raw[-1] is not None:
        rgbh2_map = rgb_map_stack[..., 0]  # dynamic
        acch2_map = acc_map_stack[..., 0]  # dynamic
        rgbh1_map = rgb_map_stack[..., 1]  # staitc
        acch1_map = acc_map_stack[..., 1]  # staitc

    # raw = network_query_fn(pts, viewdirs, network_fn)
    # rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.0), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples + N_importance, 3]

        if rays_t is not None:
            rays_t_bc = torch.reshape(rays_t, [-1, 1, 1]).expand([N_rays, N_samples + N_importance, 1])
            pts = torch.cat([pts, rays_t_bc], dim=-1)

        run_fn = network_fn if network_fine is None else network_fine
        F_smokeRaw, F_staticRaw = get_raw(run_fn, pts, viewdirs)
        raw = [F_smokeRaw, F_staticRaw]

        rgb_map, disp_map, acc_map, weights, depth_map, ti_map, rgb_map_stack, acc_map_stack = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, pytest=pytest, remove99=remove99
        )

        if raw[-1] is not None:
            rgbh20_map = rgbh2_map
            acch20_map = acch2_map
            rgbh10_map = rgbh1_map
            acch10_map = acch1_map
            rgbh2_map = rgb_map_stack[..., 0]
            acch2_map = acc_map_stack[..., 0]
            rgbh1_map = rgb_map_stack[..., 1]
            acch1_map = acc_map_stack[..., 1]

        # raw = run_network(pts, fn=run_fn)
        # raw = network_query_fn(pts, viewdirs, run_fn)
        # rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    # pdb.set_trace()
    ret["rgb_map"] = rgb_map
    ret["disp_map"] = disp_map
    ret["acc_map"] = acc_map
    if retraw:
        ret["raw"] = raw[0]
        if raw[1] is not None:
            ret["raw_static"] = raw[1]
    if N_importance > 0:
        ret["rgb0"] = rgb_map_0
        ret["disp0"] = disp_map_0
        ret["acc0"] = acc_map_0
        ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    if raw[-1] is not None:
        ret["rgbh1"] = rgbh1_map
        ret["acch1"] = acch1_map
        ret["rgbh2"] = rgbh2_map
        ret["acch2"] = acch2_map
        if N_importance > 0:
            ret["rgbh10"] = rgbh10_map
            ret["acch10"] = acch10_map
            ret["rgbh20"] = rgbh20_map
            ret["acch20"] = acch20_map
        ret["rgbM"] = rgbh1_map * 0.5 + rgbh2_map * 0.5

    # for k in ret:
    #     if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
    #         print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


#####################################################################
# custom Logger to write Log to file
class Logger(object):
    def __init__(self, summary_dir, silent=False, fname="logfile.txt"):
        self.terminal = sys.stdout
        self.silent = silent
        self.log = open(os.path.join(summary_dir, fname), "a")
        cmdline = " ".join(sys.argv) + "\n"
        self.log.write(cmdline)

    def write(self, message):
        if not self.silent:
            self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def printENV():
    pass


#####################################################################
# Visualization Tools


def velLegendHSV(hsvin, is3D, lw=-1, constV=255):
    # hsvin: (b), h, w, 3
    # always overwrite hsvin borders [lw], please pad hsvin before hand
    # or fill whole hsvin (lw < 0)
    ih, iw = hsvin.shape[-3:-1]
    if lw <= 0:  # fill whole
        a_list, b_list = [range(ih)], [range(iw)]
    else:  # fill border
        a_list = [range(ih), range(lw), range(ih), range(ih - lw, ih)]
        b_list = [range(lw), range(iw), range(iw - lw, iw), range(iw)]
    for a, b in zip(a_list, b_list):
        for _fty in a:
            for _ftx in b:
                fty = _fty - ih // 2
                ftx = _ftx - iw // 2
                ftang = np.arctan2(fty, ftx) + np.pi
                ftang = ftang * (180 / np.pi / 2)
                # print("ftang,min,max,mean", ftang.min(), ftang.max(), ftang.mean())
                # ftang,min,max,mean 0.7031249999999849 180.0 90.3515625
                hsvin[..., _fty, _ftx, 0] = np.expand_dims(ftang, axis=-1)  # 0-360
                # hsvin[...,_fty,_ftx,0] = ftang
                hsvin[..., _fty, _ftx, 2] = constV
                if (not is3D) or (lw == 1):
                    hsvin[..., _fty, _ftx, 1] = 255
                else:
                    thetaY1 = 1.0 - ((ih // 2) - abs(fty)) / float(lw if (lw > 1) else (ih // 2))
                    thetaY2 = 1.0 - ((iw // 2) - abs(ftx)) / float(lw if (lw > 1) else (iw // 2))
                    fthetaY = max(thetaY1, thetaY2) * (0.5 * np.pi)
                    ftxY, ftyY = np.cos(fthetaY), np.sin(fthetaY)
                    fangY = np.arctan2(ftyY, ftxY)
                    fangY = fangY * (240 / np.pi * 2)  # 240 - 0
                    hsvin[..., _fty, _ftx, 1] = 255 - fangY
                    # print("fangY,min,max,mean", fangY.min(), fangY.max(), fangY.mean())
    # finished velLegendHSV.


def cubecenter(cube, axis, half=0):
    # cube: (b,)h,h,h,c
    # axis: 1 (z), 2 (y), 3 (x)
    reduce_axis = [a for a in [1, 2, 3] if a != axis]
    pack = np.mean(cube, axis=tuple(reduce_axis))  # (b,)h,c
    pack = np.sqrt(np.sum(np.square(pack), axis=-1) + 1e-6)  # (b,)h

    length = cube.shape[axis - 5]  # h
    weights = np.arange(0.5 / length, 1.0, 1.0 / length)
    if half == 1:  # first half
        weights = np.where(weights < 0.5, weights, np.zeros_like(weights))
        pack = np.where(weights < 0.5, pack, np.zeros_like(pack))
    elif half == 2:  # second half
        weights = np.where(weights > 0.5, weights, np.zeros_like(weights))
        pack = np.where(weights > 0.5, pack, np.zeros_like(pack))

    weighted = pack * weights  # (b,)h
    weiAxis = np.sum(weighted, axis=-1) / np.sum(pack, axis=-1) * length  # (b,)

    return weiAxis.astype(np.int32)  # a ceiling is included


def vel2hsv(velin, is3D, logv, scale=None):  # 2D
    fx, fy = velin[..., 0], velin[..., 1]
    ori_shape = list(velin.shape[:-1]) + [3]
    if is3D:
        fz = velin[..., 2]
        ang = np.arctan2(fz, fx) + np.pi  # angXZ
        zxlen2 = fx * fx + fz * fz
        angY = np.arctan2(np.abs(fy), np.sqrt(zxlen2))
        v = np.sqrt(zxlen2 + fy * fy)
    else:
        v = np.sqrt(fx * fx + fy * fy)
        ang = np.arctan2(fy, fx) + np.pi

    if logv:
        v = np.log10(v + 1)

    hsv = np.zeros(ori_shape, np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    if is3D:
        hsv[..., 1] = 255 - angY * (240 / np.pi * 2)
    else:
        hsv[..., 1] = 255
    if scale is not None:
        hsv[..., 2] = np.minimum(v * scale, 255)
    else:
        hsv[..., 2] = v / max(v.max(), 1e-6) * 255.0
    return hsv


def vel_uv2hsv(vel, scale=160, is3D=False, logv=False, mix=False):
    # vel: a np.float32 array, in shape of (?=b,) d,h,w,3 for 3D and (?=b,)h,w, 2 or 3 for 2D
    # scale: scale content to 0~255, something between 100-255 is usually good.
    #        content will be normalized if scale is None
    # logv: visualize value with log
    # mix: use more slices to get a volumetric visualization if True, which is slow

    ori_shape = list(vel.shape[:-1]) + [3]  # (?=b,) d,h,w,3
    if is3D:
        new_range = list(range(len(ori_shape)))
        z_new_range = new_range[:]
        z_new_range[-4] = new_range[-3]
        z_new_range[-3] = new_range[-4]
        # print(z_new_range)
        YZXvel = np.transpose(vel, z_new_range)

        _xm, _ym, _zm = (ori_shape[-2] - 1) // 2, (ori_shape[-3] - 1) // 2, (ori_shape[-4] - 1) // 2

        if mix:
            _xlist = [cubecenter(vel, 3, 1), _xm, cubecenter(vel, 3, 2)]
            _ylist = [cubecenter(vel, 2, 1), _ym, cubecenter(vel, 2, 2)]
            _zlist = [cubecenter(vel, 1, 1), _zm, cubecenter(vel, 1, 2)]
        else:
            _xlist, _ylist, _zlist = [_xm], [_ym], [_zm]

        hsv = []
        for _x, _y, _z in zip(_xlist, _ylist, _zlist):
            # print(_x, _y, _z)
            _x, _y, _z = np.clip([_x, _y, _z], 0, ori_shape[-2:-5:-1])
            _yz = YZXvel[..., _x, :]
            _yz = np.stack([_yz[..., 2], _yz[..., 0], _yz[..., 1]], axis=-1)
            _yx = YZXvel[..., _z, :, :]
            _yx = np.stack([_yx[..., 0], _yx[..., 2], _yx[..., 1]], axis=-1)
            _zx = YZXvel[..., _y, :, :, :]
            _zx = np.stack([_zx[..., 0], _zx[..., 1], _zx[..., 2]], axis=-1)
            # print(_yx.shape, _yz.shape, _zx.shape)

            # in case resolution is not a cube, (res,res,res)
            _yxz = np.concatenate([_yx, _yz], axis=-2)  # yz, yx, zx  # (?=b,),h,w+zdim,3

            if ori_shape[-3] < ori_shape[-4]:
                pad_shape = list(_yxz.shape)  # (?=b,),h,w+zdim,3
                pad_shape[-3] = ori_shape[-4] - ori_shape[-3]
                _pad = np.zeros(pad_shape, dtype=np.float32)
                _yxz = np.concatenate([_yxz, _pad], axis=-3)
            elif ori_shape[-3] > ori_shape[-4]:
                pad_shape = list(_zx.shape)  # (?=b,),h,w+zdim,3
                pad_shape[-3] = ori_shape[-3] - ori_shape[-4]

                _zx = np.concatenate([_zx, np.zeros(pad_shape, dtype=np.float32)], axis=-3)

            midVel = np.concatenate([_yxz, _zx], axis=-2)  # yz, yx, zx  # (?=b,),h,w*3,3
            hsv += [vel2hsv(midVel, True, logv, scale)]
        # remove depth dim, increase with zyx slices
        ori_shape[-3] = 3 * ori_shape[-2]
        ori_shape[-2] = ori_shape[-1]
        ori_shape = ori_shape[:-1]
    else:
        hsv = [vel2hsv(vel, False, logv, scale)]

    bgr = []
    for _hsv in hsv:
        if len(ori_shape) > 3:
            _hsv = _hsv.reshape([-1] + ori_shape[-2:])
        if is3D:
            velLegendHSV(_hsv, is3D, lw=max(1, min(6, int(0.025 * ori_shape[-2]))), constV=255)
        _hsv = cv.cvtColor(_hsv, cv.COLOR_HSV2BGR)
        if len(ori_shape) > 3:
            _hsv = _hsv.reshape(ori_shape)
        bgr += [_hsv]
    if len(bgr) == 1:
        bgr = bgr[0]
    else:
        bgr = bgr[0] * 0.2 + bgr[1] * 0.6 + bgr[2] * 0.2
    return bgr.astype(np.uint8)[::-1]  # flip Y


def den_scalar2rgb(den, scale=160, is3D=False, logv=False, mix=True):
    # den: a np.float32 array, in shape of (?=b,) d,h,w,1 for 3D and (?=b,)h,w,1 for 2D
    # scale: scale content to 0~255, something between 100-255 is usually good.
    #        content will be normalized if scale is None
    # logv: visualize value with log
    # mix: use averaged value as a volumetric visualization if True, else show middle slice

    ori_shape = list(den.shape)
    if ori_shape[-1] != 1:
        ori_shape.append(1)
        den = np.reshape(den, ori_shape)

    if is3D:
        new_range = list(range(len(ori_shape)))
        z_new_range = new_range[:]
        z_new_range[-4] = new_range[-3]
        z_new_range[-3] = new_range[-4]
        # print(z_new_range)
        YZXden = np.transpose(den, z_new_range)

        if not mix:
            _yz = YZXden[..., (ori_shape[-2] - 1) // 2, :]
            _yx = YZXden[..., (ori_shape[-4] - 1) // 2, :, :]
            _zx = YZXden[..., (ori_shape[-3] - 1) // 2, :, :, :]
        else:
            _yz = np.average(YZXden, axis=-2)
            _yx = np.average(YZXden, axis=-3)
            _zx = np.average(YZXden, axis=-4)
            # print(_yx.shape, _yz.shape, _zx.shape)

        # in case resolution is not a cube, (res,res,res)
        _yxz = np.concatenate([_yx, _yz], axis=-2)  # yz, yx, zx  # (?=b,),h,w+zdim,1

        if ori_shape[-3] < ori_shape[-4]:
            pad_shape = list(_yxz.shape)  # (?=b,),h,w+zdim,1
            pad_shape[-3] = ori_shape[-4] - ori_shape[-3]
            _pad = np.zeros(pad_shape, dtype=np.float32)
            _yxz = np.concatenate([_yxz, _pad], axis=-3)
        elif ori_shape[-3] > ori_shape[-4]:
            pad_shape = list(_zx.shape)  # (?=b,),h,w+zdim,1
            pad_shape[-3] = ori_shape[-3] - ori_shape[-4]

            _zx = np.concatenate([_zx, np.zeros(pad_shape, dtype=np.float32)], axis=-3)

        midDen = np.concatenate([_yxz, _zx], axis=-2)  # yz, yx, zx  # (?=b,),h,w*3,1
    else:
        midDen = den

    if logv:
        midDen = np.log10(midDen + 1)
    if scale is None:
        midDen = midDen / max(midDen.max(), 1e-6) * 255.0
    else:
        midDen = midDen * scale
    grey = np.clip(midDen, 0, 255)

    return grey.astype(np.uint8)[::-1]  # flip y


#####################################################################
# Physics Tools


def jacobian3D(x):
    # x, (b,)d,h,w,ch, pytorch tensor
    # return jacobian and curl

    dudx = x[:, :, :, 1:, 0] - x[:, :, :, :-1, 0]
    dvdx = x[:, :, :, 1:, 1] - x[:, :, :, :-1, 1]
    dwdx = x[:, :, :, 1:, 2] - x[:, :, :, :-1, 2]
    dudy = x[:, :, 1:, :, 0] - x[:, :, :-1, :, 0]
    dvdy = x[:, :, 1:, :, 1] - x[:, :, :-1, :, 1]
    dwdy = x[:, :, 1:, :, 2] - x[:, :, :-1, :, 2]
    dudz = x[:, 1:, :, :, 0] - x[:, :-1, :, :, 0]
    dvdz = x[:, 1:, :, :, 1] - x[:, :-1, :, :, 1]
    dwdz = x[:, 1:, :, :, 2] - x[:, :-1, :, :, 2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = torch.cat((dudx, torch.unsqueeze(dudx[:, :, :, -1], 3)), 3)
    dvdx = torch.cat((dvdx, torch.unsqueeze(dvdx[:, :, :, -1], 3)), 3)
    dwdx = torch.cat((dwdx, torch.unsqueeze(dwdx[:, :, :, -1], 3)), 3)

    dudy = torch.cat((dudy, torch.unsqueeze(dudy[:, :, -1, :], 2)), 2)
    dvdy = torch.cat((dvdy, torch.unsqueeze(dvdy[:, :, -1, :], 2)), 2)
    dwdy = torch.cat((dwdy, torch.unsqueeze(dwdy[:, :, -1, :], 2)), 2)

    dudz = torch.cat((dudz, torch.unsqueeze(dudz[:, -1, :, :], 1)), 1)
    dvdz = torch.cat((dvdz, torch.unsqueeze(dvdz[:, -1, :, :], 1)), 1)
    dwdz = torch.cat((dwdz, torch.unsqueeze(dwdz[:, -1, :, :], 1)), 1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy

    j = torch.stack([dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz], -1)
    c = torch.stack([u, v, w], -1)

    return j, c


def curl2D(x, data_format="NHWC"):
    assert data_format == "NHWC"
    u = x[:, 1:, :, 0] - x[:, :-1, :, 0]  # ds/dy
    v = x[:, :, :-1, 0] - x[:, :, 1:, 0]  # -ds/dx,
    u = torch.cat([u, u[:, -1:, :]], dim=1)
    v = torch.cat([v, v[:, :, -1:]], dim=2)
    c = tf.stack([u, v], dim=-1)  # type: ignore
    return c


def curl3D(x, data_format="NHWC"):
    assert data_format == "NHWC"
    # x: bzyxc
    # dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx = x[:, :, :, 1:, 1] - x[:, :, :, :-1, 1]  #
    dwdx = x[:, :, :, 1:, 2] - x[:, :, :, :-1, 2]  #
    dudy = x[:, :, 1:, :, 0] - x[:, :, :-1, :, 0]  #
    # dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy = x[:, :, 1:, :, 2] - x[:, :, :-1, :, 2]  #
    dudz = x[:, 1:, :, :, 0] - x[:, :-1, :, :, 0]  #
    dvdz = x[:, 1:, :, :, 1] - x[:, :-1, :, :, 1]  #
    # dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # dudx = torch.cat((dudx, dudx[:,:,:,-1]), dim=3)
    dvdx = torch.cat((dvdx, dvdx[:, :, :, -1:]), dim=3)  #
    dwdx = torch.cat((dwdx, dwdx[:, :, :, -1:]), dim=3)  #

    dudy = torch.cat((dudy, dudy[:, :, -1:, :]), dim=2)  #
    # dvdy = torch.cat((dvdy, dvdy[:,:,-1:,:]), dim=2)
    dwdy = torch.cat((dwdy, dwdy[:, :, -1:, :]), dim=2)  #

    dudz = torch.cat((dudz, dudz[:, -1:, :, :]), dim=1)  #
    dvdz = torch.cat((dvdz, dvdz[:, -1:, :, :]), dim=1)  #
    # dwdz = torch.cat((dwdz, dwdz[:,-1:,:,:]), dim=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy

    # j = tf.stack([
    #       dudx,dudy,dudz,
    #       dvdx,dvdy,dvdz,
    #       dwdx,dwdy,dwdz
    # ], dim=-1)
    # curl = dwdy-dvdz,dudz-dwdx,dvdx-dudy
    c = torch.stack([u, v, w], dim=-1)

    return c


def jacobian3D_np(x):
    # x, (b,)d,h,w,ch
    # return jacobian and curl

    if len(x.shape) < 5:
        x = np.expand_dims(x, axis=0)
    dudx = x[:, :, :, 1:, 0] - x[:, :, :, :-1, 0]
    dvdx = x[:, :, :, 1:, 1] - x[:, :, :, :-1, 1]
    dwdx = x[:, :, :, 1:, 2] - x[:, :, :, :-1, 2]
    dudy = x[:, :, 1:, :, 0] - x[:, :, :-1, :, 0]
    dvdy = x[:, :, 1:, :, 1] - x[:, :, :-1, :, 1]
    dwdy = x[:, :, 1:, :, 2] - x[:, :, :-1, :, 2]
    dudz = x[:, 1:, :, :, 0] - x[:, :-1, :, :, 0]
    dvdz = x[:, 1:, :, :, 1] - x[:, :-1, :, :, 1]
    dwdz = x[:, 1:, :, :, 2] - x[:, :-1, :, :, 2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = np.concatenate((dudx, np.expand_dims(dudx[:, :, :, -1], axis=3)), axis=3)
    dvdx = np.concatenate((dvdx, np.expand_dims(dvdx[:, :, :, -1], axis=3)), axis=3)
    dwdx = np.concatenate((dwdx, np.expand_dims(dwdx[:, :, :, -1], axis=3)), axis=3)

    dudy = np.concatenate((dudy, np.expand_dims(dudy[:, :, -1, :], axis=2)), axis=2)
    dvdy = np.concatenate((dvdy, np.expand_dims(dvdy[:, :, -1, :], axis=2)), axis=2)
    dwdy = np.concatenate((dwdy, np.expand_dims(dwdy[:, :, -1, :], axis=2)), axis=2)

    dudz = np.concatenate((dudz, np.expand_dims(dudz[:, -1, :, :], axis=1)), axis=1)
    dvdz = np.concatenate((dvdz, np.expand_dims(dvdz[:, -1, :, :], axis=1)), axis=1)
    dwdz = np.concatenate((dwdz, np.expand_dims(dwdz[:, -1, :, :], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy

    j = np.stack([dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz], axis=-1)
    c = np.stack([u, v, w], axis=-1)

    return j, c


# from FFJORD github code
def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac


# from FFJORD github code
def divergence_exact(input_points, outputs):
    # requires three backward passes instead one like divergence_approx
    jac = _get_minibatch_jacobian(outputs, input_points)
    diagonal = jac.view(jac.shape[0], -1)[:, :: (jac.shape[1] + 1)]
    return torch.sum(diagonal, 1)


def PDE_EQs(D_t, D_x, D_y, D_z, U, U_t=None, U_x=None, U_y=None, U_z=None):
    eqs = []
    dts = [D_t]
    dxs = [D_x]
    dys = [D_y]
    dzs = [D_z]

    if None not in [U_t, U_x, U_y, U_z]:
        dts += U_t.split(1, dim=-1)  # [d_t, u_t, v_t, w_t] # (N,1)
        dxs += U_x.split(1, dim=-1)  # [d_x, u_x, v_x, w_x]
        dys += U_y.split(1, dim=-1)  # [d_y, u_y, v_y, w_y]
        dzs += U_z.split(1, dim=-1)  # [d_z, u_z, v_z, w_z]

    u, v, w = U.split(1, dim=-1)  # (N,1)
    for dt, dx, dy, dz in zip(dts, dxs, dys, dzs):
        _e = dt + (u * dx + v * dy + w * dz)
        eqs += [_e]
    # transport and nse equations:
    # e1 = d_t + (u*d_x + v*d_y + w*d_z) - PecInv*(c_xx + c_yy + c_zz)          , should = 0
    # e2 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - ReyInv*(u_xx + u_yy + u_zz)    , should = 0
    # e3 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - ReyInv*(v_xx + v_yy + v_zz)    , should = 0
    # e4 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - ReyInv*(w_xx + w_yy + w_zz)    , should = 0
    # e5 = u_x + v_y + w_z                                                      , should = 0
    # For simplification, we assume PecInv = 0.0, ReyInv = 0.0, pressure p = (0,0,0)

    if None not in [U_t, U_x, U_y, U_z]:
        # eqs += [ u_x + v_y + w_z ]
        eqs += [dxs[1] + dys[2] + dzs[3]]

    if True:  # scale regularization
        eqs += [(u * u + v * v + w * w) * 1e-1]

    return eqs


#####################################################################
# Coord Tools (all for torch Tensors)
# Coords:
# 1. resolution space, Frames x Depth x H x W, coord (frame_t, voxel_z, voxel_y, voxel_x),
# 2. simulation space, scale the resolution space to around 0-1,
#    (FrameLength and Width in [0-1], Height and Depth keep ratios wrt Width)
# 3. target space,
# 4. world space,
# 5. camera spaces,

# Vworld, Pworld; velocity, position in 4. world coord.
# Vsmoke, Psmoke; velocity, position in 2. simulation coord.
# w2s, 4.world to 3.target matrix (vel transfer uses rotation only; pos transfer includes offsets)
# s2w, 3.target to 4.world matrix (vel transfer uses rotation only; pos transfer includes offsets)
# scale_vector, to scale from 2.simulation space to 3.target space (no rotation, no offset)
#        for synthetic data, scale_vector = openvdb voxel size * [W,H,D] grid resolution (x first, z last),
#        for e.g., scale_vector = 0.0469 * 256 = 12.0064
# st_factor, spatial temporal resolution ratio, to scale velocity from 2.simulation unit to 1.resolution unit
#        for e.g.,  st_factor = [W/float(max_timestep),H/float(max_timestep),D/float(max_timestep)]


# functions to transfer between 4. world space and 2. simulation space,
# velocity are further scaled according to resolution as in mantaflow
def vel_world2smoke(Vworld, w2s, scale_vector, st_factor):
    _st_factor = torch.Tensor(st_factor).expand((3,))
    vel_rot = Vworld[..., None, :] * (w2s[:3, :3])
    vel_rot = torch.sum(vel_rot, -1)  # 4.world to 3.target
    vel_scale = vel_rot / (scale_vector) * _st_factor  # 3.target to 2.simulation
    return vel_scale


def vel_smoke2world(Vsmoke, s2w, scale_vector, st_factor):
    _st_factor = torch.Tensor(st_factor).expand((3,))
    vel_scale = Vsmoke * (scale_vector) / _st_factor  # 2.simulation to 3.target
    vel_rot = torch.sum(vel_scale[..., None, :] * (s2w[:3, :3]), -1)  # 3.target to 4.world
    return vel_rot


def pos_world2smoke(Pworld, w2s, scale_vector):
    pos_rot = torch.sum(Pworld[..., None, :] * (w2s[:3, :3]), -1)  # 4.world to 3.target
    pos_off = (w2s[:3, -1]).expand(pos_rot.shape)  # 4.world to 3.target
    new_pose = pos_rot + pos_off
    pos_scale = new_pose / (scale_vector)  # 3.target to 2.simulation
    return pos_scale


def off_smoke2world(Offsmoke, s2w, scale_vector):
    off_scale = Offsmoke * (scale_vector)  # 2.simulation to 3.target
    off_rot = torch.sum(off_scale[..., None, :] * (s2w[:3, :3]), -1)  # 3.target to 4.world
    return off_rot


def pos_smoke2world(Psmoke, s2w, scale_vector):
    pos_scale = Psmoke * (scale_vector)  # 2.simulation to 3.target
    pos_rot = torch.sum(pos_scale[..., None, :] * (s2w[:3, :3]), -1)  # 3.target to 4.world
    pos_off = (s2w[:3, -1]).expand(pos_rot.shape)  # 3.target to 4.world
    return pos_rot + pos_off


def get_voxel_pts(H, W, D, s2w, scale_vector, n_jitter=0, r_jitter=0.8):
    """Get voxel positions."""

    i, j, k = torch.meshgrid(torch.linspace(0, D - 1, D), torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))
    pts = torch.stack([(k + 0.5) / W, (j + 0.5) / H, (i + 0.5) / D], -1)
    # shape D*H*W*3, value [(x,y,z)] , range [0,1]

    jitter_r = torch.Tensor([r_jitter / W, r_jitter / H, r_jitter / D]).float().expand(pts.shape)
    for i_jitter in range(n_jitter):
        off_i = torch.rand(pts.shape, dtype=torch.float) - 0.5
        # shape D*H*W*3, value [(x,y,z)] , range [-0.5,0.5]

        pts = pts + off_i * jitter_r

    return pos_smoke2world(pts, s2w, scale_vector)


def get_voxel_pts_offset(H, W, D, s2w, scale_vector, r_offset=0.8):
    """Get voxel positions."""

    i, j, k = torch.meshgrid(torch.linspace(0, D - 1, D), torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))
    pts = torch.stack([(k + 0.5) / W, (j + 0.5) / H, (i + 0.5) / D], -1)
    # shape D*H*W*3, value [(x,y,z)] , range [0,1]

    jitter_r = torch.Tensor([r_offset / W, r_offset / H, r_offset / D]).expand(pts.shape)
    off_i = torch.rand([1, 1, 1, 3], dtype=torch.float) - 0.5
    # shape 1*1*1*3, value [(x,y,z)] , range [-0.5,0.5]
    pts = pts + off_i * jitter_r

    return pos_smoke2world(pts, s2w, scale_vector)


class BBox_Tool(object):
    def __init__(self, smoke_tran_inv, smoke_scale, in_min=0.0, in_max=1.0):
        self.s_w2s = torch.Tensor(smoke_tran_inv).expand([4, 4])
        self.s_scale = torch.Tensor(smoke_scale).expand([3])
        self.s_min = torch.Tensor(in_min).expand([3])
        self.s_max = torch.Tensor(in_max).expand([3])
        self.s2w = torch.inverse(self.s_w2s)

    def setMinMax(self, in_min=0.0, in_max=1.0):
        self.s_min = torch.Tensor(in_min).expand([3])
        self.s_max = torch.Tensor(in_max).expand([3])

    def world2sim(self, pts_world):
        pts_world_homo = torch.cat([pts_world, torch.ones_like(pts_world[..., :1])], dim=-1)
        pts_sim_ = torch.matmul(self.s_w2s, pts_world_homo[..., None]).squeeze(-1)[..., :3]
        pts_sim = pts_sim_ / (self.s_scale)  # 3.target to 2.simulation
        return pts_sim

    def world2sim_rot(self, pts_world):
        pts_sim_ = torch.matmul(self.s_w2s[:3, :3], pts_world[..., None]).squeeze(-1)
        pts_sim = pts_sim_ / (self.s_scale)  # 3.target to 2.simulation
        return pts_sim

    def sim2world(self, pts_sim):
        pts_sim_ = pts_sim * self.s_scale
        pts_sim_homo = torch.cat([pts_sim_, torch.ones_like(pts_sim_[..., :1])], dim=-1)
        pts_world = torch.matmul(self.s2w, pts_sim_homo[..., None]).squeeze(-1)[..., :3]
        return pts_world

    def isInside(self, inputs_pts):
        target_pts = pos_world2smoke(inputs_pts, self.s_w2s, self.s_scale)
        above = torch.logical_and(target_pts[..., 0] >= self.s_min[0], target_pts[..., 1] >= self.s_min[1])
        above = torch.logical_and(above, target_pts[..., 2] >= self.s_min[2])
        below = torch.logical_and(target_pts[..., 0] <= self.s_max[0], target_pts[..., 1] <= self.s_max[1])
        below = torch.logical_and(below, target_pts[..., 2] <= self.s_max[2])
        outputs = torch.logical_and(below, above)
        return outputs

    def insideMask(self, inputs_pts, to_float=True):
        return self.isInside(inputs_pts).to(torch.float) if to_float else self.isInside(inputs_pts)


class Voxel_Tool(object):

    def __get_tri_slice(self, _xm, _ym, _zm, _n=1):
        _yz = torch.reshape(self.pts[..., _xm : _xm + _n, :], (-1, 3))
        _zx = torch.reshape(self.pts[:, _ym : _ym + _n, ...], (-1, 3))
        _xy = torch.reshape(self.pts[_zm : _zm + _n, ...], (-1, 3))

        pts_mid = torch.cat([_yz, _zx, _xy], dim=0)
        npMaskXYZ = [np.zeros([self.D, self.H, self.W, 1], dtype=np.float32) for _ in range(3)]
        npMaskXYZ[0][..., _xm : _xm + _n, :] = 1.0
        npMaskXYZ[1][:, _ym : _ym + _n, ...] = 1.0
        npMaskXYZ[2][_zm : _zm + _n, ...] = 1.0
        return pts_mid, torch.tensor(np.clip(npMaskXYZ[0] + npMaskXYZ[1] + npMaskXYZ[2], 1e-6, 3.0))

    def __pad_slice_to_volume(self, _slice, _n, mode=0):
        # mode: 0, x_slice, 1, y_slice, 2, z_slice
        tar_shape = [self.D, self.H, self.W]
        in_shape = tar_shape[:]
        in_shape[-1 - mode] = _n
        fron_shape = tar_shape[:]
        fron_shape[-1 - mode] = (tar_shape[-1 - mode] - _n) // 2
        back_shape = tar_shape[:]
        back_shape[-1 - mode] = tar_shape[-1 - mode] - _n - fron_shape[-1 - mode]

        cur_slice = _slice.view(in_shape + [-1])
        front_0 = torch.zeros(fron_shape + [cur_slice.shape[-1]])
        back_0 = torch.zeros(back_shape + [cur_slice.shape[-1]])

        volume = torch.cat([front_0, cur_slice, back_0], dim=-2 - mode)
        return volume

    def __init__(self, smoke_tran, smoke_tran_inv, smoke_scale, D, H, W, middleView=None):
        self.s_s2w = torch.Tensor(smoke_tran).expand([4, 4])
        self.s_w2s = torch.Tensor(smoke_tran_inv).expand([4, 4])
        self.s_scale = torch.Tensor(smoke_scale).expand([3])
        self.D = D
        self.H = H
        self.W = W
        self.pts = get_voxel_pts(H, W, D, self.s_s2w, self.s_scale)
        self.pts_mid = None
        self.npMaskXYZ = None
        self.middleView = middleView
        if middleView is not None:
            _n = 1 if self.middleView == "mid" else 3
            _xm, _ym, _zm = (W - _n) // 2, (H - _n) // 2, (D - _n) // 2
            self.pts_mid, self.npMaskXYZ = self.__get_tri_slice(_xm, _ym, _zm, _n)

    def get_raw_at_pts(self, cur_pts, chunk=1024 * 32, use_viewdirs=False, network_query_fn=None, network_fn=None):
        input_shape = list(cur_pts.shape[0:-1])

        pts_flat = cur_pts.view(-1, 4)
        pts_N = pts_flat.shape[0]
        # Evaluate model
        all_raw = []
        viewdir_zeros = torch.zeros([chunk, 3], dtype=torch.float) if use_viewdirs else None
        for i in range(0, pts_N, chunk):
            pts_i = pts_flat[i : i + chunk]
            viewdir_i = viewdir_zeros[: pts_i.shape[0]] if use_viewdirs else None

            raw_i = network_query_fn(pts_i, viewdir_i, network_fn)
            all_raw.append(raw_i)

        raw = torch.cat(all_raw, 0).view(input_shape + [-1])
        return raw

    def get_density_flat(
        self, cur_pts, chunk=1024 * 32, use_viewdirs=False, network_query_fn=None, network_fn=None, getStatic=True
    ):
        flat_raw = self.get_raw_at_pts(cur_pts, chunk, use_viewdirs, network_query_fn, network_fn)
        den_raw = F.relu(flat_raw[..., -1:])
        returnStatic = getStatic and (flat_raw.shape[-1] > 4)
        if returnStatic:
            static_raw = F.relu(flat_raw[..., 3:4])
            return [den_raw, static_raw]
        return [den_raw]

    def get_velocity_flat(self, cur_pts, batchify_fn, chunk=1024 * 32, vel_model=None):
        pts_N = cur_pts.shape[0]
        world_v = []
        for i in range(0, pts_N, chunk):
            input_i = cur_pts[i : i + chunk]
            vel_i = batchify_fn(vel_model, chunk)(input_i)
            world_v.append(vel_i)
        world_v = torch.cat(world_v, 0)
        return world_v

    def get_density_and_derivatives(
        self, cur_pts, chunk=1024 * 32, use_viewdirs=False, network_query_fn=None, network_fn=None
    ):
        _den = self.get_density_flat(cur_pts, chunk, use_viewdirs, network_query_fn, network_fn, False)[0]
        # requires 1 backward passes
        # The minibatch Jacobian matrix of shape (N, D_y=1, D_x=4)
        jac = _get_minibatch_jacobian(_den, cur_pts)
        _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]  # (N,1)
        return _den, _d_x, _d_y, _d_z, _d_t

    def get_velocity_and_derivatives(self, cur_pts, chunk=1024 * 32, batchify_fn=None, vel_model=None):
        _vel = self.get_velocity_flat(cur_pts, batchify_fn, chunk, vel_model)
        # requires 3 backward passes
        # The minibatch Jacobian matrix of shape (N, D_y=3, D_x=4)
        jac = _get_minibatch_jacobian(_vel, cur_pts)
        _u_x, _u_y, _u_z, _u_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]  # (N,3)
        return _vel, _u_x, _u_y, _u_z, _u_t

    def get_voxel_density_list(
        self, t=None, chunk=1024 * 32, use_viewdirs=False, network_query_fn=None, network_fn=None, middle_slice=False
    ):
        D, H, W = self.D, self.H, self.W
        # middle_slice, only for fast visualization of the middle slice
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        pts_N = pts_flat.shape[0]
        if t is not None:
            input_t = torch.ones([pts_N, 1]) * float(t)
            pts_flat = torch.cat([pts_flat, input_t], dim=-1)

        den_list = self.get_density_flat(pts_flat, chunk, use_viewdirs, network_query_fn, network_fn)

        return_list = []
        for den_raw in den_list:
            if middle_slice:
                # only for fast visualization of the middle slice
                _n = 1 if self.middleView == "mid" else 3
                _yzV, _zxV, _xyV = torch.split(den_raw, [D * H * _n, D * W * _n, H * W * _n], dim=0)
                mixV = (
                    self.__pad_slice_to_volume(_yzV, _n, 0)
                    + self.__pad_slice_to_volume(_zxV, _n, 1)
                    + self.__pad_slice_to_volume(_xyV, _n, 2)
                )
                return_list.append(mixV / self.npMaskXYZ)
            else:
                return_list.append(den_raw.view(D, H, W, 1))
        return return_list

    def get_voxel_velocity(self, deltaT, t, batchify_fn, chunk=1024 * 32, vel_model=None, middle_slice=False):
        # middle_slice, only for fast visualization of the middle slice
        D, H, W = self.D, self.H, self.W
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        pts_N = pts_flat.shape[0]
        if t is not None:
            input_t = torch.ones([pts_N, 1]) * float(t)
            pts_flat = torch.cat([pts_flat, input_t], dim=-1)

        world_v = self.get_velocity_flat(pts_flat, batchify_fn, chunk, vel_model)
        reso_scale = [self.W * deltaT, self.H * deltaT, self.D * deltaT]
        target_v = vel_world2smoke(world_v, self.s_w2s, self.s_scale, reso_scale)

        if middle_slice:
            _n = 1 if self.middleView == "mid" else 3
            _yzV, _zxV, _xyV = torch.split(target_v, [D * H * _n, D * W * _n, H * W * _n], dim=0)
            mixV = (
                self.__pad_slice_to_volume(_yzV, _n, 0)
                + self.__pad_slice_to_volume(_zxV, _n, 1)
                + self.__pad_slice_to_volume(_xyV, _n, 2)
            )
            target_v = mixV / self.npMaskXYZ
        else:
            target_v = target_v.view(D, H, W, 3)

        return target_v

    def save_voxel_den_npz(
        self,
        den_path,
        t,
        use_viewdirs=False,
        network_query_fn=None,
        network_fn=None,
        chunk=1024 * 32,
        save_npz=True,
        save_jpg=False,
        jpg_mix=True,
        noStatic=False,
    ):
        voxel_den_list = self.get_voxel_density_list(
            t, chunk, use_viewdirs, network_query_fn, network_fn, middle_slice=not (jpg_mix or save_npz)
        )
        head_tail = os.path.split(den_path)
        namepre = ["", "static_"]
        for voxel_den, npre in zip(voxel_den_list, namepre):
            voxel_den = voxel_den.detach().cpu().numpy()
            if save_jpg:
                jpg_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0] + ".jpg")
                imageio.imwrite(jpg_path, den_scalar2rgb(voxel_den, scale=None, is3D=True, logv=False, mix=jpg_mix))
            if save_npz:
                # to save some space
                npz_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0] + ".npz")
                voxel_den = np.float16(voxel_den)
                np.savez_compressed(npz_path, vel=voxel_den)
            if noStatic:
                break

    def save_voxel_vel_npz(
        self,
        vel_path,
        deltaT,
        t,
        batchify_fn,
        chunk=1024 * 32,
        vel_model=None,
        save_npz=True,
        save_jpg=False,
        save_vort=False,
    ):
        vel_scale = 160
        voxel_vel = (
            self.get_voxel_velocity(deltaT, t, batchify_fn, chunk, vel_model, middle_slice=not save_npz)
            .detach()
            .cpu()
            .numpy()
        )

        if save_jpg:
            jpg_path = os.path.splitext(vel_path)[0] + ".jpg"
            imageio.imwrite(jpg_path, vel_uv2hsv(voxel_vel, scale=vel_scale, is3D=True, logv=False))
        if save_npz:
            if save_vort and save_jpg:
                _, NETw = jacobian3D_np(voxel_vel)
                head_tail = os.path.split(vel_path)
                imageio.imwrite(
                    os.path.join(head_tail[0], "vort" + os.path.splitext(head_tail[1])[0] + ".jpg"),
                    vel_uv2hsv(NETw[0], scale=vel_scale * 5.0, is3D=True),
                )
            # to save some space
            voxel_vel = np.float16(voxel_vel)
            np.savez_compressed(vel_path, vel=voxel_vel)


#####################################################################
# Loss Tools (all for torch Tensors)
def fade_in_weight(step, start, duration):
    return min(max((float(step) - start) / duration, 0.0), 1.0)


# Ghost Density Loss Tool
def ghost_loss_func(_rgb, bg, _acc, den_penalty=0.0):
    _bg = bg.detach()
    ghost_mask = torch.mean(torch.square(_rgb - _bg), -1)
    ghost_mask = torch.sigmoid(ghost_mask * -1.0) + den_penalty  # (0 to 0.5) + den_penalty
    ghost_alpha = ghost_mask * _acc
    return torch.mean(torch.square(ghost_alpha))


def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return torch.mean(torch.square(pred - exact))


# VGG Tool, https://github.com/crowsonkb/style-transfer-pytorch/
class VGGFeatures(nn.Module):
    poolings = {"max": nn.MaxPool2d, "average": nn.AvgPool2d}  # , 'l2': partial(nn.LPPool2d, 2)}
    pooling_scales = {"max": 1.0, "average": 2.0, "l2": 0.78}

    def __init__(self, layers, pooling="max"):
        super().__init__()
        self.layers = sorted(set(layers))

        # The PyTorch pre-trained VGG-19 expects sRGB inputs in the range [0, 1] which are then
        # normalized according to this transform, unlike Simonyan et al.'s original model.
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # The PyTorch pre-trained VGG-19 has different parameters from Simonyan et al.'s original
        # model.
        self.model = torchvision.models.vgg19(pretrained=True).features[: self.layers[-1] + 1]
        self.devices = [torch.device("cpu")] * len(self.model)

        # Reduces edge artifacts.
        self.model[0] = self._change_padding_mode(self.model[0], "replicate")

        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(self.model):
            if pooling != "max" and isinstance(layer, nn.MaxPool2d):
                # Changing the pooling type from max results in the scale of activations
                # changing, so rescale them. Gatys et al. (2015) do not do this.
                self.model[i] = Scale(self.poolings[pooling](2), pool_scale)  # type: ignore

        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _change_padding_mode(conv, padding_mode):
        new_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            padding_mode=padding_mode,
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            new_conv.bias.copy_(conv.bias)
        return new_conv

    @staticmethod
    def _get_min_size(layers):
        last_layer = max(layers)
        min_size = 1
        for layer in [4, 9, 18, 27, 36]:
            if last_layer < layer:
                break
            min_size *= 2
        return min_size

    def distribute_layers(self, devices):
        for i, layer in enumerate(self.model):
            if i in devices:
                device = torch.device(devices[i])
            self.model[i] = layer.cuda()
            self.devices[i] = device

    def forward(self, input, layers=None):
        # input shape, b,3,h,w
        layers = self.layers if layers is None else sorted(set(layers))
        h, w = input.shape[2:4]
        min_size = self._get_min_size(layers)
        if min(h, w) < min_size:
            raise ValueError(f"Input is {h}x{w} but must be at least {min_size}x{min_size}")
        feats = {"input": input}
        norm_in = torch.stack([self.normalize(input[_i]) for _i in range(input.shape[0])], dim=0)
        # input = self.normalize(input)
        for i in range(max(layers) + 1):
            norm_in = self.model[i](norm_in.to(self.devices[i]))
            if i in layers:
                feats[i] = norm_in
        return feats


# VGG Loss Tool
class VGGlossTool(object):
    def __init__(self, device="cuda", pooling="max"):
        # The default content and style layers in Gatys et al. (2015):
        #   content_layers = [22], 'relu4_2'
        #   style_layers = [1, 6, 11, 20, 29], relu layers: [ 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        # We use [5, 10, 19, 28], conv layers before relu: [ 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.layer_list = [5, 10, 19, 28]
        self.layer_names = [
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.device = device

        # Build a VGG19 model loaded with pre-trained ImageNet weights
        self.vggmodel = VGGFeatures(self.layer_list, pooling=pooling)
        device_plan = {0: device}
        self.vggmodel.distribute_layers(device_plan)

    def feature_norm(self, feature):
        # feature: b,h,w,c
        feature_len = torch.sqrt(torch.sum(torch.square(feature), dim=-1, keepdim=True) + 1e-12)
        norm = feature / feature_len
        return norm

    def cos_sim(self, a, b):
        cos_sim_ab = torch.sum(a * b, dim=-1)
        # cosine similarity, -1~1, 1 best
        cos_sim_ab_score = 1.0 - torch.mean(cos_sim_ab)  # 0 ~ 2, 0 best
        return cos_sim_ab_score

    def compute_cos_loss(self, img, ref):
        # input img, ref should be in range of [0,1]
        input_tensor = torch.stack([ref, img], dim=0)

        input_tensor = input_tensor.permute((0, 3, 1, 2))
        # print(input_tensor.shape)
        _feats = self.vggmodel(input_tensor, layers=self.layer_list)

        # Initialize the loss
        loss = []
        # Add loss
        for layer_i, layer_name in zip(self.layer_list, self.layer_names):
            cur_feature = _feats[layer_i]
            reference_features = self.feature_norm(cur_feature[0, ...])
            img_features = self.feature_norm(cur_feature[1, ...])

            feature_metric = self.cos_sim(reference_features, img_features)
            loss += [feature_metric]
        return loss


def merge_imgs(save_dir, framerate=30, prefix=""):
    os.system(
        "ffmpeg -hide_banner -loglevel error -y -i {0}/{1}%03d.png -vf palettegen {0}/palette.png".format(
            save_dir, prefix
        )
    )
    os.system(
        "ffmpeg -hide_banner -loglevel error -y -framerate {0} -i {1}/{2}%03d.png -i {1}/palette.png -lavfi paletteuse {1}/_{2}.gif".format(
            framerate, save_dir, prefix
        )
    )
    os.system(
        "ffmpeg -hide_banner -loglevel error -y -framerate {0} -i {1}/{2}%03d.png -i {1}/palette.png -lavfi paletteuse -vcodec prores {1}/_{2}.mov".format(
            framerate, save_dir, prefix
        )
    )


def save_log(basedir, expname):
    # logs
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    # date_str = datetime.datetime.now().strftime("%m%d-%H%M%S")
    filedir = "log_train"  # + ("train" if not (args.vol_output_only or args.render_only) else "test")
    # filedir += date_str
    log_dir = os.path.join(basedir, expname)
    os.makedirs(log_dir, exist_ok=True)

    # sys.stdout = Logger(log_dir, False, fname="log.out")
    # sys.stderr = Logger(log_dir, False, fname="log.err")

    # print(" ".join(sys.argv), flush=True)
    # printENV()

    # files backup
    # shutil.copyfile(args.config, os.path.join(basedir, expname, filedir, "config.txt"))
    # f = os.path.join(log_dir, "args.txt")
    # with open(f, "w") as file:
    #     for arg in sorted(vars(args)):
    #         attr = getattr(args, arg)
    #         file.write("{} = {}\n".format(arg, attr))
    # filelist = ["run_nerf.py", "run_nerf_helpers.py", "run_pinf.py", "run_pinf_helpers.py"]
    # for filename in filelist:
    #     shutil.copyfile("./" + filename, os.path.join(log_dir, filename.replace("/", "_")))

    return filedir


def model_fading_update(models, global_step, tempoDelay, velDelay, isHybrid):
    tempoDelay = tempoDelay if isHybrid else 0
    for _m in models:
        if models[_m] is None:
            continue
        if _m == "vel_model":
            models[_m].update_fading_step(global_step - tempoDelay - velDelay)
        elif isHybrid:
            models[_m].update_fading_step(global_step, global_step - tempoDelay)
        else:
            models[_m].update_fading_step(global_step)


def batchify_query(inputs, query_function, batch_size=2**20):
    """
    args:
        inputs: [..., input_dim]
    return:
        outputs: [..., output_dim]
    """
    input_dim = inputs.shape[-1]
    input_shape = inputs.shape
    inputs = inputs.view(-1, input_dim)  # flatten all but last dim
    N = inputs.shape[0]
    outputs = []
    for i in range(0, N, batch_size):
        output = query_function(inputs[i : i + batch_size])
        if isinstance(output, tuple):
            output = output[0].detach()
        outputs.append(output)
        del output
    outputs = torch.cat(outputs, dim=0)
    return outputs.view(*input_shape[:-1], -1).detach()  # unflatten


def advect_maccormack(q_grid, vel_world_prev, coord_3d_sim, dt, **kwargs):
    """
    Args:
        q_grid: [X', Y', Z', C]
        vel_world_prev: [X, Y, Z, 3]
        coord_3d_sim: [X, Y, Z, 3]
        dt: float
    Returns:
        advected_quantity: [X, Y, Z, C]
        vel_world: [X, Y, Z, 3]
    """
    q_max, q_min = q_grid[..., -1].max(), q_grid[..., -1].min()
    q_grid_next, _ = advect_SL(q_grid, vel_world_prev, coord_3d_sim, dt, **kwargs)
    q_grid_back, vel_world = advect_SL(q_grid_next, vel_world_prev, coord_3d_sim, -dt, **kwargs)
    q_advected = q_grid_next + (q_grid - q_grid_back) / 2
    q_advected[..., -1] = torch.clamp(q_advected[..., -1], q_min, q_max)
    # q_advected[...,0] = torch.clamp(q_advected[...,0],q_grid[...,0].min(), q_grid[...,0].max())
    # q_advected[...,1] = torch.clamp(q_advected[...,1],q_grid[...,1].min(), q_grid[...,1].max())
    # q_advected[...,2] = torch.clamp(q_advected[...,2],q_grid[...,2].min(), q_grid[...,2].max())
    return q_advected, vel_world


def advect_SL(
    q_grid,
    vel_world_prev,
    coord_3d_sim,
    dt,
    RK=2,
    y_start=48,
    proj_y=128,
    use_project=False,
    project_solver=None,
    bbox_model=None,
    **kwargs,
):
    """Advect a scalar quantity using a given velocity field.
    Args:
        q_grid: [X', Y', Z', C]
        vel_world_prev: [X, Y, Z, 3]
        coord_3d_sim: [X, Y, Z, 3]
        dt: float
        RK: int, number of Runge-Kutta steps
        y_start: where to start at y-axis
        proj_y: simulation domain resolution at y-axis
        use_project: whether to use Poisson solver
        project_solver: Poisson solver
        bbox_model: bounding box model
    Returns:
        advected_quantity: [X, Y, Z, 1]
        vel_world: [X, Y, Z, 3]
    """
    # q_max, q_min = q_grid[...,-1].max(), q_grid[...,-1].min()
    if RK == 1:
        vel_world = vel_world_prev.clone()
        vel_world[:, y_start : y_start + proj_y] = (
            project_solver.Poisson(vel_world[:, y_start : y_start + proj_y])
            if use_project
            else vel_world[:, y_start : y_start + proj_y]
        )
        vel_sim = bbox_model.world2sim_rot(vel_world)  # [X, Y, Z, 3]
    elif RK == 2:
        vel_world = vel_world_prev.clone()  # [X, Y, Z, 3]
        vel_world[:, y_start : y_start + proj_y] = (
            project_solver.Poisson(vel_world[:, y_start : y_start + proj_y])
            if use_project
            else vel_world[:, y_start : y_start + proj_y]
        )
        # breakpoint()
        vel_sim = bbox_model.world2sim_rot(vel_world)  # [X, Y, Z, 3]
        coord_3d_sim_midpoint = coord_3d_sim - 0.5 * dt * vel_sim  # midpoint
        midpoint_sampled = coord_3d_sim_midpoint * 2 - 1  # [X, Y, Z, 3]
        vel_sim = (
            F.grid_sample(
                vel_sim.permute(3, 2, 1, 0)[None], midpoint_sampled.permute(2, 1, 0, 3)[None], align_corners=True
            )
            .squeeze(0)
            .permute(3, 2, 1, 0)
        )  # [X, Y, Z, 3]
    else:
        raise NotImplementedError
    backtrace_coord = coord_3d_sim - dt * vel_sim  # [X, Y, Z, 3]
    backtrace_coord_sampled = backtrace_coord * 2 - 1  # ranging [-1, 1]
    q_grid = q_grid[None, ...].permute([0, 4, 3, 2, 1])  # [N, C, Z, Y, X] i.e., [N, C, D, H, W]
    q_backtraced = F.grid_sample(
        q_grid, backtrace_coord_sampled.permute(2, 1, 0, 3)[None, ...], align_corners=False
    )  # [N, C, D, H, W]
    q_backtraced = q_backtraced.squeeze(0).permute([3, 2, 1, 0])  # [X, Y, Z, C]
    # q_backtraced[...,-1] = torch.clamp(q_backtraced[...,-1],q_min, q_max)
    # q_backtraced[...,0] = torch.clamp(q_backtraced[...,0],q_grid[...,0].min(), q_grid[...,0].max())
    # q_backtraced[...,1] = torch.clamp(q_backtraced[...,1],q_grid[...,1].min(), q_grid[...,1].max())
    # q_backtraced[...,2] = torch.clamp(q_backtraced[...,2],q_grid[...,2].min(), q_grid[...,2].max())
    return q_backtraced, vel_world


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, dim=3):
    if i == -1:
        return nn.Identity(), dim

    embed_kwargs = {
        "include_input": True,
        "input_dims": dim,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(
        self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, bbox_model=None
    ):
        """ """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)]
        )

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        self.bbox_model = bbox_model

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(input_pts[:, :3])
            outputs = torch.reshape(bbox_mask, [-1, 1]) * outputs

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


# Model
class SIREN_NeRFt(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=4,
        input_ch_views=3,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
        fading_fin_step=0,
        bbox_model=None,
    ):
        """
        fading_fin_step: >0, to fade in layers one by one, fully faded in when self.fading_step >= fading_fin_step
        """

        super(SIREN_NeRFt, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.fading_step = 0
        self.fading_fin_step = fading_fin_step if fading_fin_step > 0 else 0
        self.bbox_model = bbox_model

        first_omega_0 = 30.0
        hidden_omega_0 = 1.0

        self.pts_linears = nn.ModuleList(
            [SineLayer(input_ch, W, omega_0=first_omega_0)]
            + [
                (
                    SineLayer(W, W, omega_0=hidden_omega_0)
                    if i not in self.skips
                    else SineLayer(W + input_ch, W, omega_0=hidden_omega_0)
                )
                for i in range(D - 1)
            ]
        )

        final_alpha_linear = nn.Linear(W, 1)
        self.alpha_linear = final_alpha_linear

        if use_viewdirs:
            self.views_linear = SineLayer(input_ch_views, W // 2, omega_0=first_omega_0)
            self.feature_linear = SineLayer(W, W // 2, omega_0=hidden_omega_0)
            self.feature_view_linears = nn.ModuleList([SineLayer(W, W, omega_0=hidden_omega_0)])

        final_rgb_linear = nn.Linear(W, 3)
        self.rgb_linear = final_rgb_linear

    def update_fading_step(self, fading_step):
        # should be updated with the global step
        # e.g., update_fading_step(global_step - radiance_in_step)
        if fading_step >= 0:
            self.fading_step = fading_step

    def fading_wei_list(self):
        # try print(fading_wei_list()) for debug
        step_ratio = np.clip(float(self.fading_step) / float(max(1e-8, self.fading_fin_step)), 0, 1)
        ma = 1 + (self.D - 2) * step_ratio  # in range of 1 to self.D-1
        fading_wei_list = [np.clip(1 + ma - m, 0, 1) * np.clip(1 + m - ma, 0, 1) for m in range(self.D)]
        return fading_wei_list

    def print_fading(self):
        # w_list = self.fading_wei_list()
        # _str = ["h%d:%0.03f" % (i, w_list[i]) for i in range(len(w_list)) if w_list[i] > 1e-8]
        # print("; ".join(_str))
        pass

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        h_layers = []
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

            h_layers += [h]
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # a sliding window (fading_wei_list) to enable deeper layers progressively
        if self.fading_fin_step > self.fading_step:
            fading_wei_list = self.fading_wei_list()
            h = 0
            for w, y in zip(fading_wei_list, h_layers):
                if w > 1e-8:
                    h = w * y + h

        alpha = self.alpha_linear(h)

        if self.use_viewdirs:
            input_pts_feature = self.feature_linear(h)
            input_views_feature = self.views_linear(input_views)

            h = torch.cat([input_pts_feature, input_views_feature], -1)

            for i, l in enumerate(self.feature_view_linears):
                h = self.feature_view_linears[i](h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)

        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(input_pts[:, :3])
            outputs = torch.reshape(bbox_mask, [-1, 1]) * outputs

        return outputs


# Velocity Model
class SIREN_vel(nn.Module):
    def __init__(self, D=6, W=128, input_ch=4, output_ch=3, skips=[], fading_fin_step=0, bbox_model=None):
        """
        fading_fin_step: >0, to fade in layers one by one, fully faded in when self.fading_step >= fading_fin_step
        """

        super(SIREN_vel, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.fading_step = 0
        self.fading_fin_step = fading_fin_step if fading_fin_step > 0 else 0
        self.bbox_model = bbox_model

        first_omega_0 = 30.0
        hidden_omega_0 = 1.0

        self.hid_linears = nn.ModuleList(
            [SineLayer(input_ch, W, omega_0=first_omega_0)]
            + [
                (
                    SineLayer(W, W, omega_0=hidden_omega_0)
                    if i not in self.skips
                    else SineLayer(W + input_ch, W, omega_0=hidden_omega_0)
                )
                for i in range(D - 1)
            ]
        )

        final_vel_linear = nn.Linear(W, output_ch)

        self.vel_linear = final_vel_linear

    def update_fading_step(self, fading_step):
        # should be updated with the global step
        # e.g., update_fading_step(global_step - vel_in_step)
        if fading_step >= 0:
            self.fading_step = fading_step

    def fading_wei_list(self):
        # try print(fading_wei_list()) for debug
        step_ratio = np.clip(float(self.fading_step) / float(max(1e-8, self.fading_fin_step)), 0, 1)
        ma = 1 + (self.D - 2) * step_ratio  # in range of 1 to self.D-1
        fading_wei_list = [np.clip(1 + ma - m, 0, 1) * np.clip(1 + m - ma, 0, 1) for m in range(self.D)]
        return fading_wei_list

    def print_fading(self):
        # w_list = self.fading_wei_list()
        # _str = ["h%d:%0.03f" % (i, w_list[i]) for i in range(len(w_list)) if w_list[i] > 1e-8]
        # print("; ".join(_str))
        pass

    def forward(self, x):
        h = x
        h_layers = []
        for i, l in enumerate(self.hid_linears):
            h = self.hid_linears[i](h)

            h_layers += [h]
            if i in self.skips:
                h = torch.cat([x, h], -1)

        # a sliding window (fading_wei_list) to enable deeper layers progressively
        if self.fading_fin_step > self.fading_step:
            fading_wei_list = self.fading_wei_list()
            h = 0
            for w, y in zip(fading_wei_list, h_layers):
                if w > 1e-8:
                    h = w * y + h

        vel_out = self.vel_linear(h)

        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(x[..., :3])
            vel_out = torch.reshape(bbox_mask, [-1, 1]) * vel_out

        return vel_out


# Model
class SIREN_Hybrid(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=4,
        input_ch_views=3,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
        fading_fin_step_static=0,
        fading_fin_step_dynamic=0,
        bbox_model=None,
    ):
        """
        fading_fin_step: >0, to fade in layers one by one, fully faded in when self.fading_step >= fading_fin_step
        """

        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.fading_step_static = 0
        self.fading_step_dynamic = 0
        self.fading_fin_step_static = fading_fin_step_static if fading_fin_step_static > 0 else 0
        self.fading_fin_step_dynamic = fading_fin_step_dynamic if fading_fin_step_dynamic > 0 else 0
        self.bbox_model = bbox_model

        self.static_model = SIREN_NeRFt(
            D=D,
            W=W,
            input_ch=input_ch - 1,
            input_ch_views=input_ch_views,
            output_ch=output_ch,
            skips=skips,
            use_viewdirs=use_viewdirs,
            fading_fin_step=fading_fin_step_static,
            bbox_model=bbox_model,
        )

        self.dynamic_model = SIREN_NeRFt(
            D=D,
            W=W,
            input_ch=input_ch,
            input_ch_views=input_ch_views,
            output_ch=output_ch,
            skips=skips,
            use_viewdirs=use_viewdirs,
            fading_fin_step=fading_fin_step_dynamic,
            bbox_model=bbox_model,
        )

    def update_fading_step(self, fading_step_static, fading_step_dynamic):
        # should be updated with the global step
        # e.g., update_fading_step(global_step - static_in_step, global_step - dynamic_in_step)
        self.static_model.update_fading_step(fading_step_static)
        self.dynamic_model.update_fading_step(fading_step_dynamic)

    def fading_wei_list(self, isStatic=False):
        if isStatic:
            return self.static_model.fading_wei_list()
        return self.dynamic_model.fading_wei_list()

    def print_fading(self):
        # w_list = self.fading_wei_list(isStatic=True)
        # _str = ["static_h%d:%0.03f" % (i, w_list[i]) for i in range(len(w_list)) if w_list[i] > 1e-8]
        # print("; ".join(_str))
        # w_list = self.fading_wei_list()
        # _str = ["dynamic_h%d:%0.03f" % (i, w_list[i]) for i in range(len(w_list)) if w_list[i] > 1e-8]
        # print("; ".join(_str))
        pass

    def forward(self, x):
        inputs_xyz, input_t, input_views = torch.split(x, [self.input_ch - 1, 1, self.input_ch_views], dim=-1)

        dynamic_x = x
        static_x = torch.cat((inputs_xyz, input_views), dim=-1)

        static_output = self.static_model.forward(static_x)
        dynamic_output = self.dynamic_model.forward(x)
        outputs = torch.cat([static_output, dynamic_output], dim=-1)

        return outputs

    def toDevice(self, device):
        self.static_model = self.static_model.cuda()
        self.dynamic_model = self.dynamic_model.cuda()


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H)
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    # import pdb
    # pdb.set_trace()
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    import pdb

    pdb.set_trace()
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = -1.0 / (W / (2.0 * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1.0 / (H / (2.0 * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    device = weights.get_device()
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=device), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0.0, 1.0, N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u).cuda()

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)

    below = torch.max(torch.zeros_like(inds - 1, device=device), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds, device=device), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
