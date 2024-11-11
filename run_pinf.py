import datetime
import os
import pdb
import shutil
import sys
import time

import trimesh

from lpips import LPIPS

# vel_uv2hsv, den_scalar2rgb, jacobian3D, jacobian3D_np
# vel_world2smoke, vel_smoke2world, pos_world2smoke, pos_smoke2world
# Logger, VGGlossTool, ghost_loss_func
from skimage import measure
from skimage.metrics import structural_similarity

from den2vel import DenToVel
from load_pinf import load_pinf_frame_data
from run_nerf import *
from run_pinf_helpers import *


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
    date_str = datetime.datetime.now().strftime("%m%d-%H%M%S")
    filedir = "log_" + ("train" if not (args.vol_output_only or args.render_only) else "test")
    filedir += date_str
    log_dir = os.path.join(basedir, expname, filedir)
    os.makedirs(log_dir, exist_ok=True)

    sys.stdout = Logger(log_dir, False, fname="log.out")
    sys.stderr = Logger(log_dir, False, fname="log.err")

    print(" ".join(sys.argv), flush=True)
    printENV()

    # files backup
    shutil.copyfile(args.config, os.path.join(basedir, expname, filedir, "config.txt"))
    f = os.path.join(log_dir, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    filelist = ["run_nerf.py", "run_nerf_helpers.py", "run_pinf.py", "run_pinf_helpers.py"]
    for filename in filelist:
        shutil.copyfile("./" + filename, os.path.join(log_dir, filename.replace("/", "_")))

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


def run_future_pred(
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
    y_start = int(source_height * ry)
    print("y_start: {}".format(y_start))
    # render_kwargs.update(y_start=y_start)
    # proj_y = render_kwargs['proj_y']
    for idx, i in enumerate(range(starting_frame + 1, starting_frame + n_pred + 1)):
        c2w = render_poses[0]
        mask_to_sim = coord_3d_sim[..., 1] > source_height
        n_substeps = 1
        use_reflect = False
        render_kwargs["bbox_model"] = bbox_model
        if vort_particles is not None:
            confinement_field = vort_particles(coord_3d_sim, i)
            print(
                "Vortex energy over velocity: {:.2f}%".format(
                    torch.norm(confinement_field, dim=-1).pow(2).sum() / torch.norm(vel, dim=-1).pow(2).sum() * 100
                )
            )
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

        try:
            coord_4d_world[..., 3] = time_steps[i]  # sample density source at current moment
            den[~mask_to_sim] = batchify_query(
                coord_4d_world[~mask_to_sim],
                lambda pts: render_kwargs["network_query_fn"](pts, None, render_kwargs["network_fine"]),
            )
            vel[~mask_to_sim] = batchify_query(coord_4d_world[~mask_to_sim], render_kwargs["network_query_fn_vel"])
        except IndexError:
            pass

        if save_fields:
            save_fields_to_vti(  # type: ignore
                vel.permute(2, 1, 0, 3).cpu().numpy(),
                den.permute(2, 1, 0, 3).cpu().numpy(),
                os.path.join(savedir, "fields_{:03d}.vti".format(idx)),
            )
            print("Saved fields to {}".format(os.path.join(savedir, "fields_{:03d}.vti".format(idx))))
        rgb, _, _, _ = render(
            H, W, K, c2w=c2w[:3, :4], time_step=time_steps[i][None], render_grid=True, den_grid=den, **render_kwargs
        )
        rgb = rgb[90:960, 45:540]  # HWC
        rgb8 = to8b(rgb.cpu().numpy())
        try:
            gt_img = torch.tensor(gt_imgs[i].squeeze()[90:960, 45:540], dtype=torch.float32)  # [H, W, 3]
            lpips_value = lpips_net(rgb.permute(2, 0, 1), gt_img.permute(2, 0, 1), normalize=True).item()
            import pdb

            # pdb.set_trace()
            p = -10.0 * np.log10(np.mean(np.square(rgb.detach().cpu().numpy() - gt_img.cpu().numpy())))
            ssim_value = structural_similarity(gt_img.cpu().numpy(), rgb.cpu().numpy(), data_range=1.0, channel_axis=2)
            lpipss.append(lpips_value)
            psnrs.append(p)
            ssims.append(ssim_value)
            print(f"PSNR: {p:.4g}, SSIM: {ssim_value:.4g}, LPIPS: {lpips_value:.4g}")
        except IndexError:
            pass
        imageio.imsave(os.path.join(savedir, "rgb_{:03d}.png".format(idx)), rgb8)
    merge_imgs(savedir, framerate=10, prefix="rgb_")

    if gt_imgs is not None:
        avg_psnr = sum(psnrs) / len(psnrs)
        print(f"Avg PSNR over full simulation: ", avg_psnr)
        avg_ssim = sum(ssims) / len(ssims)
        print(f"Avg SSIM over full simulation: ", avg_ssim)
        avg_lpips = sum(lpipss) / len(lpipss)
        print(f"Avg LPIPS over full simulation: ", avg_lpips)
        with open(
            os.path.join(savedir, "psnrs_{:0.2f}_ssim_{:.2g}_lpips_{:.2g}.json".format(avg_psnr, avg_ssim, avg_lpips)),
            "w",
        ) as fp:
            json.dump(psnrs, fp)


def run_advect_den(
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
    for i, c2w in enumerate(tqdm(render_poses)):
        gt_den = np.load("./data/ScalarReal/00000_syn_GT/_density/density_{:06}.npz".format(60 + i))["data"]
        gt_vel = np.load("./data/ScalarReal/00000_syn_GT/_velocity/velocity_{:06}.npz".format(60 + i))["data"]
        # positive_den = den[den[...,-1]>0]
        # import pdb
        # pdb.set_trace()
        # positive_den_position = coord_3d_world_den[gt_den[...,-1]>0.1]
        # positive_den_vel = gt_vel[gt_den[...,-1]>0.1]
        # np.save(f"./gt_particles_scene_0/position_{i}", positive_den_position.cpu().numpy())
        # np.save(f"./gt_particles_scene_0/velocity_{i}", positive_den_vel)
        # print(i, positive_den_position.shape)
        # v,f,n,val = measure.marching_cubes(den[...,-1].cpu().numpy(),0.9)
        # mesh = trimesh.Trimesh(vertices = v, faces = f)
        # import pdb
        # pdb.set_trace()
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
            # pdb.set_trace()
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
        # pdb.set_trace()
        rgb, _, _, _ = render(
            H, W, K, c2w=c2w[:3, :4], time_step=time_steps[i][None], render_grid=True, den_grid=den, **render_kwargs
        )
        rgb8 = to8b(rgb.detach().cpu().numpy())
        if gt_imgs is not None:
            try:
                gt_img = gt_imgs[i].detach().cpu().numpy()
            except:
                gt_img = gt_imgs[i]
            p = -10.0 * np.log10(np.mean(np.square(rgb.detach().cpu().numpy() - gt_img)))
            print(f"PSNR: {p:.4g}")
            psnrs.append(p)
        # pdb.set_trace()
        imageio.imsave(os.path.join(savedir, "rgb_{:03d}.png".format(i)), rgb8)
        # imageio.imsave(os.path.join(savedir, 'rgb_gt_{:03d}.png'.format(i)), gt_img.detach().cpu())
    # merge_imgs(savedir, prefix='rgb_')

    if gt_imgs is not None:
        avg_psnr = sum(psnrs) / len(psnrs)
        print(f"Avg PSNR over full simulation: ", avg_psnr)
        with open(os.path.join(savedir, "source{}_psnrs_avg{:0.2f}.json".format(source_height, avg_psnr)), "w") as fp:
            json.dump(psnrs, fp)
        # pdb.set_trace()


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


def pinf_train(parser, args):
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    logdir = save_log(basedir, expname)

    # Load data
    (
        images,
        poses,
        time_steps,
        hwfs,
        render_poses,
        render_timesteps,
        i_split,
        t_info,
        voxel_tran,
        voxel_scale,
        bkg_color,
        near,
        far,
    ) = load_pinf_frame_data(args.datadir, args.half_res, args.testskip)
    voxel_tran_inv = np.linalg.inv(voxel_tran)
    print("Loaded pinf frame data", images.shape, render_poses.shape, hwfs[0], args.datadir)
    print("Loaded voxel matrix", voxel_tran, "voxel scale", voxel_scale)

    voxel_tran_inv = torch.Tensor(voxel_tran_inv)
    voxel_tran = torch.Tensor(voxel_tran)
    voxel_scale = torch.Tensor(voxel_scale)

    i_train, i_val, i_test = i_split
    args.white_bkgd = torch.Tensor(bkg_color).to(device)
    print("Scene has background color", bkg_color, args.white_bkgd)

    Ks = [[[hwf[-1], 0, 0.5 * hwf[1]], [0, hwf[-1], 0.5 * hwf[0]], [0, 0, 1]] for hwf in hwfs]

    if args.render_test:
        render_poses = np.array(poses[i_test])
        render_timesteps = np.array(time_steps[i_test])

    # Create Bbox model
    bbox_model = None
    # in_min, in_max = 0.0, 1.0
    if args.bbox_min != "":
        in_min = [float(_) for _ in args.bbox_min.split(",")]
        in_max = [float(_) for _ in args.bbox_max.split(",")]
        bbox_model = BBox_Tool(voxel_tran_inv, voxel_scale, in_min, in_max)

    # Create vel model
    vel_model = None

    if args.nseW > 1e-8:
        # D=6, W=128, input_ch=4, output_ch=3, skips=[],
        vel_model = SIREN_vel(fading_fin_step=args.fading_layers, bbox_model=bbox_model).to(device)

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, vel_optimizer = create_nerf(
        args, vel_model=vel_model, bbox_model=bbox_model, ndim=4
    )
    global_step = start

    update_dict = {
        "near": near,
        "far": far,
        "has_t": True,
    }
    render_kwargs_train.update(update_dict)
    render_kwargs_test.update(update_dict)
    render_kwargs_train["vel_model"] = vel_model
    render_kwargs_test["remove99"] = True

    all_models = {
        "vel_model": vel_model,
        "coarse": render_kwargs_train["network_fn"],
        "fine": render_kwargs_train["network_fine"],
    }
    save_dic_keys = {
        "vel_model": "network_vel_state_dict",
        "coarse": "network_fn_state_dict",
        "fine": "network_fine_state_dict",
    }

    tempoInStep = max(0, args.tempo_delay) if args.net_model == "hybrid" else 0
    velInStep = max(0, args.vel_delay) if args.nseW > 1e-8 else 0  # after tempoInStep
    if args.net_model != "nerf":
        model_fading_update(all_models, start, tempoInStep, velInStep, args.net_model == "hybrid")

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    render_timesteps = torch.Tensor(render_timesteps).to(device)
    test_bkg_color = np.float32([0.0, 0.0, 0.3])

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print("RENDER ONLY")
        import pdb

        # pdb.set_trace()
        hwf = hwfs[0]
        hwf = [int(hwf[0]) * 2, int(hwf[1]) * 2, float(hwf[2]) * 2]
        K = Ks[0]

        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
                hwf = hwfs[i_test[0]]
                render_poses = poses[i_test]
                render_poses = torch.Tensor(render_poses).to(device)
                hwf = [int(hwf[0]), int(hwf[1]), float(hwf[2])]
                K = Ks[i_test[0]]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(
                basedir, expname, "renderonly_{}_{:06d}".format("test" if args.render_test else "path", start + 1)
            )
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", render_poses.shape)
            # pdb.set_trace()
            render_kwargs_test["network_query_fn_vel"] = vel_model
            rgbs, _ = render_path(
                render_poses,
                hwf,
                K,
                args.chunk,
                render_kwargs_test,
                gt_imgs=images,
                savedir=testsavedir,
                render_factor=args.render_factor,
                render_steps=render_timesteps,
                bbox_model=bbox_model,
                render_vel=True,
                bkgd_color=test_bkg_color,
            )
            print("Done rendering", testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, "video.mp4"), to8b(rgbs), fps=30, quality=8)

            return

    if args.vol_output_only:
        print("OUTPUT VOLUME ONLY")
        savenpz = True  # need a large space
        savejpg = True
        save_vort = True  # (vel_model is not None) and (savenpz) and (savejpg)
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname, "volumeout_{:06d}".format(start + 1))
            os.makedirs(testsavedir, exist_ok=True)

            resX = args.vol_output_W
            resY = int(args.vol_output_W * float(voxel_scale[1]) / voxel_scale[0] + 0.5)
            resZ = int(args.vol_output_W * float(voxel_scale[2]) / voxel_scale[0] + 0.5)
            voxel_writer = Voxel_Tool(voxel_tran, voxel_tran_inv, voxel_scale, resZ, resY, resX, middleView="mid3")

            t_list = list(np.arange(t_info[0], t_info[1], t_info[-1]))
            frame_N = len(t_list)
            noStatic = False
            for frame_i in range(frame_N // 10, frame_N, 10):
                print(frame_i, frame_N)
                cur_t = t_list[frame_i]
                voxel_writer.save_voxel_den_npz(
                    os.path.join(testsavedir, "d_%04d.npz" % frame_i),
                    cur_t,
                    args.use_viewdirs,
                    network_query_fn=render_kwargs_test["network_query_fn"],
                    network_fn=render_kwargs_test["network_fine" if args.N_importance > 0 else "network_fn"],
                    chunk=args.chunk,
                    save_npz=savenpz,
                    save_jpg=savejpg,
                    noStatic=noStatic,
                )
                noStatic = True
                if vel_model is not None:
                    voxel_writer.save_voxel_vel_npz(
                        os.path.join(testsavedir, "v_%04d.npz" % frame_i),
                        t_info[-1],
                        cur_t,
                        batchify,
                        args.chunk,
                        vel_model,
                        savenpz,
                        savejpg,
                        save_vort,
                    )
            print("Done output", testsavedir)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if (use_batching) or (N_rand is None):
        print("Not supported!")
        return

    # Prepare Loss Tools (VGG, Den2Vel)
    ###############################################
    vggTool = VGGlossTool(device)

    # Move to GPU, except images
    poses = torch.Tensor(poses).to(device)
    time_steps = torch.Tensor(time_steps).to(device)

    N_iters = args.N_iter + 1

    print("Begin")
    print("TRAIN views are", i_train)
    print("TEST views are", i_test)
    print("VAL views are", i_val)

    # Prepare Voxel Sampling Tools for Image Summary (voxel_writer), Physical Priors (training_voxel), Data Priors Represented by D2V (den_p_all)
    # voxel_writer: to sample low resolution data for for image summary
    resX = 64  # complexity O(N^3)
    resY = int(resX * float(voxel_scale[1]) / voxel_scale[0] + 0.5)
    resZ = int(resX * float(voxel_scale[2]) / voxel_scale[0] + 0.5)
    voxel_writer = Voxel_Tool(voxel_tran, voxel_tran_inv, voxel_scale, resZ, resY, resX, middleView="mid3")

    # training_voxel: to sample data for for velocity NSE training
    # training_voxel should have a larger resolution than voxel_writer
    # note that training voxel is also used for visualization in testing
    min_ratio = float(64 + 4 * 2) / min(voxel_scale[0], voxel_scale[1], voxel_scale[2])
    minX = int(min_ratio * voxel_scale[0] + 0.5)
    trainX = max(args.vol_output_W, minX)  # a minimal resolution of 64^3
    trainY = int(trainX * float(voxel_scale[1]) / voxel_scale[0] + 0.5)
    trainZ = int(trainX * float(voxel_scale[2]) / voxel_scale[0] + 0.5)
    training_voxel = Voxel_Tool(voxel_tran, voxel_tran_inv, voxel_scale, trainZ, trainY, trainX, middleView="mid3")
    training_pts = torch.reshape(training_voxel.pts, (-1, 3))
    # pdb.set_trace()
    # prepare grid positions for velocity d2v training, its shortest spatial dim has denTW+d2v_min_border*2 cells
    denTW = 64  # 256 will be 30 times slower
    d2v_min_border = 2
    denRatio = float(denTW + 2 * d2v_min_border) / min(trainX, trainY, trainZ)
    den_p_all = get_voxel_pts(
        int(trainY * denRatio + 1e-6),
        int(trainX * denRatio + 1e-6),
        int(trainZ * denRatio + 1e-6),
        voxel_tran,
        voxel_scale,
    )
    train_reso_scale = torch.Tensor([256 * t_info[-1], 256 * t_info[-1], 256 * t_info[-1]])
    # train_reso_scale: d2v model is trained on simulation data with resoluiton of 256^3

    split_nse_wei = [2.0, 1e-3, 1e-3, 1e-3, 1e-3, 5e-3]
    start = start + 1

    testimgdir = os.path.join(basedir, expname, "imgs_" + logdir)
    os.makedirs(testimgdir, exist_ok=True)
    # some loss terms
    ghost_loss, overlay_loss, nseloss_fine, d2v_error = None, None, None, None

    d2v_model = DenToVel() if args.d2vW > 1e-8 else None
    for i in trange(start, N_iters):
        import pdb

        # pdb.set_trace()
        # time0 = time.time()
        if args.net_model != "nerf":
            model_fading_update(all_models, global_step, tempoInStep, velInStep, args.net_model == "hybrid")

        # train radiance all the time, train vel less, train with d2v even less.
        trainImg = True
        trainVGG = (args.vggW > 0.0) and (i % 4 == 0)  # less vgg training
        trainVel = (global_step >= (tempoInStep + velInStep)) and (vel_model is not None) and (i % 10 == 0)
        trainD2V = (args.d2vW > 0.0) and (global_step >= (tempoInStep + velInStep * 2)) and trainVel and (i % 20 == 0)

        # fading in for networks
        tempo_fading = fade_in_weight(global_step, tempoInStep, 10000)
        vel_fading = fade_in_weight(global_step, tempoInStep + velInStep, 10000)
        warp_fading = fade_in_weight(global_step, tempoInStep + velInStep + 10000, 20000)
        # fading in for losses
        vgg_fading = [
            fade_in_weight(global_step, (vgg_i - 1) * 10000, 10000) for vgg_i in range(len(vggTool.layer_list), 0, -1)
        ]
        ghost_fading = fade_in_weight(global_step, tempoInStep + 2000, 20000)
        d2v_fading = fade_in_weight(global_step, tempoInStep + velInStep * 2, 20000)
        ###########################################################

        # Random from one frame
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3, :4]
        time_locate = torch.Tensor(time_steps[img_i]).to(device)

        # Cast intrinsics to right types
        H, W, focal = hwfs[img_i]
        H, W = int(H), int(W)
        focal = float(focal)
        hwf = [H, W, focal]
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

        if trainVel:
            # take a mini_batch 32*32*32
            if trainD2V:  # a cropped 32^3 as a mini_batch
                offset_w = np.int32(
                    np.random.uniform(d2v_min_border, int(trainX * denRatio + 1e-6) - denTW - d2v_min_border, [])
                )
                offset_h = np.int32(
                    np.random.uniform(d2v_min_border, int(trainY * denRatio + 1e-6) - denTW - d2v_min_border, [])
                )
                offset_d = np.int32(
                    np.random.uniform(d2v_min_border, int(trainZ * denRatio + 1e-6) - denTW - d2v_min_border, [])
                )
                den_p_crop = den_p_all[
                    offset_d : offset_d + denTW : 2,
                    offset_h : offset_h + denTW : 2,
                    offset_w : offset_w + denTW : 2,
                    :,
                ]
                training_samples = torch.reshape(den_p_crop, (-1, 3))
                # training_samples = get_voxel_pts_offset(32, 32, 32, voxel_tran, voxel_scale, r_offset=4.0)
                # training_samples = torch.reshape(training_samples, (-1,3))
            else:  # a random mini_batch
                train_random = np.random.choice(trainZ * trainY * trainX, 32 * 32 * 32)
                training_samples = training_pts[train_random]

            training_samples = training_samples.view(-1, 3)
            training_t = torch.ones([training_samples.shape[0], 1]) * time_locate
            training_samples = torch.cat([training_samples, training_t], dim=-1)

            #####  core velocity optimization loop  #####
            # allows to take derivative w.r.t. training_samples
            training_samples = training_samples.clone().detach().requires_grad_(True)
            _vel, _u_x, _u_y, _u_z, _u_t = training_voxel.get_velocity_and_derivatives(
                training_samples, chunk=args.chunk, batchify_fn=batchify, vel_model=vel_model
            )
            _den, _d_x, _d_y, _d_z, _d_t = training_voxel.get_density_and_derivatives(
                training_samples,
                chunk=args.chunk,
                use_viewdirs=False,
                network_query_fn=render_kwargs_test["network_query_fn"],
                network_fn=render_kwargs_test["network_fine" if args.N_importance > 0 else "network_fn"],
            )

            # get vorticity in 32^3 smoke resolution coord for training
            if trainD2V:  # all data are in a cropped 32^3 grid as a mini_batch
                with torch.no_grad():
                    # in trainZ, trainY, trainX resolution space
                    _den_voxel = _den.detach().view(1, 1, 32, 32, 32)  # b,c,DHW
                    den_mask = torch.where(
                        _den_voxel > 1e-6, torch.ones_like(_den_voxel), torch.zeros_like(_den_voxel)
                    ).view(32, 32, 32, 1)

                    if torch.mean(den_mask) > 0.05:  # at least 5 percent are valid
                        den_voxel_raw = F.interpolate(_den_voxel, size=64, mode="trilinear", align_corners=False) / (
                            _den_voxel.max() + 1e-8
                        )
                        vel_d2v_smoke = d2v_model(den_voxel_raw).permute([0, -1, 1, 2, 3])  # 1,3,64,64,64
                        vel_d2v_smoke = F.interpolate(
                            vel_d2v_smoke, size=32, mode="trilinear", align_corners=False
                        ).permute([0, 2, 3, 4, 1])
                        # 1,32,32,32,3
                        # scale according to vel_pred_smoke_view
                        vel_pred_smoke_view = vel_world2smoke(
                            _vel.detach(), voxel_tran_inv, voxel_scale, train_reso_scale
                        ).view(32, 32, 32, 3)
                        scale_factor = torch.mean(
                            torch.sqrt(torch.sum(torch.square(vel_pred_smoke_view * den_mask), dim=-1) + 1e-8)
                        )
                        scale_factor = scale_factor / torch.mean(
                            torch.sqrt(torch.sum(torch.square(vel_d2v_smoke * den_mask), dim=-1) + 1e-8)
                        )
                        vel_d2v_smoke = vel_d2v_smoke * scale_factor

                        # jac in trainZ, trainY, trainX resolution space
                        d2v_jac, d2v_vort = jacobian3D(vel_d2v_smoke)  # 1,32,32,32,9 and 1,32,32,32,3
                        d2v_udx = vel_smoke2world(d2v_jac[..., 0::3], voxel_tran, voxel_scale, train_reso_scale)
                        d2v_udy = vel_smoke2world(d2v_jac[..., 1::3], voxel_tran, voxel_scale, train_reso_scale)
                        d2v_udz = vel_smoke2world(d2v_jac[..., 2::3], voxel_tran, voxel_scale, train_reso_scale)
                        d2v_u_jac = torch.cat([d2v_udx, d2v_udy, d2v_udz], dim=-1).view(32, 32, 32, 9).detach()

                        smoke_baseX = off_smoke2world(torch.Tensor([1.0 / 256, 0.0, 0.0]), voxel_tran, voxel_scale)
                        smoke_baseY = off_smoke2world(torch.Tensor([0.0, 1.0 / 256, 0.0]), voxel_tran, voxel_scale)
                        smoke_baseZ = off_smoke2world(torch.Tensor([0.0, 0.0, 1.0 / 256]), voxel_tran, voxel_scale)
                    else:
                        trainD2V = False  # train d2v next time.

            vel_optimizer.zero_grad()
            split_nse = PDE_EQs(
                _d_t.detach(), _d_x.detach(), _d_y.detach(), _d_z.detach(), _vel, _u_t, _u_x, _u_y, _u_z
            )
            nse_errors = [mean_squared_error(x, 0.0) for x in split_nse]
            nseloss_fine = 0.0
            for ei, wi in zip(nse_errors, split_nse_wei):
                nseloss_fine = ei * wi + nseloss_fine
            vel_loss = nseloss_fine * args.nseW * vel_fading

            if trainD2V:
                worldU_smokeX = smoke_baseX[0] * _u_x + smoke_baseX[1] * _u_y + smoke_baseX[2] * _u_z
                worldU_smokeY = smoke_baseY[0] * _u_x + smoke_baseY[1] * _u_y + smoke_baseY[2] * _u_z
                worldU_smokeZ = smoke_baseZ[0] * _u_x + smoke_baseZ[1] * _u_y + smoke_baseZ[2] * _u_z
                cur_jac = torch.cat([worldU_smokeX, worldU_smokeY, worldU_smokeZ], dim=-1)  # 9
                cur_jac = cur_jac.view(32, 32, 32, 9)

                d2v_jac_scale = torch.mean(torch.sqrt(torch.sum(torch.square(d2v_u_jac * den_mask), dim=-1))).detach()
                cur_jac_scale = torch.mean(torch.sqrt(torch.sum(torch.square(cur_jac * den_mask), dim=-1))).detach()

                d2v_error = mean_squared_error(cur_jac * den_mask, d2v_u_jac * den_mask) / torch.mean(den_mask)
                vel_loss += d2v_error * args.d2vW * d2v_fading

            vel_loss.backward()
            vel_optimizer.step()

        if trainImg:
            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            if trainVGG:  # get a cropped img (dw,dw) to train vgg
                strides = args.vgg_strides + i % 3 - 1

                # args.vgg_strides-1, args.vgg_strides, args.vgg_strides+1
                dw = int(max(20, min(40, N_rand**0.5)))
                vgg_min_border = 10
                strides = min(strides, min(H - vgg_min_border, W - vgg_min_border) / dw)
                strides = int(strides)

                coords = torch.stack(
                    torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1
                )  # (H, W, 2)
                if True:
                    target_grey = torch.mean(torch.abs(target - args.white_bkgd), dim=-1, keepdim=True)  # H,W,1
                    img_wei = coords.to(torch.float32) * target_grey
                    center_coord = torch.sum(img_wei, dim=(0, 1)) / torch.sum(target_grey)
                    center_coord = center_coord.cpu().numpy()
                    # add random jitter
                    random_R = dw * strides / 2.0
                    # mean and standard deviation: center_coord, random_R/3.0, so that 3sigma < random_R
                    random_x = np.random.normal(center_coord[1], random_R / 3.0) - 0.5 * dw * strides
                    random_y = np.random.normal(center_coord[0], random_R / 3.0) - 0.5 * dw * strides
                else:
                    random_x = (
                        np.random.uniform(
                            low=vgg_min_border + 0.5 * dw * strides, high=W - 0.5 * dw * strides - vgg_min_border
                        )
                        - 0.5 * dw * strides
                    )
                    random_y = (
                        np.random.uniform(
                            low=vgg_min_border + 0.5 * dw * strides, high=W - 0.5 * dw * strides - vgg_min_border
                        )
                        - 0.5 * dw * strides
                    )

                offset_w = int(min(max(vgg_min_border, random_x), W - dw * strides - vgg_min_border))
                offset_h = int(min(max(vgg_min_border, random_y), H - dw * strides - vgg_min_border))

                coords_crop = coords[
                    offset_h : offset_h + dw * strides : strides, offset_w : offset_w + dw * strides : strides, :
                ]

                select_coords = torch.reshape(coords_crop, [-1, 2]).long()
            else:
                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                        ),
                        -1,
                    )
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}"
                        )
                else:
                    coords = torch.stack(
                        torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1
                    )  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)

                select_coords = coords[select_inds].long()  # (N_rand, 2)

            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            if args.train_warp and vel_model is not None and (global_step >= tempoInStep + velInStep):
                render_kwargs_train["warp_fading_dt"] = warp_fading * t_info[-1]
                # fading * delt_T, need to update every iteration

            #####  core radiance optimization loop  #####
            rgb, disp, acc, extras = render(
                H,
                W,
                K,
                chunk=args.chunk,
                rays=batch_rays,
                verbose=i < 10,
                retraw=True,
                time_step=time_locate,
                bkgd_color=args.white_bkgd,
                **render_kwargs_train,
            )
            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            if (args.net_model == "hybrid") and ("rgbh1" in extras) and (tempo_fading < (1.0 - 1e-8)):  # rgbh1: static
                img_loss = img_loss * tempo_fading + img2mse(extras["rgbh1"], target_s) * (1.0 - tempo_fading)
                # rgb = rgb * tempo_fading + extras['rgbh1'] * (1.0-tempo_fading)

            # trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if "rgb0" in extras:
                img_loss0 = img2mse(extras["rgb0"], target_s)
                if (
                    (args.net_model == "hybrid") and ("rgbh10" in extras) and (tempo_fading < (1.0 - 1e-8))
                ):  # rgbh1: static
                    img_loss0 = img_loss0 * tempo_fading + img2mse(extras["rgbh10"], target_s) * (1.0 - tempo_fading)
                    # extras['rgb0'] = extras['rgb0'] * tempo_fading + extras['rgbh10'] * (1.0-tempo_fading)

                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            if trainVGG:
                vgg_loss_func = vggTool.compute_cos_loss
                vgg_tar = torch.reshape(target_s, [dw, dw, 3])
                vgg_img = torch.reshape(rgb, [dw, dw, 3])
                vgg_loss = vgg_loss_func(vgg_img, vgg_tar)
                w_vgg = args.vggW / float(len(vgg_loss))
                vgg_loss_sum = 0
                for _w, _wf in zip(vgg_loss, vgg_fading):
                    if _wf > 1e-8:
                        vgg_loss_sum = _w * _wf * w_vgg + vgg_loss_sum

                if "rgb0" in extras:
                    vgg_img0 = torch.reshape(extras["rgb0"], [dw, dw, 3])
                    vgg_loss0 = vgg_loss_func(vgg_img0, vgg_tar)
                    for _w, _wf in zip(vgg_loss0, vgg_fading):
                        if _wf > 1e-8:
                            vgg_loss_sum = _w * _wf * w_vgg + vgg_loss_sum
                loss += vgg_loss_sum

            if (args.ghostW > 0.0) and args.white_bkgd is not None:
                w_ghost = ghost_fading * args.ghostW
                if w_ghost > 1e-8:
                    static_back = args.white_bkgd
                    ghost_loss = ghost_loss_func(rgb, static_back, acc, den_penalty=0.0)
                    if args.net_model == "hybrid":
                        if global_step > tempoInStep and ("rgbh1" in extras):  # static part
                            # ghost_loss += 0.1*ghost_loss_func(extras['rgbh1'], static_back, extras['acch1'], den_penalty=0.0)
                            if "rgbh2" in extras:  # dynamic part
                                ghost_loss += 0.1 * ghost_loss_func(
                                    extras["rgbh2"], extras["rgbh1"], extras["acch2"], den_penalty=0.0
                                )

                    if "rgb0" in extras:
                        ghost_loss0 = ghost_loss_func(extras["rgb0"], static_back, extras["acc0"], den_penalty=0.0)
                        if args.net_model == "hybrid":
                            if global_step > tempoInStep and ("rgbh10" in extras):  # static part
                                # ghost_loss0 += 0.1*ghost_loss_func(extras['rgbh10'], static_back, extras['acch10'], den_penalty=0.0)
                                if "rgbh20" in extras:  # dynamic part
                                    ghost_loss0 += 0.1 * ghost_loss_func(
                                        extras["rgbh20"], extras["rgbh10"], extras["acch20"], den_penalty=0.0
                                    )
                        ghost_loss += ghost_loss0

                    loss += ghost_loss * w_ghost

            if (args.net_model == "hybrid") and (args.overlayW > 0):
                # density should be either from smoke or from static, not mixed.
                w_overlay = args.overlayW * ghost_fading  # with fading

                smoke_den, static_den = F.relu(extras["raw"][..., -1]), F.relu(extras["raw_static"][..., -1])
                overlay_loss = torch.div(
                    2.0 * (smoke_den * static_den), (torch.square(smoke_den) + torch.square(static_den) + 1e-8)
                )
                overlay_loss = torch.mean(overlay_loss)
                loss += overlay_loss * w_overlay

            loss.backward()
            optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate
        if trainVel and vel_optimizer is not None:
            for param_group in vel_optimizer.param_groups:
                param_group["lr"] = new_lrate

        ################################
        # dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i))
            save_dic = {
                "global_step": global_step,
                "network_fn_state_dict": render_kwargs_train["network_fn"].state_dict(),
                "network_fine_state_dict": render_kwargs_train["network_fine"].state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            if vel_optimizer is not None:
                save_dic["vel_optimizer_state_dict"] = vel_optimizer.state_dict()

            for _m in all_models:
                if all_models[_m] is None:
                    continue
                if _m != "vel_model" and args.net_model == "hybrid":
                    save_dic[save_dic_keys[_m] + "_static"] = all_models[_m].static_model.state_dict()
                    save_dic[save_dic_keys[_m] + "_dynamic"] = all_models[_m].dynamic_model.state_dict()
                else:
                    save_dic[save_dic_keys[_m]] = all_models[_m].state_dict()

            torch.save(save_dic, path)
            print("Saved checkpoints at", path)

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode

            with torch.no_grad():
                hwf = hwfs[0]
                hwf = [int(hwf[0]), int(hwf[1]), float(hwf[2])]
                # the path rendering can be very slow.

                testsavedir = os.path.join(basedir, expname, "run_advect_den_{:06d}".format(start))
                os.makedirs(testsavedir, exist_ok=True)

                i_render = i_test
                images_test = images[i_render]
                render_poses_test = poses[i_render]
                hwf = hwfs[i_render[0]]
                hwf = [int(hwf[0]), int(hwf[1]), float(hwf[2])]
                K = Ks[i_render[0]]
                N_timesteps = images_test.shape[0]
                test_timesteps = torch.arange(N_timesteps) / (N_timesteps - 1.0)
                render_kwargs_test.update(network_query_fn_vel=vel_model)
                # rgbs, disps = render_path(render_poses_test, hwf, K, args.chunk, render_kwargs_test, render_steps=render_timesteps, bkgd_color=test_bkg_color,render_vel=False)
                run_advect_den(
                    render_poses_test,
                    hwf,
                    K,
                    test_timesteps,
                    testsavedir,
                    images_test,
                    bbox_model,
                    **render_kwargs_test,
                )
                # import pdb
                # pdb.set_trace()
            # print('Done, saving', rgbs.shape, disps.shape)
            # moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            # imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            # imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            v_deltaT = 0.025
            with torch.no_grad():
                vel_rgbs = []
                for _t in range(int(1.0 / v_deltaT)):
                    # pdb.set_trace()
                    # middle_slice, True: only sample middle slices for visualization, very fast, but cannot save as npz
                    #               False: sample whole volume, can be saved as npz, but very slow
                    voxel_vel = training_voxel.get_voxel_velocity(
                        t_info[-1], _t * v_deltaT, batchify, args.chunk, vel_model, middle_slice=True
                    )
                    voxel_vel = voxel_vel.view([-1] + list(voxel_vel.shape))
                    _, voxel_vort = jacobian3D(voxel_vel)
                    _vel = vel_uv2hsv(np.squeeze(voxel_vel.detach().cpu().numpy()), scale=300, is3D=True, logv=False)
                    _vort = vel_uv2hsv(
                        np.squeeze(voxel_vort.detach().cpu().numpy()), scale=1500, is3D=True, logv=False
                    )
                    vel_rgbs.append(np.concatenate([_vel, _vort], axis=0))
            moviebase = os.path.join(basedir, expname, "{}_volume_{:06d}_".format(expname, i))
            imageio.mimwrite(moviebase + "velrgb.mp4", np.stack(vel_rgbs, axis=0).astype(np.uint8), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, "testset_{:06d}".format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", poses[i_test].shape)
            with torch.no_grad():
                hwf = hwfs[i_test[0]]
                hwf = [int(hwf[0]), int(hwf[1]), float(hwf[2])]
                render_path(
                    torch.Tensor(poses[i_test]).to(device),
                    hwf,
                    Ks[i_test[0]],
                    args.chunk,
                    render_kwargs_test,
                    gt_imgs=images[i_test],
                    savedir=testsavedir,
                    render_steps=time_steps[i_test],
                    bkgd_color=args.white_bkgd,
                )
            print("Saved test set")

        if i % args.i_print == 0:
            print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            if trainImg:
                if args.N_importance > 0:
                    print("img_loss: ", img_loss.item(), img_loss0.item())
                else:
                    print("img_loss: ", img_loss.item())

                if trainVGG:
                    print("vgg_loss: %0.4f *" % w_vgg, vgg_loss_sum.item())
                    for vgg_i in range(len(vgg_loss)):
                        _wf = vgg_fading[vgg_i]
                        if args.N_importance > 0:
                            print(
                                vggTool.layer_names[vgg_i],
                                vgg_loss[vgg_i].item(),
                                "+",
                                vgg_loss0[vgg_i].item(),
                                "with vgg_fading: %0.4f" % _wf,
                            )
                        else:
                            print(vggTool.layer_names[vgg_i], vgg_loss[vgg_i].item(), "with vgg_fading: %0.4f" % _wf)

                if ghost_loss is not None:
                    _g = ghost_loss.item()
                    _cg = ghost_loss0.item() if args.N_importance > 0 else 0
                    print("w_ghost: %0.4f," % w_ghost, "ghost_loss: ", _g, "coarse: ", _cg, "fine: ", _g - _cg)

                if overlay_loss is not None:
                    print("w_overlay: %0.4f," % w_overlay, "overlay_loss: ", overlay_loss.item())

            if trainVel:
                print("vel_loss: ", vel_loss.item())

                if nseloss_fine is not None:
                    print(" ".join(["nse(e1-e6):"] + [str(ei.item()) for ei in nse_errors]))
                    print(
                        "NSE loss sum = ",
                        nseloss_fine.item(),
                        "* w_nse(%0.4f) * vel_fading(%0.4f)" % (args.nseW, vel_fading),
                    )

                if d2v_error is not None:
                    print(
                        "d2v_error, ", d2v_error.item(), "* w_d2v(%0.4f) * d2v_fading(%0.4f)" % (args.d2vW, d2v_fading)
                    )
                    print(
                        "d2v_scale_factor",
                        scale_factor.item(),
                        "d2v_jac_scale",
                        d2v_jac_scale.item(),
                        "cur_jac_scale",
                        cur_jac_scale.item(),
                    )

            if args.net_model != "nerf":
                for _m in all_models:
                    if all_models[_m] is None:
                        continue
                    all_models[_m].print_fading()

            # mem_t = torch.cuda.get_device_properties(0).total_memory
            # mem_r = torch.cuda.memory_reserved(0)
            # mem_a = torch.cuda.memory_allocated(0)
            # mem_mr = torch.cuda.max_memory_reserved(0)
            # mem_ma = torch.cuda.max_memory_allocated(0)
            # print("[GPU_MEM] Total %0.2fG, alloc/reserv %0.2fG/%0.2fG, max_alloc/reserv %0.2fG/%0.2fG"%
            #     ((mem_t>>20)/1024.0, (mem_a>>20)/1024.0, (mem_r>>20)/1024.0, (mem_ma>>20)/1024.0, (mem_mr>>20)/1024.0))
            #
            # # Runing the Sphere scene using a single NVTIDIA Quadro RTX 8000, we get :
            # # [GPU memory] Total 47.46G, alloc/reserv 3.84G/19.09G, max_alloc/reserv 16.02G/19.09G

            sys.stdout.flush()

        if i % args.i_img == 0:
            with torch.no_grad():
                if trainVGG:
                    vgg_img = np.concatenate([vgg_tar.cpu().detach().numpy(), vgg_img.cpu().detach().numpy()], axis=1)
                    imageio.imwrite(os.path.join(testimgdir, "vggcmp_{:06d}.jpg".format(i)), to8b(vgg_img))

                voxel_den_list = voxel_writer.get_voxel_density_list(
                    0.5,
                    args.chunk,
                    args.use_viewdirs,
                    network_query_fn=render_kwargs_test["network_query_fn"],
                    network_fn=render_kwargs_test["network_fine" if args.N_importance > 0 else "network_fn"],
                    middle_slice=False,
                )[::-1]
                if trainVel:
                    voxel_den_list.append(
                        voxel_writer.get_voxel_velocity(
                            t_info[-1] * float(args.vol_output_W) / resX, 0.5, batchify, args.chunk, vel_model, True
                        )
                    )
                voxel_img = []
                for voxel in voxel_den_list:
                    voxel = voxel.detach().cpu().numpy()
                    if voxel.shape[-1] == 1:
                        voxel_img.append(
                            np.repeat(den_scalar2rgb(voxel, scale=None, is3D=True, logv=False, mix=True), 3, axis=-1)
                        )
                    else:
                        voxel_img.append(vel_uv2hsv(voxel, scale=300, is3D=True, logv=False))
                voxel_img = np.concatenate(voxel_img, axis=0)  # 128,64*3,3
                imageio.imwrite(os.path.join(testimgdir, "vox_{:06d}.png".format(i)), voxel_img)

                if d2v_error is not None:
                    # gen model vorticity in smoke coord for visualization
                    cur_ux_smoke = (
                        vel_world2smoke(worldU_smokeX, voxel_tran_inv, voxel_scale, train_reso_scale)
                        .view(32, 32, 32, 3)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    cur_uy_smoke = (
                        vel_world2smoke(worldU_smokeY, voxel_tran_inv, voxel_scale, train_reso_scale)
                        .view(32, 32, 32, 3)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    cur_uz_smoke = (
                        vel_world2smoke(worldU_smokeZ, voxel_tran_inv, voxel_scale, train_reso_scale)
                        .view(32, 32, 32, 3)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    u = cur_uz_smoke[..., 1] - cur_uy_smoke[..., 2]  # dwdy - dvdz
                    v = cur_ux_smoke[..., 2] - cur_uz_smoke[..., 0]  # dudz - dwdx
                    w = cur_uy_smoke[..., 0] - cur_ux_smoke[..., 1]  # dvdx - dudy
                    cur_vort = np.stack([u, v, w], axis=-1)

                    img_den = den_scalar2rgb(
                        den_voxel_raw[0, 0, ...].detach().cpu().numpy(), scale=None, is3D=True, logv=False, mix=True
                    )  # 64,64*3,3
                    img_vel1 = vel_uv2hsv(
                        np.squeeze(vel_d2v_smoke.detach().cpu().numpy()), scale=300, is3D=True, logv=False
                    )  # 32, 32*3,3
                    img_vel2 = vel_uv2hsv(
                        np.squeeze(vel_pred_smoke_view.detach().cpu().numpy()), scale=300, is3D=True, logv=False
                    )  # 32, 32*3,3
                    img_vor1 = vel_uv2hsv(
                        np.squeeze(d2v_vort.detach().cpu().numpy()), scale=300, is3D=True, logv=False
                    )  # 32, 32*3,3
                    #####################################################################
                    ### d2v loss helps to enhance the vorticity caused by buoyancy, but the vorticity is still not as large as the d2v_vort
                    ### we use a larger scaling factor to visualize. d2v_jac_scale and cur_jac_scale are printed in the log file
                    #####################################################################
                    img_vor2 = vel_uv2hsv(np.squeeze(cur_vort), scale=1500, is3D=True, logv=False)  # 32, 32*3,3
                    img_d2v = np.concatenate([img_vel1, img_vor1], axis=1)  # 32, 64*3,3
                    img_cur = np.concatenate([img_vel2, img_vor2], axis=1)  # 32, 64*3,3
                    img_ = np.concatenate([np.repeat(img_den, 3, axis=-1), img_d2v, img_cur], axis=0)  # 32, 64*3,3
                    imageio.imwrite(os.path.join(testimgdir, "d2v_{:06d}.png".format(i)), img_)

        global_step += 1


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    parser = config_parser()
    args = parser.parse_args()
    set_rand_seed(args.fix_seed)

    bkg_flag = args.white_bkgd
    args.white_bkgd = np.ones([3], dtype=np.float32) if bkg_flag else None

    if args.dataset_type == "pinf_data":
        pinf_train(parser, args)
    else:
        train(parser, args)  # call train in run_nerf
