import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *
from opt import *
from mcdropout_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
from torch.utils.tensorboard import SummaryWriter

from entropy_loss import *
from visualization import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
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
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    accs = []
    others = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, extra = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        accs.append(acc.cpu().numpy())
        others.append(extra)


        if i==0:
            print(rgb.shape, disp.shape, acc.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    accs = np.stack(accs, 0)
    extras = {}

    for k in others[-1].keys():
        k_values = [extra[k].cpu().numpy() for extra in others]
        extras[k] = np.stack(k_values,0)

    return rgbs, disps, accs, extras


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)[1:-1]
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    # model = nn.DataParallel(model)
    model.to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        # model_fine = nn.DataParallel(model_fine)
        model_fine.to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['out_others'] = True

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False,out_others=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    sigma = F.relu(raw[..., 3] + noise)
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    jacobs = torch.zeros(alpha.shape)
    if raw.size(-1)>4:
        h = raw[...,3:]
        jacob_pt = (h**2).mean(-1)
        jacobs = torch.sum(weights*jacob_pt,-1).cpu()

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    others = {}
    if out_others:
        others['alpha'] = alpha
        others['sigma'] = sigma
        others['dists'] = dists
        others['jacobs'] = jacobs
        return rgb_map, disp_map, acc_map, weights, depth_map, others

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                out_others=False):
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
      lindisp: bool. If True, sample linearly in inverse depth rather than in dfepth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
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
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)
        if out_others:
            rgb_map, disp_map, acc_map, weights, depth_map,others = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest,out_others=True)
        else:
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                 white_bkgd, pytest=pytest)
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}

    if out_others:
        ret['sigma'] = others['sigma']
        ret['alpha'] = others['alpha']
        ret['z_vals'] = z_vals
        ret['dists'] = others['dists']
        ret['jacobs'] = others['jacobs']

    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def batch_render(K, args,hwf, render_kwargs, render_poses, gt_imgs=None, savedir=None,batch_size=20):
    N_poses = len(render_poses)
    rgbs = []
    disps = []
    accs = []
    others = []
    extras = {}
    # entropy_maps = []

    if args.mc_dropout:
        enable_dropout(render_kwargs['network_fn'])
        H, W, focal = hwf
        uncerts = []
        for j, c2w in enumerate(tqdm(render_poses)):
            dropout_rgbs = []
            for n in range(args.n_passes):
                dropout_rgb, _, _, _ = render(H, W, K, chunk=args.chunk, c2w=c2w[:3, :4], **render_kwargs)
                dropout_rgbs.append(dropout_rgb)
            dropout_rgbs = torch.stack(dropout_rgbs,0)
            uncert = torch.mean(torch.std(dropout_rgbs,dim=0),dim=-1)
            uncerts.append(uncert.cpu().numpy())
        extras['uncerts'] = np.array(uncerts)
        close_dropout(render_kwargs['network_fn'])

    n_batches = int(np.ceil(N_poses / batch_size))
    for i_batch in range(n_batches):
        start = int(i_batch*batch_size)
        end = int(i_batch*batch_size+batch_size)
        batch_poses = render_poses[start:end]
        if gt_imgs is not None:
            gt_imgs = gt_imgs[start:end]
        batch_rgbs, batch_disps, batch_accs, batch_extras = render_path(batch_poses, hwf, K,
                                                                        args.chunk, render_kwargs,gt_imgs=gt_imgs,savedir=savedir)
        rgbs.append(batch_rgbs)
        disps.append(batch_disps)
        accs.append(batch_accs)
        others.append(batch_extras)

        batch_alpha_all = torch.Tensor(batch_extras['alpha']).view(-1, batch_extras['alpha'].shape[-1])
        batch_accs_all = torch.Tensor(batch_accs).view(-1)
        # batch_entropy_ray_zvals = entropy_loss.ray_zvals_per_ray(batch_alpha_all, batch_accs_all)
        # batch_entropy_maps = batch_entropy_ray_zvals.view(batch_disps.shape).cpu().numpy()
        # entropy_maps.append(batch_entropy_maps)
    rgbs = np.concatenate(rgbs, 0)
    disps = np.concatenate(disps, 0)
    accs = np.concatenate(accs, 0)
    # entropy_maps = np.concatenate(entropy_maps, 0)
    for k in others[-1].keys():
        k_values = [extra[k] for extra in others]
        extras[k] = np.concatenate(k_values, 0)
    return rgbs, accs, disps, extras


def train():

    global entropy_loss
    parser = config_parser()
    args = parser.parse_args()
    logger = SummaryWriter(os.path.join(args.basedir, args.expname,'summaries'))

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])


        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    ### few-shot setting

    if args.train_scene is None:
        if args.fewshot > 0:
            np.random.seed(args.fewshot_seed)
            i_train = np.random.choice(i_train, args.fewshot, replace=False)
    else:
        i_train = np.array([i for i in args.train_scene if
                            (i not in i_test and i not in i_val)])

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _, _,_ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand

    if args.entropy:
        N_entropy = args.N_entropy
        entropy_loss = EntropyLoss(args)

    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        file.write('{} = {}\n'.format('train_scene', i_train))
        file.write('{} = {}\n'.format('test_scene', i_test))
        file.write('{} = {}\n'.format('val_scene', i_val))

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1

    if args.eval_only:
        N_iters = start + 2
        args.i_testset = 1

    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            if args.save_video:
                # Turn on testing mode
                with torch.no_grad():
                    rgbs, accs, disps, extras = batch_render(K, args, hwf,render_kwargs_test, render_poses)
                for j in range(disps.shape[0]):
                    disps[j] = (disps[j] / np.quantile(disps[j], 0.9)) * 0.8
                # entropy_maps = extras['entropys']
                print('Done, saving', rgbs.shape, disps.shape)
                moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
                imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps), fps=30, quality=8)
                imageio.mimwrite(moviebase + 'acc.mp4', to8b(accs), fps=30, quality=8)
                if args.mc_dropout:
                    uncerts_color = color_error_image_func()(torch.Tensor(extras['uncerts']))
                    uncerts_color = uncerts_color.cpu().numpy()
                    imageio.mimwrite(moviebase + 'uncert_color.mp4', to8b(uncerts_color), fps=30, quality=8)
            # imageio.mimwrite(moviebase + 'entropy.mp4', to8b(entropy_maps / np.nanmax(entropy_maps)), fps=30, quality=8)
            # imageio.mimwrite(moviebase + 'errors.mp4', to8b(errors.cpu().numpy()), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                #test_rgbs, test_disps, test_accs, test_extras = render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
                test_poses = torch.Tensor(poses[i_test]).to(device)
                test_rgbs, test_disps, test_accs, test_extras = batch_render(K, args, hwf,render_kwargs_test, test_poses, gt_imgs=images[i_test], savedir=testsavedir)

            test_loss = img2mse(torch.Tensor(test_rgbs), torch.Tensor(images[i_test]))
            test_psnr = mse2psnr(test_loss)
            test_ssim, test_msssim = img2ssim(torch.Tensor(test_rgbs), torch.Tensor(images[i_test]))
            # test_alpha_all = torch.Tensor(test_extras['alpha']).view(-1,test_extras['alpha'].shape[-1]) # 【N_rays. N_samples]
            test_accs_all = torch.Tensor(test_accs).view(-1)
            # test_entropy_ray_zvals = entropy_loss.ray_zvals_per_ray(test_alpha_all,test_accs_all)
            test_errors = (torch.Tensor(test_rgbs) - torch.Tensor(images[i_test]))**2
            test_errors = test_errors.mean(axis=-1)
            test_mse = test_errors.view(-1)
            test_jacobs = test_extras['jacobs']

            # coefficient = correlation_coefficient(test_entropy_ray_zvals,test_mse)

            logger.add_scalar('TEST/loss', test_loss, global_step)
            logger.add_scalar('TEST/psnr', test_psnr, global_step)
            logger.add_scalar('TEST/ssim', test_ssim, global_step)
            logger.add_scalar('TEST/ms_ssim', test_msssim, global_step)
            # logger.add_scalar('TEST/coefficient_entropy_errors',coefficient, global_step)

            handout_id = np.random.choice(test_rgbs.shape[0])
            # test_entropy_maps = test_entropy_ray_zvals.view(test_disps.shape).cpu()

            with torch.no_grad():
                test_errors_color = color_error_image_func()(test_errors.cpu(),torch.Tensor(images[i_test]).mean(axis=-1))
                test_errors_color = test_errors_color.cpu().numpy()
                test_jacobs_color = color_error_image_func()(torch.Tensor(test_extras['jacobs'])).cpu().numpy()
                # test_entropy_maps_color = color_error_image_func()(test_entropy_maps).cpu().numpy()

            logger.add_image('TEST/rgb', to8b(test_rgbs[handout_id]), global_step, dataformats='HWC')
            logger.add_image('TEST/disp', to8b((test_disps[handout_id] / np.quantile(test_disps[-1],0.9))*0.8), global_step, dataformats='HW')
            logger.add_image('TEST/acc', to8b(test_accs[handout_id]), global_step, dataformats='HW')
            logger.add_image('TEST/gt_image', to8b(images[i_test][handout_id].cpu().numpy()), global_step, dataformats='HWC')
            logger.add_image('TEST/err', to8b(test_errors_color[handout_id]), global_step, dataformats='HWC')

            # test_entropy_ray_zvals = test_entropy_ray_zvals.cpu()
            test_mse = test_mse.cpu()
            logger.add_histogram('TEST/errors',test_mse,global_step)
            if args.mc_dropout:
                test_uncerts = test_extras['uncerts']
                test_uncerts_color = color_error_image_func()(torch.Tensor(test_uncerts))
                test_uncerts_color = test_uncerts_color.cpu().numpy()
                test_sigma2 = test_uncerts.flatten()
                logger.add_image('TEST/uncert', to8b(test_uncerts_color[handout_id]), global_step, dataformats='HWC')
                logger.add_histogram('TEST/uncerts',test_sigma2, global_step)


            for n in range(len(i_test)):
                disp8 = to8b((test_disps[n]/np.quantile(test_disps[n],0.9))*0.8)
                # disp[i] = (disp[i] / np.quantile(disp[i], 0.9)) * 0.8
                disp_filename = os.path.join(testsavedir, 'disp_{:03d}.png'.format(n))
                imageio.imwrite(disp_filename, disp8)

                acc8 = to8b(test_accs[n])
                acc_filename = os.path.join(testsavedir, 'acc_{:03d}.png'.format(n))
                imageio.imwrite(acc_filename, acc8)

                gt8 = to8b(images[i_test][n].cpu().numpy())
                gt_filename = os.path.join(testsavedir, 'gt_{:03d}.png'.format(n))
                imageio.imwrite(gt_filename, gt8)

                err8 = to8b(test_errors_color[n])
                err_filename = os.path.join(testsavedir, 'err_{:03d}.png'.format(n))
                imageio.imwrite(err_filename, err8)

                jacob8 = to8b(test_jacobs_color[n])
                jacob_filename = os.path.join(testsavedir, 'jacob_{:03d}.png'.format(n))
                imageio.imwrite(jacob_filename, jacob8)

                # entropy8 = to8b(test_entropy_maps_color[n])
                # entropy_filename = os.path.join(testsavedir, 'entropy_{:03d}.png'.format(n))
                # imageio.imwrite(entropy_filename, entropy8)
                if args.mc_dropout:
                    uncet8 = to8b(test_uncerts_color[n])
                    uncert_filename = os.path.join(testsavedir, 'uncert_{:03d}.png'.format(n))
                    imageio.imwrite(uncert_filename, uncet8)

            if args.eval_only:
                outputsavedir = os.path.join(testsavedir, 'rawoutput')
                os.makedirs(outputsavedir, exist_ok=True)
                np.save(os.path.join(outputsavedir,'test_rgbs'),test_rgbs)
                np.save(os.path.join(outputsavedir,'test_disps'), test_disps)
                np.save(os.path.join(outputsavedir,'test_errors'),test_errors.cpu().numpy())
                np.save(os.path.join(outputsavedir, 'test_jacobs'), test_jacobs)
                if args.mc_dropout:
                    np.save(os.path.join(outputsavedir,'test_uncerts'),test_uncerts)

            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} Time: {dt}")
            logger.add_scalar('TRAIN/loss', loss, global_step)
            logger.add_scalar('TRAIN/psnr', psnr,global_step)
            if args.N_importance > 0:
                logger.add_scalar('TRAIN/psnr0', psnr0,global_step)

            if i % args.i_img == 0:
                img_i = np.random.choice(i_val)
                target = images[img_i][None,...]
                pose = torch.Tensor(poses[img_i])[None,...]
                with torch.no_grad():
                    rgb, disp, acc,_ = render_path(pose.to(device), hwf, K, args.chunk, render_kwargs_train,gt_imgs=target)


                psnr = mse2psnr(img2mse(torch.Tensor(rgb), torch.Tensor(target)))
                ssim,ms_ssim = img2ssim(torch.Tensor(rgb),torch.Tensor(target))

                err = (torch.Tensor(rgb) - torch.Tensor(target)) ** 2
                err = err.mean(axis=-1)
                with torch.no_grad():
                    err = color_error_image_func()(err.cpu(),torch.Tensor(target).mean(axis=-1))
                    err = err.cpu().numpy()

                logger.add_image('TRAIN/rgb', to8b(rgb[-1]),global_step,dataformats='HWC')
                logger.add_image('TRAIN/disp', to8b((disp[-1]/np.quantile(disp[-1],0.9))*0.8),global_step,dataformats='HW')
                logger.add_image('TRAIN/acc', to8b(acc[-1]),global_step,dataformats='HW')
                logger.add_image('TRAIN/gt_image',to8b(target[-1].cpu().numpy()),global_step,dataformats='HWC')
                logger.add_image('TRAIN/err', to8b(err[-1]), global_step, dataformats='HWC')

                logger.add_scalar('TRAIN/psnr_holdout', psnr,global_step)
                logger.add_scalar('TRAIN/ssim_holdout', ssim, global_step)
                logger.add_scalar('TRAIN/ms_ssim_holdout', ms_ssim, global_step)


        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)

            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1






if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
