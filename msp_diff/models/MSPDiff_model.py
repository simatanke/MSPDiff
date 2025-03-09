import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from msp_diff.utils.base_model import BaseModel
from torch.nn import functional as F
from functools import partial
import numpy as np
from msp_diff.utils.beta_schedule import make_beta_schedule, default
import sys
from torch.nn.parallel import DataParallel, DistributedDataParallel

from msp_diff.utils.loss import init_loss
from ldm.ddpm import DDPM


@MODEL_REGISTRY.register()
class MSPDIff(BaseModel):
    """MSPDiff model for train."""

    def __init__(self, opt):
        super(MSPDIff, self).__init__(opt)

        # define network

        self.unet = build_network(opt['network_unet'])
        self.unet = self.model_to_device(self.unet)
        opt['network_ddpm']['denoise_fn'] = self.unet

        self.feature_extraction = build_network(opt['network_feature_extraction'])
        self.feature_extraction = self.model_to_device(self.feature_extraction)
        opt['network_ddpm']['color_fn'] = self.feature_extraction

        self.ddpm = build_network(opt['network_ddpm'])
        self.ddpm = self.model_to_device(self.ddpm)
        if isinstance(self.ddpm, (DataParallel, DistributedDataParallel)):
            self.bare_model = self.ddpm.module
        else:
            self.bare_model = self.ddpm

        self.bare_model.set_new_noise_schedule(schedule_opt=opt['ddpm_schedule'],
                                               device=self.device)
        self.bare_model.set_loss(device=self.device)
        self.print_network(self.ddpm)


        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_le', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_le', 'params')
            self.load_network(self.net_le, load_path, self.opt['path'].get('strict_load_le', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_g1', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g1', 'params')
            self.load_network(self.net_le, load_path, self.opt['path'].get('strict_load_g1', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.ddpm, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if 'metrics' in self.opt['val'] and 'lpips' in self.opt['val']['metrics']:
            import lpips
            self.lpips = lpips.LPIPS(net='alex')
            self.lpips = self.model_to_device(self.lpips)
            if isinstance(self.lpips, (DataParallel, DistributedDataParallel)):
                self.lpips_bare_model = self.lpips.module
            else:
                self.lpips_bare_model = self.lpips

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.ddpm.train()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # ----------- optimizer g ----------- #
        net_g_reg_ratio = 1
        normal_params = []
        logger = get_root_logger()
        for _, param in self.ddpm.named_parameters():
            if self.opt['train'].get('frozen_denoise', False):
                if 'denoise' in _:
                    logger.info(f'frozen {_}')
                    continue
            normal_params.append(param)
        optim_params_g = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_g']['lr']
        }]
        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g']['lr'] * net_g_reg_ratio
        betas = (0**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr, betas=betas)
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        # if self.opt['train'].get('mask_loss', False):
        #     assert self.opt['train'].get('cal_noise_only', False), "mask_loss can only used with cal_noise_only, now"
        # optimize net_g
        assert 'ddpm_cs' in self.opt['train'].get('train_type', None), "train_type must be ddpm_cs"
        self.optimizer_g.zero_grad()
        pred_noise, noise, x_recon_cs, x_start, t, color_scale = self.ddpm(self.gt, self.lq,
                                                                           train_type=self.opt['train'].get(
                                                                               'train_type', None),
                                                                           different_t_in_one_batch=self.opt[
                                                                               'train'].get('different_t_in_one_batch',
                                                                                            None),
                                                                           t_sample_type=self.opt['train'].get(
                                                                               't_sample_type', None),
                                                                           pred_type=self.opt['train'].get('pred_type',
                                                                                                           None),
                                                                           clip_noise=self.opt['train'].get(
                                                                               'clip_noise', None),
                                                                           color_shift=self.opt['train'].get(
                                                                               'color_shift', None),
                                                                           color_shift_with_schedule=self.opt[
                                                                               'train'].get('color_shift_with_schedule',
                                                                                            None),
                                                                           t_range=self.opt['train'].get('t_range',
                                                                                                         None),
                                                                           cs_on_shift=self.opt['train'].get(
                                                                               'cs_on_shift', None),
                                                                           cs_shift_range=self.opt['train'].get(
                                                                               'cs_shift_range', None),
                                                                           t_border=self.opt['train'].get('t_border',
                                                                                                          None),
                                                                           down_uniform=self.opt['train'].get(
                                                                               'down_uniform', False),
                                                                           down_hw_split=self.opt['train'].get(
                                                                               'down_hw_split', False),
                                                                           pad_after_crop=self.opt['train'].get(
                                                                               'pad_after_crop', False),
                                                                           input_mode=self.opt['train'].get(
                                                                               'input_mode', None),
                                                                           crop_size=self.opt['train'].get('crop_size',
                                                                                                           None),
                                                                           divide=self.opt['train'].get('divide', None),
                                                                           frozen_denoise=self.opt['train'].get(
                                                                               'frozen_denoise', None),
                                                                           cs_independent=self.opt['train'].get(
                                                                               'cs_independent', None),
                                                                           shift_x_recon_detach=self.opt['train'].get(
                                                                               'shift_x_recon_detach', None))
        if self.opt['train'].get('vis_train', False) and current_iter <= self.opt['train'].get('vis_num', 100) and \
                self.opt['rank'] == 0:
            '''
            When the parameter 'vis_train' is set to True, the training process will be visualized. 
            The value of 'vis_num' corresponds to the number of visualizations to be generated.
            '''
            save_img_path = osp.join(self.opt['path']['visualization'], 'train',
                                     f'{current_iter}_noise_level_{self.bare_model.t}.png')
            x_recon_print = tensor2img(self.bare_model.x_recon, min_max=(-1, 1))
            noise_print = tensor2img(self.bare_model.noise, min_max=(-1, 1))
            pred_noise_print = tensor2img(self.bare_model.pred_noise, min_max=(-1, 1))
            x_start_print = tensor2img(self.bare_model.x_start, min_max=(-1, 1))
            x_noisy_print = tensor2img(self.bare_model.x_noisy, min_max=(-1, 1))

            img_print = np.concatenate([x_start_print, noise_print, x_noisy_print, pred_noise_print, x_recon_print],
                                       axis=0)
            imwrite(img_print, save_img_path)
        l_g_total = 0
        loss_dict = OrderedDict()

        l_g_x0 = F.l1_loss(x_recon_cs, x_start) * self.opt['train'].get('l_g_x0_w', 1.0)
        if self.opt['train'].get('gamma_limit_train', None) and color_scale <= self.opt['train'].get(
                'gamma_limit_train', None):
            l_g_x0 = l_g_x0 * 1e-12
        loss_dict['l_g_x0'] = l_g_x0
        l_g_total += l_g_x0

        if not self.opt['train'].get('frozen_denoise', False):
            l_g_noise = F.l1_loss(pred_noise, noise)
            loss_dict['l_g_noise'] = l_g_noise
            l_g_total += l_g_noise

        l_g_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        scale = self.opt.get('scale', 1)
        window_size = 8
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        gt = F.pad(self.gt, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        if hasattr(self, 'net_g_ema'):
            print("TODO: wrong")
        else:
            with torch.no_grad():
                self.bare_model.denoise_fn.eval()

                self.output = self.bare_model.ddim_pyramid_sample(img,
                                                                  schedule_list=self.opt['val'].get('schedule_list'),
                                                                  continous=self.opt['val'].get('ret_process', False),
                                                                  ddim_timesteps=self.opt['val'].get('ddim_timesteps',
                                                                                                     50),
                                                                  return_pred_noise=self.opt['val'].get(
                                                                      'return_pred_noise', False),
                                                                  return_x_recon=self.opt['val'].get('ret_x_recon',
                                                                                                     False),
                                                                  ddim_discr_method=self.opt['val'].get(
                                                                      'ddim_discr_method', 'uniform'),
                                                                  ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                                  pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                                  clip_noise=self.opt['val'].get('clip_noise', False),
                                                                  save_noise=self.opt['val'].get('save_noise', False),
                                                                  color_gamma=self.opt['val'].get('color_gamma', None),
                                                                  color_times=self.opt['val'].get('color_times', 1),
                                                                  return_all=self.opt['val'].get('ret_all', False),
                                                                  fine_diffV2=self.opt['val'].get('fine_diffV2', False),
                                                                  fine_diffV2_st=self.opt['val'].get('fine_diffV2_st',
                                                                                                     200),
                                                                  fine_diffV2_num_timesteps=self.opt['val'].get(
                                                                      'fine_diffV2_num_timesteps', 20),
                                                                  do_some_global_deg=self.opt['val'].get(
                                                                      'do_some_global_deg', False),
                                                                  use_up_v2=self.opt['val'].get('use_up_v2', False))
                self.bare_model.denoise_fn.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')

                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            print("TODO")
        else:
            # self.save_network(self.net_g, 'net_g', current_iter)
            self.save_network(self.ddpm, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
