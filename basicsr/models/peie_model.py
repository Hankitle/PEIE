import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.metrics import calculate_metric
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class PEIEModel(BaseModel):
    def __init__(self, opt):
        super(PEIEModel, self).__init__(opt)

        # define network net_g
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        # load weights
        load_path_g = self.opt['path'].get('pretrain_network_g', None)
        if load_path_g is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path_g, self.opt['path'].get('strict_load_g', True), param_key)
            
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']

        # define network net_d
        if 'network_d' in self.opt:
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)

            # load weights
            load_path_d = self.opt['path'].get('pretrain_network_d', None)
            if load_path_d is not None:
                param_key = self.opt['path'].get('param_key_d', 'params')
                self.load_network(self.net_d, load_path_d, self.opt['path'].get('strict_load_d', True), param_key)
        
        if 'network_eval' in self.opt:
            self.net_eval = build_network(self.opt['network_eval'])
            self.net_eval = self.model_to_device(self.net_eval)
            self.print_network(self.net_eval)

            # load weights
            load_path_eval = self.opt['path'].get('pretrain_network_eval', None)
            if load_path_eval is not None:
                param_key = self.opt['path'].get('param_key_eval', 'params')
                self.load_network(self.net_eval, load_path_eval, self.opt['path'].get('strict_load_eval', True), param_key)

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        self.net_g.train()
        if hasattr(self, 'net_d'):
            self.net_d.train()
        if hasattr(self, 'net_eval'):
            self.net_eval.eval()
            
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if hasattr(self, 'net_d') and train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None
        
        if train_opt.get('contrastperceptual_opt'):
            self.cri_contrast_perceptual = build_loss(train_opt['contrastperceptual_opt']).to(self.device)
        else:
            self.cri_contrast_perceptual = None
        
        if train_opt.get('daclip_opt'):
            self.cri_da_clip = build_loss(train_opt['daclip_opt']).to(self.device)
        else:
            self.cri_da_clip = None
        
        if train_opt.get('myloss_opt'):
            self.cri_myloss = build_loss(train_opt['myloss_opt']).to(self.device)
        else:
            self.cri_myloss = None
        
        if train_opt.get('progressive_opt'):
            self.cri_progressive = build_loss(train_opt['progressive_opt']).to(self.device)
        else:
            self.cri_progressive = None

        if hasattr(self, 'net_d'):
            self.net_d_iters = train_opt.get('net_d_iters', 1)
            self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    
    def setup_optimizers(self):
        train_opt = self.opt['train']

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d   
        if hasattr(self, 'net_d'):
            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        if 'lq' in data:
            self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):    
        # 1) Update generator with discriminator frozen.
        if hasattr(self, 'net_d'):
            for p in self.net_d.parameters():
                p.requires_grad = False    

        self.optimizer_g.zero_grad()
        if hasattr(self, 'lq'):
            self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        
        if (not hasattr(self, 'net_d')) or (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix 

            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, _ = self.cri_perceptual(self.output, self.gt)
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
                
            # gan loss
            if hasattr(self, 'net_d'):
                if isinstance(self.output, list): 
                    fake_g_pred1 = self.net_d(self.output[0])
                    l_g_gan = self.cri_gan(fake_g_pred1, True, is_disc=False)
                    fake_g_pred2 = self.net_d(self.output[1])
                    l_g_gan += self.cri_gan(fake_g_pred2, True, is_disc=False)
                    loss_dict['l_g_gan'] = l_g_gan
                    l_g_total += l_g_gan
                else:
                    fake_g_pred = self.net_d(self.output)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    loss_dict['l_g_gan'] = l_g_gan
                    l_g_total += l_g_gan
            
            l_g_total.backward()
            self.optimizer_g.step()

        # 2) Update discriminator with generator outputs detached.
        if hasattr(self, 'net_d'):
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()

            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real    
            loss_dict['out_d_real'] = torch.sigmoid(real_d_pred.detach()).mean()
            l_d_real.backward()

            if isinstance(self.output, list): 
                fake_g_pred1 = self.net_d(self.output[0].detach())
                l_d_fake = self.cri_gan(fake_g_pred1, False, is_disc=True)
                fake_g_pred2 = self.net_d(self.output[1].detach())
                l_d_fake += self.cri_gan(fake_g_pred2, False, is_disc=True)
                loss_dict['l_d_fake'] = l_d_fake
                loss_dict['out_d_fake'] = torch.sigmoid(((fake_g_pred1 + fake_g_pred2)/2).detach()).mean()
                l_d_fake.backward()
            else:
                fake_g_pred = self.net_d(self.output.detach())
                l_d_fake = self.cri_gan(fake_g_pred, False, is_disc=True)
                loss_dict['l_d_fake'] = l_d_fake
                loss_dict['out_d_fake'] = torch.sigmoid(fake_g_pred.detach()).mean()
                l_d_fake.backward()

            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    @torch.no_grad()
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)

        metric_data = dict()

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            if use_pbar:
                pbar.set_description(f'Test {img_name}')
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
                save_img_path2 = None
                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name, img_name, f'{img_name}_{current_iter}.png')
                    save_img_path2 = osp.join(
                        self.opt['path']['visualization'], f'{current_iter}', dataset_name, f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, f'{img_name}_{self.opt["name"]}.png')
                if save_img_path2:
                    imwrite(sr_img, save_img_path2)
                    metric_data['path'] = save_img_path2
                else:
                    imwrite(sr_img, save_img_path)
                    metric_data['path'] = save_img_path

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
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
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        if hasattr(self, 'net_d'):
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)