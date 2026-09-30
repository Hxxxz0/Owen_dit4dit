"""Run the upstream StarVLA trainer and record actual updates on every rank."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from starVLA.training import train_starvla as upstream
from starVLA.model.framework.share_tools import apply_config_compat


class VerifiedTrainer(upstream.VLATrainer):
    def prepare_training(self):
        if self.config.trainer.get('gradient_checkpointing', False):
            backbone = self.model.qwen_vl_interface.model
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            backbone.config.use_cache = False
        super().prepare_training()
        # Hugging Face loads the backbone in eval mode; enable its checkpointed
        # training path explicitly (the upstream loop does not call train()).
        self.model.train()
        assert self.accelerator.num_processes == 16
        self.validation_steps = []
        self.validation_params = {}
        model = self.accelerator.unwrap_model(self.model)
        for name, param in model.named_parameters():
            group = 'backbone' if 'qwen_vl_interface' in name and 'q_proj.weight' in name else (
                'action_head' if 'action_model' in name and 'fc1.weight' in name else None)
            if group and group not in self.validation_params and param.requires_grad:
                self.validation_params[group] = (name, param, param.detach().clone())
        assert set(self.validation_params) == {'backbone', 'action_head'}

    def _train_step(self, batch_vla, batch_vlm=None):
        started = time.perf_counter()
        metrics = super()._train_step(batch_vla, batch_vlm)
        loss = metrics['action_dit_loss']
        assert math.isfinite(loss), f'Non-finite loss on rank {dist.get_rank()}'
        self.validation_steps.append({'step': len(self.validation_steps)+1, 'loss': loss,
                                      'seconds': time.perf_counter()-started,
                                      'instruction': batch_vla[0]['lang']})
        print(f'PPU_TRAIN rank={dist.get_rank()} step={len(self.validation_steps)} loss={loss:.6f}', flush=True)
        return metrics

    def _finalize_training(self):
        record = {'rank': dist.get_rank(), 'device': torch.cuda.get_device_name(),
                  'peak_allocated_GiB': torch.cuda.max_memory_allocated()/1024**3,
                  'steps': self.validation_steps, 'parameter_updates': {}}
        for group, (name, param, before) in self.validation_params.items():
            difference = (param.detach().float()-before.float()).abs().max().item()
            assert difference > 0 and math.isfinite(difference), f'{group} did not update on rank {dist.get_rank()}'
            record['parameter_updates'][group] = {'name': name, 'max_abs_change': difference}
        records = [None]*dist.get_world_size()
        dist.all_gather_object(records,record)
        if self.accelerator.is_main_process:
            (Path(self.config.output_dir)/'ppu_rank_training_validation.json').write_text(json.dumps(records,indent=2))
        super()._finalize_training()
        if self.accelerator.is_main_process:
            print('PPU_TRAIN_VERIFIED: all 16 ranks completed real LIBERO training, updated backbone and head, saved final checkpoint',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_yaml',required=True)
    args,overrides=parser.parse_known_args()
    cfg=OmegaConf.merge(OmegaConf.load(args.config_yaml),OmegaConf.from_dotlist(upstream.normalize_dotlist_args(overrides)))
    cfg=apply_config_compat(cfg)
    cfg.config_yaml=args.config_yaml
    upstream.VLATrainer=VerifiedTrainer
    upstream.main(cfg)
