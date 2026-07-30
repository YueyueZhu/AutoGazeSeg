import os

import torch
import torch.nn as nn

try:
    import wandb
except ImportError:
    wandb = None


class BaseTrainer:
    def __init__(
        self,
        args,
        logger,
        model,
        optimizer,
        criterion,
        train_dataloader=None,
        test_dataloader=None,
    ):
        self.args = args
        self.logger = logger

        if isinstance(model, (list, tuple)):
            self.model = list(model)
            for i in range(len(self.model)):
                if isinstance(self.model[i], nn.Module):
                    self.model[i] = self.model[i].cuda()
                    if torch.cuda.device_count() > 1:
                        self.model[i] = nn.DataParallel(self.model[i])
            self.optimizer = list(optimizer) if isinstance(optimizer, tuple) else optimizer
        else:
            self.model = model
            if isinstance(self.model, nn.Module):
                self.model = self.model.cuda()
                if torch.cuda.device_count() > 1:
                    self.logger.info(f"Use {torch.cuda.device_count()} GPUs")
                    self.model = nn.DataParallel(self.model)
            self.optimizer = optimizer

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.criterion = criterion

        self.init_metrics()

        self.epoch = 1
        self.iteration = 1

        if self.train_dataloader is not None:
            if len(self.train_dataloader) == 0:
                raise ValueError("The training dataloader is empty.")
            self.total_epoch = self.args.max_ite // len(self.train_dataloader) + 1
        else:
            self.total_epoch = 0

        if args.fp16 and self.train_dataloader is not None:
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):
                self.scaler = torch.cuda.amp.GradScaler()

        if args.resume:
            self.resume_configure(args.ckpt_path)

        if args.wandb:
            if wandb is None:
                raise ImportError("Install wandb to enable --wandb logging.")
            wandb.init(project="AutoGazeSeg", reinit=True, name=args.run_id)
            for report_mode in ["train", "val", "test"]:
                wandb.define_metric(f"{report_mode}/iteration")
                wandb.define_metric(f"{report_mode}/*", step_metric=f"{report_mode}/iteration")

    def _update(self, minibatch):
        raise NotImplementedError

    def validate(self, dataloader, model=None, save_pred=False, save_root=None):
        raise NotImplementedError

    def run(self):
        if self.train_dataloader is None:
            raise ValueError("A training dataloader is required to call run().")

        dataloader_iter = iter(self.train_dataloader)
        self._epoch_begin_hook()
        while self.iteration <= self.args.max_ite:
            try:
                minibatch = next(dataloader_iter)
            except StopIteration:
                self._epoch_end_hook()
                self.epoch += 1
                dataloader_iter = iter(self.train_dataloader)
                try:
                    minibatch = next(dataloader_iter)
                except StopIteration as error:
                    raise RuntimeError("The training dataloader yielded no batches.") from error
                self._epoch_begin_hook()

            loss = self._update(minibatch)

            if self.iteration % self.args.log_step == 0:
                self.report_progress(progress_dict=loss, mode="train", use_wandb=self.args.wandb)

            if (
                self.test_dataloader is not None
                and self.args.val_step > 0
                and self.iteration % self.args.val_step == 0
            ):
                self._validate_and_track_best(mode="test", use_wandb=self.args.wandb)

            self.iteration += 1

        if self.test_dataloader is not None:
            self._validate_and_track_best(mode="latest", use_wandb=False)
            if self.best_performance_dict is not None:
                self.report_progress_result(
                    progress_dict=self.best_performance_dict,
                    mode="best",
                    use_wandb=False,
                )

        checkpoint_path = self.best_checkpoint_path
        if not os.path.exists(checkpoint_path):
            self.save(checkpoint_path)

    @property
    def best_checkpoint_path(self):
        """Canonical checkpoint consumed by the train-then-test scripts."""
        return os.path.join(self.args.params_path, f"{self.args.experiment_name}.pth")

    def _validate_and_track_best(self, mode, use_wandb):
        performance_dict = self.validate(
            self.test_dataloader,
            save_pred=self.args.save_pred,
            save_root=os.path.join(self.args.exp_result_path, self.args.experiment_name),
        )
        self.report_progress_result(performance_dict, mode=mode, use_wandb=use_wandb)

        if self.main_metric is None:
            return performance_dict
        if self.main_metric not in performance_dict:
            raise KeyError(f"Validation did not return the main metric: {self.main_metric}")

        if self.best_iteration is None or performance_dict[self.main_metric] > self.best_performance:
            self.best_iteration = self.iteration
            self.best_performance = performance_dict[self.main_metric]
            self.best_performance_dict = performance_dict
            self.save(self.best_checkpoint_path)

        if self.best_performance_dict is not None and mode != "latest":
            self.report_progress_result(self.best_performance_dict, mode="best", use_wandb=False)
        return performance_dict

    def report_progress(self, progress_dict, mode, use_wandb=False):
        self.logger.info(
            "[{} | Epoch {} Ite {}] {}".format(
                mode,
                self.epoch,
                self.iteration,
                ", ".join("{}: {}".format(k, v) for k, v in progress_dict.items()),
            )
        )

        if use_wandb:
            wandb_dict = {f"{mode}/iteration": self.iteration}

            for k, v in progress_dict.items():
                wandb_dict[f"{mode}/{k}"] = v

            wandb.log(wandb_dict)

    def report_progress_result(self, progress_dict, mode, use_wandb=False):
        self.logger.info(
            "[{}] {}".format(
                mode,
                ", ".join("{}: {}".format(k, v) for k, v in progress_dict.items()),
            )
        )
        if mode == "best":
            self.logger.info("#####################################################################################")
        if use_wandb:
            wandb_dict = {f"{mode}/iteration": self.iteration}
            wandb_dict.update({f"{mode}/{key}": value for key, value in progress_dict.items()})
            wandb.log(wandb_dict)

    def init_metrics(self):
        self.best_performance = 0
        self.best_iteration = None
        self.best_performance_dict = None

        self.main_metric = None

    def resume_configure(self, ckpt_path):
        ckpt_path = self._resolve_checkpoint_path(ckpt_path)
        checkpoint = self._load_checkpoint(ckpt_path)
        self._load_model_states(checkpoint)

        optimizer_states = checkpoint.get("opt_state_dicts") if isinstance(checkpoint, dict) else None
        if optimizer_states is None and isinstance(checkpoint, dict) and "opt" in checkpoint:
            optimizer_states = checkpoint["opt"]
        if isinstance(optimizer_states, dict):
            optimizer_states = [optimizer_states]

        optimizers = self.optimizer if isinstance(self.optimizer, list) else [self.optimizer]
        if optimizer_states is None or len(optimizer_states) != len(optimizers):
            raise ValueError("The resume checkpoint does not contain matching optimizer states.")
        for optimizer, state_dict in zip(optimizers, optimizer_states):
            optimizer.load_state_dict(state_dict)

        self.epoch = checkpoint.get("epoch", self.epoch)
        self.iteration = checkpoint.get("iteration", self.iteration)
        self.best_performance = checkpoint.get("best_performance", self.best_performance)
        self.best_iteration = checkpoint.get("best_iteration", self.best_iteration)
        self.best_performance_dict = checkpoint.get("best_performance_dict", self.best_performance_dict)

        self.logger.info(f"Resume training from iteration {self.iteration}.")

    def load(self, ckpt_path, target_model=None):
        ckpt_path = self._resolve_checkpoint_path(ckpt_path)
        checkpoint = self._load_checkpoint(ckpt_path)
        self._load_model_states(checkpoint, target_model=target_model)
        self.logger.info(f"Loaded pretrained checkpoint: {ckpt_path}")

    def load_for_eval(self, ckpt_path):
        ckpt_path = self._resolve_checkpoint_path(ckpt_path)
        checkpoint = self._load_checkpoint(ckpt_path)
        self._load_model_states(checkpoint)
        for model in self._models():
            model.eval()

        self.logger.info(f"Loaded checkpoint for evaluation: {ckpt_path}")

    def _resolve_checkpoint_path(self, ckpt_path):
        if ckpt_path is None:
            raise ValueError("--ckpt_path is required for loading or resuming.")
        if os.path.isdir(ckpt_path):
            candidates = [
                os.path.join(ckpt_path, f"{self.args.experiment_name}.pth"),
                os.path.join(ckpt_path, "model_best.pth"),
                os.path.join(ckpt_path, "model_latest.pth"),
            ]
            resolved = next((path for path in candidates if os.path.isfile(path)), None)
            if resolved is None:
                raise FileNotFoundError(f"No checkpoint found. Checked: {candidates}")
            return resolved
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    @staticmethod
    def _load_checkpoint(ckpt_path):
        try:
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(ckpt_path, map_location="cpu")

    def _models(self, target_model=None):
        selected = self.model if target_model is None else target_model
        return list(selected) if isinstance(selected, (list, tuple)) else [selected]

    @staticmethod
    def _is_raw_state_dict(value):
        return isinstance(value, dict) and bool(value) and all(
            torch.is_tensor(item) or isinstance(item, nn.Parameter) for item in value.values()
        )

    def _extract_model_states(self, checkpoint):
        if isinstance(checkpoint, (list, tuple)) and all(isinstance(item, dict) for item in checkpoint):
            return list(checkpoint)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

        states = checkpoint.get("model_state_dicts")
        if states is None:
            states = checkpoint.get("branch_state_dicts")
        if states is None and "state_dict" in checkpoint:
            states = checkpoint["state_dict"]
        if states is None and "model_state_dict" in checkpoint:
            states = checkpoint["model_state_dict"]
        if states is None and self._is_raw_state_dict(checkpoint):
            states = checkpoint

        if isinstance(states, dict):
            states = [states]
        if not isinstance(states, (list, tuple)) or not all(isinstance(item, dict) for item in states):
            raise KeyError("Checkpoint has no supported model state dictionary.")
        return list(states)

    @staticmethod
    def _strip_data_parallel_prefix(state_dict):
        if state_dict and all(key.startswith("module.") for key in state_dict):
            return {key[len("module.") :]: value for key, value in state_dict.items()}
        return state_dict

    def _load_model_states(self, checkpoint, target_model=None):
        models = self._models(target_model)
        model_states = self._extract_model_states(checkpoint)
        if len(model_states) != len(models):
            raise ValueError(
                f"Checkpoint has {len(model_states)} model state(s), "
                f"but AutoGazeSeg uses {len(models)} branch(es)."
            )

        for model, state_dict in zip(models, model_states):
            target = model.module if isinstance(model, nn.DataParallel) else model
            target.load_state_dict(self._strip_data_parallel_prefix(state_dict), strict=True)

    def save(self, path, model=None):
        if model is None:
            net = self.model
            opt = self.optimizer
        else:
            net = model
            opt = self.optimizer

        if isinstance(net, (list, tuple)):
            model_state_dicts = []
            for sub_net in net:
                if isinstance(sub_net, nn.DataParallel):
                    model_state_dicts.append(sub_net.module.state_dict())
                else:
                    model_state_dicts.append(sub_net.state_dict())
        else:
            if isinstance(net, nn.DataParallel):
                model_state_dicts = [net.module.state_dict()]
            else:
                model_state_dicts = [net.state_dict()]

        if isinstance(opt, (list, tuple)):
            opt_state_dicts = [o.state_dict() for o in opt]
        else:
            opt_state_dicts = [opt.state_dict()]

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model_state_dicts": model_state_dicts,
                "opt_state_dicts": opt_state_dicts,
                "epoch": self.epoch,
                "iteration": self.iteration,
                "best_performance": self.best_performance,
                "best_iteration": self.best_iteration,
                "best_performance_dict": self.best_performance_dict,
            },
            path,
        )
        self.logger.info(f"Saved checkpoint to {path}")

    def _epoch_begin_hook(self):
        pass

    def _epoch_end_hook(self):
        pass
