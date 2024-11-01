import os, sys
import torch
import numpy as np
import torch.multiprocessing
import logging
from tqdm import tqdm
from metrics import evaluate_model_uncond
from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from models.model import ImagenTime
from models.sampler import DiffusionProcess
from utils.utils import save_checkpoint, restore_state, create_model_name_and_dir, print_model_params, \
    log_config_and_tags
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_uncond

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    # model name and directory
    name = create_model_name_and_dir(args)

    # log args
    logging.info(args)

    # set-up neptune logger. switch to your desired logger
    with CompositeLogger([NeptuneLogger()]) if args.neptune \
            else PrintLogger() as logger:

        # log config and tags
        log_config_and_tags(args, logger, name)

        # set-up data and device
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader = gen_dataloader(args)
        logging.info(args.dataset + ' dataset is ready.')

        model = ImagenTime(args=args, device=args.device).to(args.device)
        if args.use_stft:
            model.init_stft_embedder(train_loader)

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        state = dict(model=model, epoch=0)
        init_epoch = 0

        # restore checkpoint
        if args.resume:
            ema_model = model.model_ema if args.ema else None # load ema model if available
            init_epoch = restore_state(args, state, ema_model=ema_model)

        # print model parameters
        print_model_params(logger, model)

        # --- train model ---
        logging.info(f"Continuing training loop from epoch {init_epoch}.")
        best_score = float('inf')  # marginal score for long-range metrics, dice score for short-range metrics
        for epoch in range(init_epoch, args.epochs):
            model.train()
            model.epoch = epoch
            logger.log_name_params('train/epoch', epoch)

            # --- train loop ---
            for i, data in enumerate(train_loader, 1):
                x_ts = data[0].to(args.device)
                x_img = model.ts_to_img(x_ts)
                optimizer.zero_grad()
                loss = model.loss_fn(x_img)
                if len(loss) == 2:
                    loss, to_log = loss
                    for key, value in to_log.items():
                        logger.log(f'train/{key}', value, epoch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                model.on_train_batch_end()

            # --- evaluation loop ---
            if epoch % args.logging_iter == 0:
                gen_sig = []
                real_sig = []
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        process = DiffusionProcess(args, model.net,
                                                   (args.input_channels, args.img_resolution, args.img_resolution))
                        for data in tqdm(test_loader):
                            # sample from the model
                            x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                            # --- convert to time series --
                            x_ts = model.img_to_ts(x_img_sampled)

                            # special case for temperature_rain dataset
                            if args.dataset in ['temperature_rain']:
                                x_ts = torch.clamp(x_ts, 0, 1)

                            gen_sig.append(x_ts.detach().cpu().numpy())
                            real_sig.append(data[0].detach().cpu().numpy())

                gen_sig = np.vstack(gen_sig)
                real_sig = np.vstack(real_sig)
                scores = evaluate_model_uncond(real_sig, gen_sig, args)
                for key, value in scores.items():
                    logger.log(f'test/{key}', value, epoch)

                # --- save checkpoint ---
                curr_score = scores['marginal_score_mean'] if 'marginal_score_mean' in scores else scores['disc_mean']
                if curr_score < best_score:
                    best_score = curr_score
                    ema_model = model.model_ema if args.ema else None
                    save_checkpoint(args.log_dir, state, epoch , ema_model)

        logging.info("Training is complete")


if __name__ == '__main__':
    args = parse_args_uncond()  # parse unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
