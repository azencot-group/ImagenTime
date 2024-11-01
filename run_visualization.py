import numpy as np
import os, sys
import torch
from utils.loggers import CompositeLogger, NeptuneLogger, PrintLogger
from utils.utils_args import parse_args_uncond
from models.model import ImagenTime
from models.sampler import DiffusionProcess
import logging
from utils.utils_data import gen_dataloader
from utils.utils import create_model_name_and_dir, restore_state, log_config_and_tags
from utils.utils_vis import prepare_data, PCA_plot, TSNE_plot, density_plot, jensen_shannon_divergence
import matplotlib
from tqdm import tqdm

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# matplotlib.use('Agg')


def main(args):
    with CompositeLogger([NeptuneLogger()]) if args.neptune \
            else PrintLogger() as logger:

        name = create_model_name_and_dir(args)
        log_config_and_tags(args, logger, name)
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader = gen_dataloader(args) # combine train and test loaders?
        model = ImagenTime(args=args, device=args.device).to(args.device)
        if args.use_stft:
            model.init_stft_embedder(train_loader)
        else:
            _ = model.ts_to_img(next(iter(train_loader))[0].to(args.device)) # initialize delay embedder

        # restore checkpoint
        state = dict(model=model, epoch=0)
        ema_model = model.model_ema if args.ema else None
        restore_state(args, state, ema_model=ema_model)

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
        ori_sig = np.vstack(real_sig)
        logging.info("Data generation is complete")
        prep_ori, prep_gen, sample_num = prepare_data(ori_sig, gen_sig)

        # PCA Analysis
        PCA_plot(prep_ori, prep_gen, sample_num, logger, args)
        # Do t-SNE Analysis together
        TSNE_plot(prep_ori, prep_gen, sample_num, logger, args)
        # Density plot
        density_plot(prep_ori, prep_gen, logger, args)
        # jensen shannon divergence
        jensen_shannon_divergence(prep_ori, prep_gen, logger)


if __name__ == '__main__':
    args = parse_args_uncond()  # load unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
