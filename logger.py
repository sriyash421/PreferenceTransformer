import os

from matplotlib import pyplot as plt
import numpy as np
import wandb

class Logger(object):
    def __init__(self, FLAGS, save_dir):
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "preference_rl"),
            entity=os.environ.get("MY_WANDB_ID", "sriyash-uw"),
            group=FLAGS.env_name,
            job_type=FLAGS.comment,
            dir=save_dir
        )
        wandb.config.update(FLAGS)
        wandb.config.update(dict(FLAGS.config))

    def log(self, key, value, step):
        wandb.log({key: value}, step=step)

    def log_video(self, key, frames, step):
        if isinstance(frames, list):
            frames = np.stack(frames, axis=0)
            frames = frames.transpose(0, 3, 1, 2)
        video = wandb.Video(frames, fps=4, format="gif")
        wandb.log({key: video}, step=step)

    def log_histogram(self, key, histogram, step):
        fig = plt.figure()
        plt.hist(histogram.flatten(), bins=100)
        im = wandb.Image(fig)
        wandb.log({key: im}, step=step)
        plt.close(fig)

