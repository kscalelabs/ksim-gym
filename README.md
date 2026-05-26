<div align="center">
<h1>K-Sim Gym</h1>
<p>Train and deploy your own humanoid robot controller in 700 lines of Python</p>
<h3>
  <a href="https://youtu.be/c64FnSvj8kQ">Tutorial</a> ·
  <a href="https://kscale.dev/benchmarks">Leaderboard</a> ·
  <a href="https://docs.kscale.dev/docs/quick-start#/">Documentation</a>
  <br />
  <a href="https://github.com/kscalelabs/ksim/tree/master/examples">K-Sim Examples</a> ·
  <a href="https://github.com/kscalelabs/kbot-joystick">Joystick Example</a>
</h3>

https://github.com/user-attachments/assets/82e5e998-1d62-43e2-ae52-864af6e72629

</div>

## Getting Started

You can use this repository as a GitHub template or as a Google Colab.

### Google Colab

You can quickly try out the humanoid benchmark by running the [training notebook](https://colab.research.google.com/github/kscalelabs/ksim-gym/blob/master/train.ipynb) in Google Colab.

### On your own GPU

1. Read through the [current leaderboard](https://www.kscale.dev/benchmarks) submissions and through the [ksim examples](https://github.com/kscalelabs/ksim/tree/master/examples)
2. Create a new repository from this template by clicking [here](https://github.com/new?template_name=ksim-gym&template_owner=kscalelabs)
3. Clone the new repository you create from this template:

```bash
git clone git@github.com:<YOUR USERNAME>/ksim-gym.git
cd ksim-gym
```

4. Create a new Python environment (we require Python 3.11 or later and recommend using [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html))
5. Install the package with its dependencies:

```bash
pip install -r requirements.txt
pip install 'jax[cuda12]>=0.6.2,<0.7'  # If using GPU machine, install JAX CUDA libraries
python -c "import jax; print(jax.default_backend())" # Should print "gpu"
```

The template pins JAX and MuJoCo to the currently supported compatibility
window. Installing unpinned JAX 0.7+ can break transitive dependencies that
still reference deprecated JAX internals.

6. Train a policy:
  - Your robot should be walking within ~80 training steps, which takes 30 minutes on an RTX 4090 GPU.
  - Training runs indefinitely, unless you set the `max_steps` argument. You can also use `Ctrl+C` to stop it.
  - Click on the TensorBoard link in the terminal to visualize the current run's training logs and videos.
  - List all the available arguments with `python -m train --help`.
```bash
python -m train
```
```bash
# You can override default arguments like this
python -m train max_steps=100
```
7. To see the TensorBoard logs for all your runs:
```bash
tensorboard --logdir humanoid_walking_task
```
8. To view your trained checkpoint in the interactive viewer:
- Use the mouse to move the camera around
- Hold `Ctrl` and double click to select a body on the robot, and then left or right click to apply forces to it.
```bash
python -m train run_mode=view load_from_ckpt_path=humanoid_walking_task/run_<number>/checkpoints/ckpt.bin
```

9. Convert your trained checkpoint to a `kinfer` model, which can be deployed on a real robot:

```bash
python -m convert /path/to/ckpt.bin /path/to/model.kinfer
```

10. Visualize the converted model in [`kinfer-sim`](https://docs.kscale.dev/docs/k-infer):

```bash
kinfer-sim assets/model.kinfer kbot --start-height 1.2 --save-video video.mp4
```

11. Commit the K-Infer model and the recorded video to this repository
12. Push your code and model to your repository, and make sure the repository is public (you may need to use [Git LFS](https://git-lfs.com))
13. Write a message with a link to your repository on our [Discord](https://url.kscale.dev/discord) in the "【🧠】submissions" channel
14. Wait for one of us to run it on the real robot - this should take about a day, but if we are dragging our feet, please message us on Discord
15. Voila! Your name will now appear on our [leaderboard](https://url.kscale.dev/leaderboard)

## Troubleshooting

If you encounter issues, please consult the [ksim documentation](https://docs.kscale.dev/docs/ksim#/) or reach out to us on [Discord](https://url.kscale.dev/discord).

## Tips and Tricks

To see all the available command line arguments, use the command:

```bash
python -m train --help
```

To visualize running your model without using `kinfer-sim`, use the command:

```bash
python -m train run_mode=view
```

To see an example of a locomotion task with more complex reward tuning, see our [kbot-joystick](https://github.com/kscalelabs/kbot-joystick) task which was generated from this template. It also contains a pretrained checkpoint that you can initialize training from by running

```bash
python -m train load_from_ckpt_path=assets/ckpt.bin
```

You can also visualize the pre-trained model by combining these two commands:

```bash
python -m train load_from_ckpt_path=assets/ckpt.bin run_mode=view
```

If you want to use the Jupyter notebook and don't want to commit your training logs, we suggest using [pre-commit](https://pre-commit.com/) to clean the notebook before committing:

```bash
pip install pre-commit
pre-commit install
```
