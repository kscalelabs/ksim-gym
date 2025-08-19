"""Converts a checkpoint to a deployable model."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import ksim
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

from train import HumanoidWalkingTask, Model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    if not (ckpt_path := Path(args.checkpoint_path)).exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    task: HumanoidWalkingTask = HumanoidWalkingTask.load_task(ckpt_path)
    mujoco_model = task.get_mujoco_model()
    key = jax.random.PRNGKey(0)
    init_params = ksim.InitParams(key=key, physics_model=mujoco_model)
    model: Model = task.load_ckpt(ckpt_path, init_params=init_params, part="model")[0]

    # Loads the Mujoco model and gets the joint names.
    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]  # Removes the root joint.

    # Constant values.
    depth = task.config.depth
    hidden_size = task.config.hidden_size
    carry_shape = (depth * hidden_size,)

    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=3,
        carry_size=carry_shape,
    )

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_shape)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        projected_gravity: Array,
        gyroscope: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:

        # Call the model.
        obs = jnp.concatenate(
            [
                joint_angles,
                joint_angular_velocities / 10.0,
                projected_gravity,
                gyroscope,
                command,
            ],
            axis=-1,
        )
        dist, new_carry = model.actor.forward(obs, carry)

        return dist.mode(), new_carry

    init_onnx = export_fn(
        model=init_fn,
        metadata=metadata,
    )

    step_onnx = export_fn(
        model=step_fn,
        metadata=metadata,
    )

    kinfer_model = pack(
        init_fn=init_onnx,
        step_fn=step_onnx,
        metadata=metadata,
    )

    # Saves the resulting model.
    (output_path := Path(args.output_path)).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(kinfer_model)


if __name__ == "__main__":
    main()
