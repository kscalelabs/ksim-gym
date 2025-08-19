"""Defines simple task for training a walking policy for the default humanoid."""

import asyncio
import functools
import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

# These are in the order of the neural network outputs.
ZEROS: list[tuple[str, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0),
    ("dof_right_shoulder_roll_03", math.radians(-10.0)),
    ("dof_right_shoulder_yaw_02", 0.0),
    ("dof_right_elbow_02", math.radians(45.0)),
    ("dof_right_wrist_00", 0.0),
    ("dof_left_shoulder_pitch_03", 0.0),
    ("dof_left_shoulder_roll_03", math.radians(10.0)),
    ("dof_left_shoulder_yaw_02", 0.0),
    ("dof_left_elbow_02", math.radians(-45.0)),
    ("dof_left_wrist_00", 0.0),
    ("dof_right_hip_pitch_04", math.radians(-10.0)),
    ("dof_right_hip_roll_03", math.radians(-0.0)),
    ("dof_right_hip_yaw_03", 0.0),
    ("dof_right_knee_04", math.radians(-30.0)),
    ("dof_right_ankle_02", math.radians(20.0)),
    ("dof_left_hip_pitch_04", math.radians(10.0)),
    ("dof_left_hip_roll_03", math.radians(0.0)),
    ("dof_left_hip_yaw_03", 0.0),
    ("dof_left_knee_04", math.radians(30.0)),
    ("dof_left_ankle_02", math.radians(-20.0)),
]


@dataclass
class HumanoidWalkingTaskConfig(ksim.PPOConfig):
    """Config for the humanoid walking task."""

    # Model parameters.
    hidden_size: int = xax.field(
        value=512,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=2,
        help="The depth for the MLPs.",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )
    num_hidden_layers: int = xax.field(
        value=2,
        help="The number of hidden layers for the MLPs.",
    )
    var_scale: float = xax.field(
        value=0.5,
        help="The scale for the standard deviations of the actor.",
    )

    # Reward parameters.
    linear_velocity_range: tuple[float, float] = xax.field(
        value=(1.0, 3.0),
        help="The range for the linear velocity command.",
    )
    linear_velocity_max_yaw: float = xax.field(
        value=math.radians(45.0),
        help="The maximum yaw for the linear velocity command.",
    )
    linear_velocity_zero_prob: float = xax.field(
        value=0.2,
        help="The probability of the linear velocity command being zero.",
    )
    linear_velocity_backward_prob: float = xax.field(
        value=0.0,
        help="The probability of the linear velocity command being backward.",
    )
    linear_velocity_switch_prob: float = xax.field(
        value=0.005,
        help="The probability of the linear velocity command being switched.",
    )
    angular_velocity_range: tuple[float, float] = xax.field(
        value=(-math.radians(45), math.radians(45)),
        help="The range for the angular velocity command.",
    )
    angular_velocity_zero_prob: float = xax.field(
        value=0.2,
        help="The probability of the angular velocity command being zero.",
    )
    angular_velocity_switch_prob: float = xax.field(
        value=0.005,
        help="The probability of the angular velocity command being switched.",
    )
    gait_period: float = xax.field(
        value=0.6,
        help="The period for the sinusoidal gait command.",
    )
    max_foot_height: float = xax.field(
        value=0.3,
        help="The maximum height for the sinusoidal gait command.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=1e-3,
        help="Learning rate for PPO.",
    )
    grad_clip: float = xax.field(
        value=1.0,
        help="Gradient clipping for the optimizer.",
    )
    warmup_steps: int = xax.field(
        value=1000,
        help="Number of warmup steps for the optimizer.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.LSTMCell, ...]
    output_proj: eqx.nn.MLP
    num_inputs: int = eqx.field()
    num_outputs: int = eqx.field()
    num_mixtures: int = eqx.field()
    clip_positions: ksim.ClipPositions = eqx.field()
    min_std: float = eqx.field()
    max_std: float = eqx.field()
    var_scale: float = eqx.field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        physics_model: ksim.PhysicsModel,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
        num_hidden_layers: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layers (LSTM)
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.LSTMCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=k,
                )
                for k in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=num_outputs * 3 * num_mixtures,
            width_size=hidden_size,
            depth=num_hidden_layers,
            activation=jax.nn.gelu,
            use_final_bias=False,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_mixtures = num_mixtures
        self.clip_positions = ksim.ClipPositions.from_physics_model(physics_model)
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, obs_n: Array, carry: Array) -> tuple[xax.Distribution, Array]:
        # carry shape: (2, depth, hidden_size) -> [0]=h, [1]=c
        x_n = self.input_proj(obs_n)
        new_h = []
        new_c = []
        for i, rnn in enumerate(self.rnns):
            h_i = carry[0, i]
            c_i = carry[1, i]
            h_o, c_o = rnn(x_n, (h_i, c_i))
            x_n = h_o
            new_h.append(h_o)
            new_c.append(c_o)
        out_n = self.output_proj(x_n)

        # Reshape the output to be a mixture of gaussians.
        slice_len = self.num_outputs * self.num_mixtures
        mean_nm = out_n[..., :slice_len].reshape(self.num_outputs, self.num_mixtures)
        std_nm = out_n[..., slice_len : slice_len * 2].reshape(self.num_outputs, self.num_mixtures)
        logits_nm = out_n[..., slice_len * 2 :].reshape(self.num_outputs, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)

        # Apply bias to the means.
        mean_nm = mean_nm + jnp.array([v for _, v in ZEROS])[:, None]

        # Clip the target positions to the minimum and maximum ranges.
        mean_nm = jax.vmap(self.clip_positions.clip, in_axes=-1, out_axes=-1)(mean_nm)

        dist_n = xax.MixtureOfGaussians(means_nm=mean_nm, stds_nm=std_nm, logits_nm=logits_nm)

        next_carry = jnp.stack([jnp.stack(new_h, axis=0), jnp.stack(new_c, axis=0)], axis=0)
        return dist_n, next_carry


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.LSTMCell, ...]
    output_proj: eqx.nn.MLP
    num_inputs: int = eqx.field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
        num_hidden_layers: int,
    ) -> None:
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layers (LSTM)
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.LSTMCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=k,
                )
                for k in rnn_keys
            ]
        )

        # Create MLP
        self.output_proj = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=num_outputs,
            width_size=hidden_size,
            depth=num_hidden_layers,
            activation=jax.nn.elu,
            use_final_bias=False,
            key=key,
        )

        self.num_inputs = num_inputs

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        # carry shape: (2, depth, hidden_size) -> [0]=h, [1]=c
        x_n = self.input_proj(obs_n)
        new_h = []
        new_c = []
        for i, rnn in enumerate(self.rnns):
            h_i = carry[0, i]
            c_i = carry[1, i]
            h_o, c_o = rnn(x_n, (h_i, c_i))
            x_n = h_o
            new_h.append(h_o)
            new_c.append(c_o)
        out_n = self.output_proj(x_n)

        next_carry = jnp.stack([jnp.stack(new_h, axis=0), jnp.stack(new_c, axis=0)], axis=0)
        return out_n, next_carry


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        physics_model: ksim.PhysicsModel,
        num_actor_inputs: int,
        num_actor_outputs: int,
        num_critic_inputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
        num_hidden_layers: int,
    ) -> None:
        actor_key, critic_key = jax.random.split(key)
        self.actor = Actor(
            actor_key,
            physics_model=physics_model,
            num_inputs=num_actor_inputs,
            num_outputs=num_actor_outputs,
            min_std=min_std,
            max_std=max_std,
            var_scale=var_scale,
            hidden_size=hidden_size,
            num_mixtures=num_mixtures,
            depth=depth,
            num_hidden_layers=num_hidden_layers,
        )
        self.critic = Critic(
            critic_key,
            hidden_size=hidden_size,
            depth=depth,
            num_inputs=num_critic_inputs,
            num_hidden_layers=num_hidden_layers,
        )


class HumanoidWalkingTask(ksim.PPOTask[HumanoidWalkingTaskConfig]):
    def get_optimizer(self) -> optax.GradientTransformation:
        scheduler = optax.warmup_constant_schedule(
            init_value=self.config.learning_rate * 0.01,
            peak_value=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
        )

        def decay_mask(pytree: PyTree) -> PyTree:
            return jax.tree.map(lambda w: w.ndim >= 2, pytree)

        return optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(self.config.grad_clip),
            optax.masked(optax.add_decayed_weights(self.config.adam_weight_decay), decay_mask),
            optax.scale_by_adam(),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1.0),
        )

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot-headless", name="robot"))
        return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata("kbot-headless"))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        if metadata.actuator_type_to_metadata is None:
            raise ValueError("Actuator metadata is not available")
        return metadata

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.PositionActuators(
            physics_model=physics_model,
            metadata=metadata,
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.PhysicsRandomizer]:
        return {
            "static_friction": ksim.StaticFrictionRandomizer(),
            "floor_friction": ksim.FloorFrictionRandomizer.from_geom_name(physics_model, "floor"),
            "armature": ksim.ArmatureRandomizer(),
            "all_bodies_mass_multiplication": ksim.AllBodiesMassMultiplicationRandomizer(
                scale_lower=0.95,
                scale_upper=1.05,
            ),
            "joint_damping": ksim.JointDampingRandomizer(),
            "joint_zero_position": ksim.JointZeroPositionRandomizer(
                scale_lower=math.radians(-2),
                scale_upper=math.radians(2),
            ),
        }

    def get_events(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Event]:
        return {
            "push": ksim.LinearPushEvent(
                linvel=1.0,
                vel_range=(0.0, 1.0),
                interval_range=(4.0, 8.0),
                curriculum_range=(0.0, 1.0),
            ),
            "jump": ksim.JumpEvent(
                jump_height_range=(0.1, 0.3),
                interval_range=(4.0, 8.0),
                curriculum_range=(0.0, 1.0),
            ),
        }

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v in ZEROS}, scale=0.1),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Observation]:
        return {
            "joint_position": ksim.JointPositionObservation(noise=ksim.AdditiveUniformNoise(mag=math.radians(2))),
            "joint_velocity": ksim.JointVelocityObservation(noise=ksim.AdditiveUniformNoise(mag=math.radians(30))),
            "actuator_force": ksim.ActuatorForceObservation(),
            "center_of_mass_inertia": ksim.CenterOfMassInertiaObservation(),
            "center_of_mass_velocity": ksim.CenterOfMassVelocityObservation(),
            "base_position": ksim.BasePositionObservation(),
            "base_orientation": ksim.BaseOrientationObservation(),
            "base_linear_velocity": ksim.BaseLinearVelocityObservation(),
            "base_angular_velocity": ksim.BaseAngularVelocityObservation(),
            "base_linear_acceleration": ksim.BaseLinearAccelerationObservation(),
            "base_angular_acceleration": ksim.BaseAngularAccelerationObservation(),
            "actuator_acceleration": ksim.ActuatorAccelerationObservation(),
            "imu_projected_gravity": ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                noise=ksim.AdditiveGaussianNoise(std=0.01),
                min_lag=0.001,
                max_lag=0.005,
                bias=math.radians(2.0),
            ),
            "projected_gravity": ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
            ),
            "imu_acc": ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_acc",
                noise=ksim.AdditiveUniformNoise(mag=0.01),
            ),
            "imu_gyro": ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=ksim.AdditiveUniformNoise(mag=math.radians(10)),
            ),
            "feet_contact": ksim.FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names=[
                    "KB_D_501L_L_LEG_FOOT_collision_capsule_0",
                    "KB_D_501L_L_LEG_FOOT_collision_capsule_1",
                ],
                foot_right_geom_names=[
                    "KB_D_501R_R_LEG_FOOT_collision_capsule_0",
                    "KB_D_501R_R_LEG_FOOT_collision_capsule_1",
                ],
                floor_geom_names=["floor"],
            ),
            "feet_position": ksim.FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_body_name="KB_D_501L_L_LEG_FOOT",
                foot_right_body_name="KB_D_501R_R_LEG_FOOT",
            ),
            "feet_force": ksim.FeetForceObservation.create(
                physics_model=physics_model,
                foot_left_body_name="KB_D_501L_L_LEG_FOOT",
                foot_right_body_name="KB_D_501R_R_LEG_FOOT",
            ),
        }

    def get_commands(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Command]:
        return {
            "linvel": ksim.LinearVelocityCommand(
                min_vel=self.config.linear_velocity_range[0],
                max_vel=self.config.linear_velocity_range[1],
                max_yaw=self.config.linear_velocity_max_yaw,
                zero_prob=self.config.linear_velocity_zero_prob,
                backward_prob=self.config.linear_velocity_backward_prob,
                switch_prob=self.config.linear_velocity_switch_prob,
            ),
            "angvel": ksim.AngularVelocityCommand(
                min_vel=self.config.angular_velocity_range[0],
                max_vel=self.config.angular_velocity_range[1],
                zero_prob=self.config.angular_velocity_zero_prob,
                switch_prob=self.config.angular_velocity_switch_prob,
            ),
        }

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Reward]:
        return {
            "stay_alive": ksim.StayAliveReward(scale=100.0),
            "foot_airtime": ksim.FeetAirTimeReward(
                ctrl_dt=self.config.ctrl_dt,
                period=self.config.gait_period / 2.0,
                contact_obs="feet_contact",
                scale=3.0,
            ),
            "foot_height": ksim.FeetHeightReward(
                contact_obs="feet_contact",
                position_obs="feet_position",
                height=self.config.max_foot_height,
                scale=10.0,
            ),
            "linvel": ksim.LinearVelocityReward(cmd="linvel", scale=1.0),
            "angvel": ksim.AngularVelocityReward(cmd="angvel", scale=0.1),
        }

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Termination]:
        return {
            "bad_z": ksim.BadZTermination(min_z=0.3, final_min_z=0.6, max_z=1.2),
            "far_from_origin": ksim.FarFromOriginTermination(max_dist=20.0),
        }

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.DistanceFromOriginCurriculum(
            min_level_steps=25,
            increase_threshold=8.0,
            decrease_threshold=8.0,
        )

    def get_model(self, params: ksim.InitParams) -> Model:
        return Model(
            params.key,
            physics_model=params.physics_model,
            num_actor_inputs=49,
            num_actor_outputs=len(ZEROS),
            num_critic_inputs=463,
            min_std=0.01,
            max_std=1.0,
            var_scale=self.config.var_scale,
            hidden_size=self.config.hidden_size,
            num_mixtures=self.config.num_mixtures,
            depth=self.config.depth,
            num_hidden_layers=self.config.num_hidden_layers,
        )

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, PyTree],
        commands: xax.FrozenDict[str, PyTree],
        carry: Array,
    ) -> tuple[xax.Distribution, Array]:
        joint_pos_n = observations["noisy_joint_position"]
        joint_vel_n = observations["noisy_joint_velocity"]
        proj_grav_3 = observations["noisy_imu_projected_gravity"]
        imu_gyro_3 = observations["noisy_imu_gyro"]

        # Command tensors.
        linvel_cmd: ksim.LinearVelocityCommandValue = commands["linvel"]
        angvel_cmd: ksim.AngularVelocityCommandValue = commands["angvel"]

        # Stacks into tensors.
        linvel_cmd_2 = jnp.stack([linvel_cmd.vel, linvel_cmd.yaw], axis=-1)
        angvel_cmd_1 = jnp.stack([angvel_cmd.vel], axis=-1)

        obs = [
            joint_pos_n,  # NUM_JOINTS
            joint_vel_n / 10.0,  # NUM_JOINTS
            proj_grav_3,  # 3
            imu_gyro_3,  # 3
            linvel_cmd_2,  # 2
            angvel_cmd_1,  # 1
        ]

        obs_n = jnp.concatenate(obs, axis=-1)
        action, carry = model.forward(obs_n, carry)

        return action, carry

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, PyTree],
        commands: xax.FrozenDict[str, PyTree],
        carry: Array,
    ) -> tuple[Array, Array]:
        dh_joint_pos_j = observations["joint_position"]
        dh_joint_vel_j = observations["joint_velocity"]
        com_inertia_n = observations["center_of_mass_inertia"]
        com_vel_n = observations["center_of_mass_velocity"]
        imu_acc_3 = observations["imu_acc"]
        imu_gyro_3 = observations["imu_gyro"]
        proj_grav_3 = observations["projected_gravity"]
        act_frc_obs_n = observations["actuator_force"]
        base_pos_3 = observations["base_position"]
        base_quat_4 = observations["base_orientation"]
        lin_vel_obs_3 = observations["base_linear_velocity"]
        ang_vel_obs_3 = observations["base_angular_velocity"]
        feet_contact_2 = observations["feet_contact"].any(axis=-2)
        feet_height_2 = observations["feet_position"][..., 2]
        feet_force_obs_23 = observations["feet_force"]

        # Flattens the last two dimensions.
        feet_force_obs_6 = feet_force_obs_23.reshape(*feet_force_obs_23.shape[:-2], 6)

        # Command tensors.
        linvel_cmd: ksim.LinearVelocityCommandValue = commands["linvel"]
        angvel_cmd: ksim.AngularVelocityCommandValue = commands["angvel"]

        # Stacks into tensors.
        linvel_cmd_2 = jnp.stack([linvel_cmd.vel, linvel_cmd.yaw], axis=-1)
        angvel_cmd_1 = jnp.stack([angvel_cmd.vel], axis=-1)

        obs_n = jnp.concatenate(
            [
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,
                com_vel_n,
                imu_acc_3,
                imu_gyro_3,
                proj_grav_3,
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,
                base_quat_4,
                lin_vel_obs_3,
                ang_vel_obs_3,
                feet_contact_2,
                feet_height_2,
                feet_force_obs_6 / 100.0,
                linvel_cmd_2,
                angvel_cmd_1,
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def _model_scan_fn(
        self,
        actor_critic_carry: tuple[Array, Array],
        xs: tuple[ksim.Trajectory, PRNGKeyArray],
        model: Model,
    ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
        transition, rng = xs

        actor_carry, critic_carry = actor_critic_carry
        actor_dist, next_actor_carry = self.run_actor(
            model=model.actor,
            observations=transition.obs,
            commands=transition.command,
            carry=actor_carry,
        )

        # Gets the log probabilities of the action.
        log_probs = actor_dist.log_prob(transition.action)
        assert isinstance(log_probs, Array)

        value, next_critic_carry = self.run_critic(
            model=model.critic,
            observations=transition.obs,
            commands=transition.command,
            carry=critic_carry,
        )

        transition_ppo_variables = ksim.PPOVariables(
            log_probs=log_probs,
            values=value.squeeze(-1),
        )

        next_carry = jax.tree.map(
            lambda x, y: jnp.where(transition.done, x, y),
            self.get_initial_model_carry(model, rng),
            (next_actor_carry, next_critic_carry),
        )

        return next_carry, transition_ppo_variables

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        scan_fn = functools.partial(self._model_scan_fn, model=model)
        next_model_carry, ppo_variables = xax.scan(
            scan_fn,
            model_carry,
            (trajectory, jax.random.split(rng, len(trajectory.done))),
            jit_level=4,
        )
        return ppo_variables, next_model_carry

    def get_initial_model_carry(self, model: Model, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(2, self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(2, self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: Model,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(key=rng)
        return ksim.Action(action=action_j, carry=(actor_carry, critic_carry_in))


if __name__ == "__main__":
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=4096,
            batch_size=512,
            num_passes=4,
            rollout_length_frames=24,
            # Simulation parameters.
            dt=0.004,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            action_latency_range=(0.001, 0.01),  # Simulate 1-10ms of latency.
            drop_action_prob=0.05,  # Drop 5% of commands.
            # Visualization parameters.
            # If running this on Mac and you are getting segaults,
            # you might need to disable `render_markers`
            render_markers=True,
            render_track_body_id=0,
            disable_multiprocessing=True,
        ),
    )
