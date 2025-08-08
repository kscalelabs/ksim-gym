"""Defines simple task for training a walking policy for the default humanoid."""

import asyncio
import functools
import math
from dataclasses import dataclass
from typing import Self

import attrs
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
    ("dof_right_elbow_02", math.radians(90.0)),
    ("dof_right_wrist_00", 0.0),
    ("dof_left_shoulder_pitch_03", 0.0),
    ("dof_left_shoulder_roll_03", math.radians(10.0)),
    ("dof_left_shoulder_yaw_02", 0.0),
    ("dof_left_elbow_02", math.radians(-90.0)),
    ("dof_left_wrist_00", 0.0),
    ("dof_right_hip_pitch_04", math.radians(-20.0)),
    ("dof_right_hip_roll_03", math.radians(-0.0)),
    ("dof_right_hip_yaw_03", 0.0),
    ("dof_right_knee_04", math.radians(-50.0)),
    ("dof_right_ankle_02", math.radians(30.0)),
    ("dof_left_hip_pitch_04", math.radians(20.0)),
    ("dof_left_hip_roll_03", math.radians(0.0)),
    ("dof_left_hip_yaw_03", 0.0),
    ("dof_left_knee_04", math.radians(50.0)),
    ("dof_left_ankle_02", math.radians(-30.0)),
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
    target_linear_velocity: float = xax.field(
        value=3.0,
        help="The linear velocity for the joystick command.",
    )
    target_angular_velocity: float = xax.field(
        value=math.radians(90.0),
        help="The angular velocity for the joystick command.",
    )
    gait_period: float = xax.field(
        value=0.6,
        help="The period for the sinusoidal gait command.",
    )
    max_foot_height: float = xax.field(
        value=0.2,
        help="The maximum height for the sinusoidal gait command.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
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


@attrs.define(frozen=True, kw_only=True)
class JointPositionPenalty(ksim.JointDeviationPenalty):
    @classmethod
    def create_from_names(
        cls,
        names: list[str],
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        zeros = {k: v for k, v in ZEROS}
        joint_targets = [zeros[name] for name in names]

        return cls.create(
            physics_model=physics_model,
            joint_names=tuple(names),
            joint_targets=tuple(joint_targets),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class BentArmPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "dof_right_shoulder_pitch_03",
                "dof_right_shoulder_roll_03",
                "dof_right_shoulder_yaw_02",
                "dof_right_elbow_02",
                "dof_right_wrist_00",
                "dof_left_shoulder_pitch_03",
                "dof_left_shoulder_roll_03",
                "dof_left_shoulder_yaw_02",
                "dof_left_elbow_02",
                "dof_left_wrist_00",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class StraightLegPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "dof_left_hip_roll_03",
                # "dof_left_hip_yaw_03",
                "dof_right_hip_roll_03",
                # "dof_right_hip_yaw_03",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
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

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
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
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
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

        return dist_n, jnp.stack(out_carries, axis=0)


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
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

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
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
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


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

        return optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(self.config.grad_clip),
            optax.add_decayed_weights(self.config.adam_weight_decay),
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

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.FloorFrictionRandomizer.from_geom_name(physics_model, "floor"),
            ksim.ArmatureRandomizer(),
            ksim.AllBodiesMassMultiplicationRandomizer(scale_lower=0.95, scale_upper=1.05),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(scale_lower=math.radians(-2), scale_upper=math.radians(2)),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.LinearPushEvent(
                linvel=0.2,
                vel_range=(0.0, 1.0),
                interval_range=(0.5, 2.0),
                curriculum_range=(0.0, 1.0),  # Always apply pushes.
            ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v in ZEROS}, scale=0.1),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.TimestepObservation(),
            ksim.JointPositionObservation(noise=math.radians(2)),
            ksim.JointVelocityObservation(noise=math.radians(30)),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ActuatorAccelerationObservation(),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                lag_range=(0.0, 0.1),
                noise=math.radians(1),
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_acc",
                noise=1.0,
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=math.radians(10),
            ),
            ksim.FeetContactObservation.create(
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
            ksim.FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_body_name="KB_D_501L_L_LEG_FOOT",
                foot_right_body_name="KB_D_501R_R_LEG_FOOT",
            ),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.EasyJoystickCommand(
                gait=ksim.SinusoidalGaitCommand(
                    gait_period=self.config.gait_period,
                    ctrl_dt=self.config.ctrl_dt,
                    max_height=self.config.max_foot_height,
                    height_offset=0.06,
                ),
                joystick=ksim.JoystickCommand(
                    run_speed=self.config.target_linear_velocity,
                    walk_speed=self.config.target_linear_velocity / 2.0,
                    strafe_speed=self.config.target_linear_velocity / 2.0,
                    rotation_speed=self.config.target_angular_velocity,
                    # Only allow forward and standing.
                    # sample_probs=(0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                ),
            ),
            ksim.BaseHeightCommand(
                min_height=0.9,
                max_height=1.02,
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            # Standard rewards.
            # ksim.StayAliveReward(balance=2.0, scale=1.0),
            ksim.BaseHeightTrackingReward(scale=0.5),
            # ksim.UprightReward(scale=1.0),
            ksim.EasyJoystickReward(
                gait=ksim.SinusoidalGaitReward(
                    scale=0.4,
                    ctrl_dt=self.config.ctrl_dt,
                    max_height=self.config.max_foot_height,
                ),
                joystick=ksim.JoystickReward(scale=0.1),
                airtime=ksim.FeetAirTimeReward(
                    threshold=self.config.gait_period / 2.0,
                    ctrl_dt=self.config.ctrl_dt,
                    scale=0.1,
                ),
            ),
            # Avoid movement penalties.
            # ksim.AngularVelocityPenalty(index=("x", "y"), scale=-0.1),
            # ksim.LinearVelocityPenalty(index=("z"), scale=-0.1),
            # Normalization penalties.
            # ksim.AvoidLimitsPenalty.create(physics_model, scale=-0.01),
            # ksim.JointAccelerationPenalty(scale=-0.01),
            # ksim.JointJerkPenalty(scale=-0.01),
            # ksim.LinkAccelerationPenalty(scale=-0.01),
            # ksim.LinkJerkPenalty(scale=-0.01),
            # ksim.ActionAccelerationPenalty(scale=-0.01),
            # Bespoke rewards.
            # BentArmPenalty.create_penalty(physics_model, scale=-1.0),
            # StraightLegPenalty.create_penalty(physics_model, scale=-1.0),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.7, unhealthy_z_upper=1.2),
            ksim.FarFromOriginTermination(max_dist=10.0),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        # return ksim.DistanceFromOriginCurriculum(
        #     min_level_steps=5,
        # )
        return ksim.ConstantCurriculum(level=1.0)

    def get_model(self, params: ksim.InitParams) -> Model:
        return Model(
            params.key,
            physics_model=params.physics_model,
            num_actor_inputs=56,
            num_actor_outputs=len(ZEROS),
            num_critic_inputs=457,
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
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]

        # Sinusoidal gait joystick command.
        sgj_cmd: ksim.EasyJoystickCommandValue = commands["easy_joystick_command"]
        joystick_cmd_ohe_8 = sgj_cmd.joystick.command

        # Foot target height.
        gait_phase_1 = sgj_cmd.gait.phase[..., None]

        base_height_1 = commands["base_height_command"][..., None]

        obs = [
            joint_pos_n,  # NUM_JOINTS
            joint_vel_n,  # NUM_JOINTS
            proj_grav_3,  # 3
            imu_gyro_3,  # 3
            gait_phase_1,  # 1
            base_height_1,  # 1
            joystick_cmd_ohe_8,  # 8
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
        time_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]

        # Sinusoidal gait joystick command.
        sgj_cmd: ksim.EasyJoystickCommandValue = commands["easy_joystick_command"]
        joystick_cmd_ohe_8 = sgj_cmd.joystick.command

        # Foot height difference.
        foot_height_2 = observations["feet_position_observation"][..., 2]
        foot_tgt_height_2 = sgj_cmd.gait.height
        foot_height_diff_2 = foot_height_2 - foot_tgt_height_2

        base_height_1 = commands["base_height_command"][..., None]

        obs_n = jnp.concatenate(
            [
                jnp.sin(time_1),
                jnp.cos(time_1),
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                joystick_cmd_ohe_8,  # 8
                foot_height_diff_2,  # 2
                base_height_1,  # 1
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
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
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
            num_envs=2048,
            batch_size=256,
            num_passes=4,
            rollout_length_frames=24,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            action_latency_range=(0.003, 0.01),  # Simulate 3-10ms of latency.
            drop_action_prob=0.05,  # Drop 5% of commands.
            # Visualization parameters.
            render_markers=False,
            render_track_body_id=0,
        ),
    )
