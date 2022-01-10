from collections import OrderedDict
import numpy as np
from robosuite.utils.transform_utils import convert_quat

from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.models.arenas import TableArena
from robosuite.models.objects import PotWithHandlesObject
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

import robosuite.utils.transform_utils as T
from robosuite.models.objects import (
    MilkObject,
    BreadObject,
    CerealObject,
    CanObject,
    LaptopObject,
    BottleObject,
)

class ur5e_pickandplace(SingleArmEnv):
    """
    This class corresponds to the lifting task for two robot arms.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be either 2 single single-arm robots or 1 bimanual robot!

        env_configuration (str): Specifies how to position the robots within the environment. Can be either:

            :`'bimanual'`: Only applicable for bimanual robot setups. Sets up the (single) bimanual robot on the -x
                side of the table
            :`'single-arm-parallel'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots next to each other on the -x side of the table
            :`'single-arm-opposed'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots opposed from each others on the opposite +/-y sides of the table.

        Note that "default" corresponds to either "bimanual" if a bimanual robot is used or "single-arm-opposed" if two
        single-arm robots are used.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        ValueError: [Invalid number of robots specified]
        ValueError: [Invalid env configuration]
        ValueError: [Invalid robots for specified env configuration]
    """

    def __init__(
        self,
        robots="UR5e",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 1.2, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        single_object_mode=0,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):

        self.single_object_mode = single_object_mode
        
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.obj_names = ["milk", "bread", "cereal", "can"]
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 3.0 is provided if the pot is lifted and is parallel within 30 deg to the table

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.5], per-arm component that is proportional to the distance between each arm and its
              respective pot handle, and exactly 0.5 when grasping the handle
              - Note that the agent only gets the lifting reward when flipping no more than 30 degrees.
            - Grasping: in {0, 0.25}, binary per-arm component awarded if the gripper is grasping its correct handle
            - Lifting: in [0, 1.5], proportional to the pot's height above the table, and capped at a certain threshold

        Note that the final reward is normalized and scaled by reward_scale / 3.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0

        # check if the pot is tilted more than 30 degrees
        # mat = T.quat2mat(self._pot_quat)
        # z_unit = [0, 0, 1]
        # z_rotated = np.matmul(mat, z_unit)
        # cos_z = np.dot(z_unit, z_rotated)
        # cos_30 = np.cos(np.pi / 6)
        # direction_coef = 1 if cos_z >= cos_30 else 0
        direction_coef = 1

        # check for goal completion: cube is higher than the table top above a margin
        if self._check_success():
            reward = 3.0 * direction_coef

        # use a shaping reward
        elif self.reward_shaping:
            # lifting reward
            #pot_bottom_height = self.sim.data.site_xpos[self.pot_center_id][2] - self.pot.top_offset[2]
            table_height = self.sim.data.site_xpos[self.table_top_id][2]
            
            obj_height = self.sim.data.body_xpos[self.obj_body_id["Milk"]][2]
            elevation = obj_height - table_height
            r_lift = min(max(elevation - 0.05, 0), 0.15)
            reward += 10. * direction_coef * r_lift




            _gripper0_to_handle0 = self._gripper0_to_box
            _gripper1_to_handle1 = self._gripper1_to_box

            # gh stands for gripper-handle
            # When grippers are far away, tell them to be closer

            # Get contacts
            (g0, g1) = (self.robots[0].gripper["right"], self.robots[0].gripper["left"]) if \
                self.env_configuration == "bimanual" else (self.robots[0].gripper, self.robots[1].gripper)

            _g0h_dist = np.linalg.norm(_gripper0_to_handle0)
            _g1h_dist = np.linalg.norm(_gripper1_to_handle1)

            # Grasping reward
            if self._check_grasp(gripper=g0, object_geoms=self.cube):
                reward += 0.25
            # Reaching reward
            reward += 0.5 * (1 - np.tanh(10.0 * _g0h_dist))

            # Grasping reward
            if self._check_grasp(gripper=g1, object_geoms=self.cube):
                reward += 0.25
            # Reaching reward
            reward += 0.5 * (1 - np.tanh(10.0 * _g1h_dist))

        if self.reward_scale is not None:
            reward *= self.reward_scale / 3.0

        return reward










    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        self.objects = []
        self.obstacles = []
    

        # Adjust base pose(s) accordingly
        if self.env_configuration == "bimanual":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "single-arm-opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi/2, -np.pi/2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    #xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            else:   # "single-arm-parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    #xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        # initialize objects of interest

        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size=[0.02, 0.02, 0.02],
            #size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            #size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        
        milk_obj = MilkObject(name="milk")
        bread_obj = BreadObject(name="bread")
        cereal_obj = CerealObject(name="cereal")
        can_obj = CanObject(name="can")
        bottle_obj = BottleObject(name='bottle')
        laptop_obj = LaptopObject(name="laptop")
        #self.objects.append(self.cube)
        self.objects.append(milk_obj)
        self.obstacles.append(laptop_obj)
        
        # self.objects.append(bread_obj)
        # self.objects.append(cereal_obj)
        # self.objects.append(can_obj)
        # self.objects.append(bottle_obj)
        self.obj_names = ['milk' , "laptop"]
        
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # Create placement initializer
        # if self.placement_initializer is not None:
        #     self.placement_initializer.reset()
        #     self.placement_initializer.add_objects(self.objects)
        # else:
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.objects,
            x_range=[-0.25, -0.23],
            y_range=[-0.53, -0.52],
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            #rotation=(np.pi + -np.pi / 3, np.pi + np.pi / 3),
            rotation=(0, 0),
            z_offset=0.00,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
            name="obstaclesSampler",
            mujoco_objects=self.obstacles,
            x_range=[0.12, 0.13],
            y_range=[0.0 , 0.01],
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            #rotation=(np.pi + -np.pi / 3, np.pi + np.pi / 3),
            rotation=(0, 0),
            z_offset=0.00,
            )
        )
        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=self.objects + self.obstacles,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        # self.pot_body_id = self.sim.model.body_name2id(self.pot.root_body)
        # self.handle0_site_id = self.sim.model.site_name2id(self.pot.important_sites["handle0"])
        # self.handle1_site_id = self.sim.model.site_name2id(self.pot.important_sites["handle1"])
        self.table_top_id = self.sim.model.site_name2id("table_top")
        self.table_id = self.sim.model.body_name2id("table")
        self.table_visual_id = self.sim.model.geom_name2id("table_visual")
        self.table_collision_id = self.sim.model.geom_name2id("table_collision")
        self.world_id = self.sim.model.body_name2id("world")

        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        # object-specific ids
        for obj in (self.objects):
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            #self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]
        # self.pot_center_id = self.sim.model.site_name2id(self.pot.important_sites["center"])
        #self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        for obj in (self.obstacles):
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
        self.table_legs_visual_id = []
        for i in range(4):
            self.table_legs_visual_id.append(self.sim.model.geom_name2id(f"table_leg{i+1}_visual"))


        


    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:

            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"
            
            self.object_id_to_sensors = {}

            # for conversion to relative gripper frame
            @sensor(modality=modality)
            def world_pose_in_gripper(obs_cache):
                return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))) if\
                    f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)
            sensors = [world_pose_in_gripper]
            names = ["world_pose_in_gripper"]
            enableds = [True]
            actives = [False]

            for i, obj in enumerate(self.objects):
                # Create object sensors
                using_obj = (self.single_object_mode == 0 or self.object_id == i)
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj.name, modality=modality)
                sensors += obj_sensors
                names += obj_sensor_names
                enableds += [using_obj] * 4
                actives += [using_obj] * 4
                self.object_id_to_sensors[i] = obj_sensor_names
                
            for i, obj in enumerate(self.obstacles):
                # Create object sensors
                using_obj = (self.single_object_mode == 0 or self.object_id == i)
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj.name, modality=modality)
                sensors += obj_sensors
                names += obj_sensor_names
                enableds += [using_obj] * 4
                actives += [using_obj] * 4
                self.object_id_to_sensors[i] = obj_sensor_names            
                
                
                

            if self.single_object_mode == 1:
                # This is randomly sampled object, so we need to include object id as observation
                @sensor(modality=modality)
                def obj_id(obs_cache):
                    return self.object_id

                sensors.append(obj_id)
                names.append("obj_id")
                enableds.append(True)
                actives.append(True)
            


            @sensor(modality=modality)
            def pot_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.pot_body_id])

            @sensor(modality=modality)
            def pot_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.pot_body_id], to="xyzw")

            @sensor(modality=modality)
            def handle0_xpos(obs_cache):
                return np.array(self._handle0_xpos)

            @sensor(modality=modality)
            def handle1_xpos(obs_cache):
                return np.array(self._handle1_xpos)

            @sensor(modality=modality)
            def gripper0_to_handle0(obs_cache):
                return obs_cache["handle0_xpos"] - obs_cache[f"{pf0}eef_pos"] if \
                    "handle0_xpos" in obs_cache and f"{pf0}eef_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def gripper1_to_handle1(obs_cache):
                return obs_cache["handle1_xpos"] - obs_cache[f"{pf1}eef_pos"] if \
                    "handle1_xpos" in obs_cache and f"{pf1}eef_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cube_pos(obs_cache):
                return obs_cache[f"{pf}eef_pos"] - obs_cache["cube_pos"] if \
                    f"{pf}eef_pos" in obs_cache and "cube_pos" in obs_cache else np.zeros(3)


            #sensors = [cube_pos, cube_quat, gripper_to_cube_pos]
            #names = [s.__name__ for s in sensors]


            # Create observables
            for name, s, enabled, active in zip(names, sensors, enableds, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=enabled,
                    active=active,
                )
        return observables
    
    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        return sensors, names

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # Set the visual object body locations
                if "visual" in obj.name.lower():
                    self.sim.model.body_pos[self.obj_body_id[obj.name]] = obj_pos
                    self.sim.model.body_quat[self.obj_body_id[obj.name]] = obj_quat
                else:
                    # Set the collision object joints
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        # Set the bins to the desired position


    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the closest object.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the closest object
        if vis_settings["grippers"]:
            # find closest object
            dists = [
                self._gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=obj.root_body,
                    target_type="body",
                    return_distance=True,
                ) for obj in self.objects
            ]
            closest_obj_id = np.argmin(dists)
            # Visualize the distance to this target
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.objects[closest_obj_id].root_body,
                target_type="body",
            )

    def _check_success(self):
        """
        Check if pot is successfully lifted

        Returns:
            bool: True if pot is lifted
        """
        #cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        obj_height = []
        for n in self.obj_names:
            obj_height.append(self.sim.data.body_xpos[self.obj_body_id[n]][2])
        table_height = self.sim.data.site_xpos[self.table_top_id][2]

        # cube is higher than the table top above a margin
        return max(obj_height) > table_height + 0.30

    @property
    def _handle0_xpos(self):
        """
        Grab the position of the left (blue) hammer handle.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.site_xpos[self.handle0_site_id]
    
    @property
    def obj_pos(self):
        
        
        return self.sim.data.body_xpos[self.obj_body_id["Milk"]]
        
        
    
    @property
    def cube_pos(self):

        return self.sim.data.body_xpos[self.cube_body_id]
    
    
    

    @property
    def cube_quaternion(self):
        return T.convert_quat(self.sim.data.body_xquat[self.cube_body_id], to="xyzw")



    @property
    def _handle1_xpos(self):
        """
        Grab the position of the right (green) hammer handle.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.site_xpos[self.handle1_site_id]

    @property
    # def _pot_quat(self):
    #     """
    #     Grab the orientation of the pot body.

    #     Returns:
    #         np.array: (x,y,z,w) quaternion of the pot body
    #     """
    #     return T.convert_quat(self.sim.data.body_xquat[self.pot_body_id], to="xyzw")

    # @property
    # def _gripper0_to_handle0(self):
    #     """
    #     Calculate vector from the left gripper to the left pot handle.

    #     Returns:
    #         np.array: (dx,dy,dz) distance vector between handle and EEF0
    #     """
    #     return self._handle0_xpos - self._eef0_xpos

    # @property
    # def _gripper1_to_handle1(self):
    #     """
    #     Calculate vector from the right gripper to the right pot handle.

    #     Returns:
    #         np.array: (dx,dy,dz) distance vector between handle and EEF0
    #     """
    #     return self._handle1_xpos - self._eef1_xpos

    @property
    def _gripper0_to_box(self):
        """
        Calculate vector from the left gripper to the left pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self.cube_pos - self._eef0_xpos

    @property
    def _gripper1_to_box(self):
        """
        Calculate vector from the right gripper to the right pot handle.

        Returns:
            np.array: (dx,dy,dz) distance vector between handle and EEF0
        """
        return self.cube_pos - self._eef1_xpos
    
    
    @property
    def robot_pos(self):

        return self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
    
    
    @property
    def ros_objects_pos(self):
        
        objects_pos = {}
        
        for obj_n in self.obj_names:
            objects_pos[obj_n] = self.sim.data.body_xpos[self.obj_body_id[obj_n]] - self.robot_pos
        
        return objects_pos
    
    
    
    @property
    def ros_objects_quaternion(self):
        
        objects_quaternion = {}
        
        for obj_n in self.obj_names:
            objects_quaternion[obj_n] = T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_n]], to="xyzw")
        
        return objects_quaternion



    # @property
    # def ros_cube_pos(self):
        
    #     return self.sim.data.body_xpos[self.cube_body_id] - self.robot_pos

    # @property
    # def ros_cube_quaternion(self):
    #     return T.convert_quat(self.sim.data.body_xquat[self.cube_body_id], to="xyzw")

    @property
    def ros_cube_size(self):
        return self.cube.size

    @property
    def ros_table_visual_pos(self):
        return self.sim.data.geom_xpos[self.table_visual_id] - self.robot_pos

    @property
    def ros_table_visual_size(self):
        return self.sim.model.geom_size[self.table_visual_id]

    @property
    def ros_table_legs_pos(self):
        legs_pos = []
        for i in range(4):
            legs_pos.append(self.sim.data.geom_xpos[self.table_legs_visual_id[i]]-self.robot_pos)
        return legs_pos
    
    
    @property
    def ros_table_legs_size(self):
        return self.sim.model.geom_size[self.table_legs_visual_id[0]]
    


    def _object_pose(self):
        print("cube quaternion:" + f"{self.ros_cube_quaternion}")
        print("cube pose: " + f"{self.ros_cube_pos}")
        print("cube size: " + f"{self.cube.size}")
        return

    def test_(self):
        print("table site pos: " + f"{self.sim.data.site_xpos[self.table_top_id]}")
        print("table pos: " + f"{self.sim.data.body_xpos[self.table_id]}")
        print("world pos: " + f"{self.sim.data.body_xpos[self.world_id]}")
        return

    def test_property(self):
        print(self.ros_table_legs_size)
        return 