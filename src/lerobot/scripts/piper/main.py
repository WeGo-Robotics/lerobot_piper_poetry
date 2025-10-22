import logging
import os
import pickle  # nosec
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch
from transitions import Machine

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots import (
    make_robot_from_config,
)
from lerobot.robots.piper_follower import PiperFollower
from lerobot.scripts.server.configs import RobotClientConfig
from lerobot.scripts.server.constants import SUPPORTED_ROBOTS
from lerobot.scripts.server.filter import action_filter
from lerobot.scripts.server.helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    validate_robot_cameras_for_policy,
    visualize_action_queue_size,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks


class RobotClient(object):
    prefix = "robot_client"
    logger = get_logger(prefix)

    states = [
        'Config',
        'Waiting',
        'AsyncMode',
        'SyncMode',
        'ManualMode',
        'TeachMode',
        'SafeMode'
    ]

    transitions = [
        { 'trigger' : 'config_done', 'source' : 'Config', 'dest' : 'Waiting' },
        { 'trigger' : 'start_async', 'source' : 'Waiting', 'dest' : 'AsyncMode' },
        { 'trigger' : 'start_sync', 'source' : 'Waiting', 'dest' : 'SyncMode' },
        { 'trigger' : 'start_manual', 'source' : 'Waiting', 'dest' : 'ManualMode' },
        { 'trigger' : 'start_teach', 'source' : 'Waiting', 'dest' : 'TeachMode' },
        { 'trigger' : 'start_teleop', 'source' : 'Waiting', 'dest' : 'TeleopMode' },
        { 'trigger' : 'enter_safe', 'source' : ['AsyncMode', 'SyncMode', 'ManualMode', 'TeachMode', 'TeleopMode'], 'dest' : 'SafeMode' },
        { 'trigger' : 'exit_safe', 'source' : 'SafeMode', 'dest' : 'Waiting' },
        { 'trigger' : 'reset', 'source' : 'SafeMode', 'dest' : 'Waiting' },
        { 'trigger' : 'shutdown', 'source' : ['SafeMode', 'Waiting'], 'dest' : '*' },
        { 'trigger' : 'stop', 'source' : ['AsyncMode', 'SyncMode', 'ManualMode', 'TeachMode', 'TeleopMode'], 'dest' : 'Waiting' },
    ]

    def __init__(self, config: RobotClientConfig):
        self.machine = Machine(model=self, states=self.states, transitions=self.transitions, initial='Config')
        self.config = config
        self.init_status = {
            'robot_initialized': False,
            'camera_initialized': False,
            'rpc_initialized': False,
            'logs_initialized': False,
        }
        self.task = config.task
        self.verbose = config.verbose

    def run(self) -> bool:
        while True:
            match self.state:
                case 'Config':
                    self.configure()
                case 'Waiting':
                    self.wait_for_start()
                case 'AsyncMode':
                    self.run_async_mode()
                case 'SyncMode':
                    self.run_sync_mode()
                case 'ManualMode':
                    self.run_manual_mode()
                case 'SafeMode':
                    self.run_safe_mode()
                case 'TeachMode':
                    self.run_teach_mode()
                case 'TeleopMode':
                    self.run_teleop_mode()
                case _:
                    raise ValueError(f"Unknown state: {self.machine.states}")

    def configure(self):
        lerobot_features = None

        def init_robot(self):
            nonlocal lerobot_features
            # TODO Detect CAN activity
            self.robot = make_robot_from_config(self.config.robot)
            self.robot = PiperFollower(self.config.robot)
            self.robot.connect()
            # self.robot.get_cameras()

            lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
            self.init_status['robot_initialized'] = True

        def init_camera(self):
            nonlocal lerobot_features
            if self.config.verify_robot_cameras:
                # Load policy config for validation
                policy_config = PreTrainedConfig.from_pretrained(self.config.pretrained_name_or_path)
                policy_image_features = policy_config.image_features

                # The cameras specified for inference must match the one supported by the policy chosen
                validate_robot_cameras_for_policy(lerobot_features, policy_image_features)
                self.init_status['camera_initialized'] = True

        def init_rpc(self):
            nonlocal lerobot_features
            # Use environment variable if server_address is not provided in config
            self.server_address = self.config.server_address

            self.policy_config = RemotePolicyConfig(
                self.config.policy_type,
                self.config.pretrained_name_or_path,
                lerobot_features,
                self.config.actions_per_chunk,
                self.config.policy_device,
            )
            self.channel = grpc.insecure_channel(
                self.server_address, grpc_channel_options(initial_backoff=f"{self.config.environment_dt:.4f}s")
            )
            self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
            self.logger.info(f"Initializing client to connect to server at {self.server_address}")
            self.logger.info(f"Loop time dt: {self.config.environment_dt:.4f}s")

            self.shutdown_event = threading.Event()

            # Initialize client side variables
            self.latest_action_lock = threading.Lock()
            self.latest_action = -1
            self.action_chunk_size = -1

            self._chunk_size_threshold = self.config.chunk_size_threshold

            self.action_queue = Queue()
            self.action_queue_lock = threading.Lock()  # Protect queue operations
            self.action_queue_size = []
            self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

            # FPS measurement
            self.fps_tracker = FPSTracker(target_fps=self.config.fps)
            self.init_status['rpc_initialized'] = True

        # TODO : remote logging
        def init_logs(self):
            self.logger.info("Robot connected and ready")
            if self.config.action_log:
                torch.set_printoptions(profile="full", linewidth=2000)
                self.action_log_fd = open("robot_actions_log.yaml", "w")
                print("timed_actions:" , file=self.action_log_fd)
            if self.config.obs_log:
                torch.set_printoptions(profile="full", linewidth=2000)
                self.obs_log_fd = open("robot_observations_log.yaml", "w")
                print("obs:" , file=self.obs_log_fd)
            self.init_status['logs_initialized'] = True

        def init_policy(self):
            """Start the robot client and connect to the policy server"""
            try:
                # client-server handshake
                start_time = time.perf_counter()
                self.stub.Ready(services_pb2.Empty())
                end_time = time.perf_counter()
                self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

                # send policy instructions
                policy_config_bytes = pickle.dumps(self.policy_config)
                policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

                self.logger.info("Sending policy instructions to policy server")
                self.logger.debug(
                    f"Policy type: {self.policy_config.policy_type} | "
                    f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                    f"Device: {self.policy_config.device}"
                )

                self.stub.SendPolicyInstructions(policy_setup)

                self.shutdown_event.clear()

                return True

            except grpc.RpcError as e:
                self.logger.error(f"Failed to connect to policy server: {e}")
                return False

        def init_receivers(self):
            # Create and start action receiver thread
            self.action_receiver_thread = threading.Thread(target=self.receive_actions, daemon=True)

            # Start action receiver thread
            self.action_receiver_thread.start()

        # TODO : add re-initialization without restarting the client
        if not self.init_status['robot_initialized']:
            init_robot(self)
        if not self.init_status['camera_initialized']:
            init_camera(self)
        if not self.init_status['rpc_initialized']:
            init_rpc(self)
        if not self.init_status['logs_initialized']:
            init_logs(self)

        init_policy(self)
        init_receivers(self)

        # TODO : REST API Server

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

        self.config_done()

    def wait_for_start(self):
        match self.config.client_mode:
            case 'async':
                self.start_async()
            case 'sync':
                self.start_sync()
            case 'manual':
                self.start_manual()
            case 'teach':
                self.start_teach()
            case 'teleop':
                self.start_teleop()
            case _:
                input("Client mode is not selected.\nPress Enter to continue...")
                self.start_async()
        pass

    def run_async_mode(self):
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(self.verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(self.task, self.verbose)

            self.logger.info(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))


    def run_sync_mode(self):
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")
        self.current_state = "OBSERVATION"

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            match self.current_state:
                case "ACTION":
                    """Control loop: (1) Performing actions, when available"""
                    if self.actions_available():
                        _performed_action = self.control_loop_action(self.verbose)
                    else:
                        self.current_state = "OBSERVATION"
                case "OBSERVATION":
                    """Control loop: (2) Streaming observations to the remote policy server"""
                    _captured_observation = self.control_loop_observation(self.task, self.verbose)
                    self.current_state = "WAITING"
                case "WAITING":
                    if self.actions_available():
                        self.current_state = "ACTION"
                        continue

            self.logger.info(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

    def run_manual_mode(self):
        pass

    def run_safe_mode(self):
        pass

    def run_teach_mode(self):
        pass

    def stop(self):
        self.action_receiver_thread.join()

        """Stop the robot client"""
        self.shutdown_event.set()

        self.robot.parking()
        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.info(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        preproc_list :list[TimedAction] = []
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                preproc_list.append(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            preproc_list.append(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )
        preproc_list = action_filter(preproc_list)
        for val in preproc_list:
            future_action_queue.put(val)

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!
                if self.config.action_log:
                    # self.action_queue_log.put(timed_actions)
                    print("- iter:" , file=self.action_log_fd)
                    for action in timed_actions:
                        print(f"  - timestamp: {action.timestamp}", file=self.action_log_fd)
                        print(f"    timestep: {action.timestep}"  , file=self.action_log_fd)
                        print(f"    action: {action.action}"      , file=self.action_log_fd)

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            motors = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
            obs_pos = { v : raw_observation[f"{v}.pos"] for _, v in enumerate(motors)}
            if self.config.obs_log:
                print(f" - timestamp: {observation.timestamp}", file=self.obs_log_fd)
                print(f"   timestep: {observation.timestep}"  , file=self.obs_log_fd)
                print(f"   observation: {obs_pos}" , file=self.obs_log_fd)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

@draccus.wrap()
def piper_client(cfg: RobotClientConfig):
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting piper client...")
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type != "piper_follower":
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    client.run()

    client.stop()
    client.logger.info("Client stopped")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        config_file_path = os.path.join(os.getcwd(), 'src/lerobot/scripts/piper/default.yaml')
        sys.argv.append(f"--config_path={config_file_path}")
    piper_client()
