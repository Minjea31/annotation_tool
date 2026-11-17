from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    defaults = {
        'params_file': str(Path(get_package_share_directory('autorace_lane_lite')) / 'config' / 'lane.yaml'),
        'use_sim_time': 'false', 'backend': 'auto', 'model_path': '', 'device': 'cuda',
        'input_width': '512', 'input_height': '288', 'fallback_enabled': 'true',
    }
    typed = {'use_sim_time': bool, 'input_width': int, 'input_height': int, 'fallback_enabled': bool}
    overrides = {name: ParameterValue(LaunchConfiguration(name), value_type=typed.get(name, str))
                 for name in defaults if name != 'params_file'}
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()],
        Node(package='autorace_lane_lite', executable='lane_node', name='lane_node', output='screen',
             parameters=[LaunchConfiguration('params_file'), overrides]),
    ])
