import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    ExecuteProcess,
    TimerAction,
)
from launch.launch_description_sources import AnyLaunchDescriptionSource

def generate_launch_description():
    # Include the simulation launch file
    sim_launch_path = os.path.join(
        get_package_share_directory('crowd_nav'),
        'launch',
        'sim.launch.xml'
    )
    sim_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(sim_launch_path),
        launch_arguments={'sim_moving': 'true'}.items()
    )

    # Service call to move pedestrians
    move_ped = ExecuteProcess(
        cmd=['ros2', 'service', 'call', '/move_ped', 'std_srvs/srv/Empty']
    )

    # Service call to set the robot's goal (x=8, y=0)
    set_goal = ExecuteProcess(
        cmd=[
            'ros2', 'service', 'call',
            '/set_goal_pose',
            'crowd_nav_interfaces/srv/GoalReq',
            '{x: 8.0, y: 0.0}'
        ]
    )

    # Delay service calls to ensure simulation is ready
    return LaunchDescription([
        sim_launch,
        TimerAction(period=10.0, actions=[move_ped]),  # Wait 5s after simulation starts
        TimerAction(period=10.5, actions=[set_goal]), # Wait 10s after simulation starts
    ])
