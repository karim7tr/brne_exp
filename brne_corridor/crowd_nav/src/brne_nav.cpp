#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <armadillo>
#include <fstream> // Add this include for file output
#include <mutex>
#include <sstream>
#include <iomanip>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "crowd_nav_interfaces/msg/pedestrian_array.hpp"
#include "crowd_nav_interfaces/msg/twist_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Transform.h"
#include "brnelib/brne.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "nav_msgs/msg/path.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

using namespace std::chrono_literals;

struct RobotPose
{
  double x;
  double y;
  double theta;
  arma::rowvec toVec()
  {
    return arma::rowvec(std::vector<double>{x, y, theta});
  }
};

double dist(double x1, double y1, double x2, double y2)
{
  return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

class PathPlan : public rclcpp::Node
{
public:
  PathPlan()
  : Node("brne"), goal_set{false}, walls_generated{false}
  {
    // define parameters
    declare_parameter("replan_freq", 1.0);
    declare_parameter("dt", 0.1);
    declare_parameter("maximum_agents", 1);
    declare_parameter("n_samples", 1);
    declare_parameter("n_steps", 1);
    declare_parameter("cost_a1", 1.0);
    declare_parameter("cost_a2", 1.0);
    declare_parameter("cost_a3", 1.0);
    declare_parameter("kernel_a1", 1.0);
    declare_parameter("kernel_a2", 1.0);
    declare_parameter("y_min", -1.0);
    declare_parameter("y_max", 1.0);
    declare_parameter("people_timeout", 1.0);
    declare_parameter("goal_threshold", 1.0);
    declare_parameter("brne_activate_threshold", 1.0);
    declare_parameter("max_lin_vel", 1.0);
    declare_parameter("nominal_lin_vel", 1.0);
    declare_parameter("max_ang_vel", 1.0);
    declare_parameter("close_stop_threshold", 1.0);

    declare_parameter("offset_unitree_vel", false);

    // get parameters
    replan_freq = get_parameter("replan_freq").as_double();
    dt = get_parameter("dt").as_double();
    maximum_agents = get_parameter("maximum_agents").as_int();
    n_samples = get_parameter("n_samples").as_int();
    n_steps = get_parameter("n_steps").as_int();
    cost_a1 = get_parameter("cost_a1").as_double();
    cost_a2 = get_parameter("cost_a2").as_double();
    cost_a3 = get_parameter("cost_a3").as_double();
    kernel_a1 = get_parameter("kernel_a1").as_double();
    kernel_a2 = get_parameter("kernel_a2").as_double();
    y_min = get_parameter("y_min").as_double();
    y_max = get_parameter("y_max").as_double();
    people_timeout = get_parameter("people_timeout").as_double();
    goal_threshold = get_parameter("goal_threshold").as_double();
    brne_activate_threshold = get_parameter("brne_activate_threshold").as_double();
    max_ang_vel = get_parameter("max_ang_vel").as_double();
    max_lin_vel = get_parameter("max_lin_vel").as_double();
    nominal_lin_vel = get_parameter("nominal_lin_vel").as_double();
    close_stop_threshold = get_parameter("close_stop_threshold").as_double();
    offset_unitree_vel = get_parameter("offset_unitree_vel").as_bool();

    // print out parameters
    RCLCPP_INFO_STREAM(get_logger(), "Replan frequency: " << replan_freq << " Hz");
    RCLCPP_INFO_STREAM(get_logger(), "dt: " << dt);
    RCLCPP_INFO_STREAM(get_logger(), "Maximum agents: " << maximum_agents);
    RCLCPP_INFO_STREAM(get_logger(), "Number of samples: " << n_samples);
    RCLCPP_INFO_STREAM(get_logger(), "Number of timesteps: " << n_steps);
    RCLCPP_INFO_STREAM(get_logger(), "Costs: " << cost_a1 << " " << cost_a2 << " " << cost_a3);
    RCLCPP_INFO_STREAM(get_logger(), "Kernels: " << kernel_a1 << " " << kernel_a2);
    RCLCPP_INFO_STREAM(get_logger(), "Hallway: " << y_min << "->" << y_max);
    RCLCPP_INFO_STREAM(get_logger(), "People timeout after " << people_timeout << "s");
    RCLCPP_INFO_STREAM(get_logger(), "Goal Threshold " << goal_threshold << "m");
    RCLCPP_INFO_STREAM(get_logger(), "Close stop threshold " << close_stop_threshold << "m");
    RCLCPP_INFO_STREAM(get_logger(), "Brne Activate Threshold " << brne_activate_threshold << "m");
    RCLCPP_INFO_STREAM(
      get_logger(),
      "Max Lin: " << max_lin_vel << " nominal lin: " << nominal_lin_vel << " max ang: " <<
        max_ang_vel);
    RCLCPP_INFO_STREAM(get_logger(), "Offset Unitree Velocity? " << offset_unitree_vel);


    brne = brne::BRNE{kernel_a1, kernel_a2,
      cost_a1, cost_a2, cost_a3,
      dt, n_steps, n_samples,
      y_min, y_max};

    trajgen = brne::TrajGen{max_lin_vel, max_ang_vel, n_samples, n_steps, dt};

    // Print out the parameters of the BRNE object to make sure it got initialized right
    RCLCPP_INFO_STREAM(get_logger(), brne.param_string());

    // Define publishers and subscribers
    pedestrian_sub_ = create_subscription<crowd_nav_interfaces::msg::PedestrianArray>(
      "pedestrians", 10, std::bind(&PathPlan::pedestrians_cb, this, std::placeholders::_1));
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "goal_pose", 10, std::bind(&PathPlan::goal_cb, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "brne_odom", 10, std::bind(&PathPlan::odom_cb, this, std::placeholders::_1));
    cmd_buf_pub_ = create_publisher<crowd_nav_interfaces::msg::TwistArray>("cmd_buf", 10);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/optimal_path", 10);
    walls_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/walls", 10);
    ped_prediction_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/pedestrian_predictions", 10
    );

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Create a timer that executes at replan_freq
    std::chrono::milliseconds rate = (std::chrono::milliseconds) ((int)(1000. / replan_freq));
    timer_ = create_wall_timer(rate, std::bind(&PathPlan::timer_callback, this));

    optimal_path.header.frame_id = "brne_odom";

    // Open a CSV file for logging (existing logfile)
    logfile_.open("navigation_metrics.csv");
    logfile_ << "time,robot_x,robot_y,ped_x,ped_y,distance\n"; // Header row

    // --- NEW CSV Logging: Using the same format as your second code ---
    step_csv_filename = "output_data_corridor.csv";
    {
      std::lock_guard<std::mutex> lock(step_csv_mutex);
      std::ofstream step_file(step_csv_filename);
      if (step_file.is_open()) {
        step_file << "RosTime,RobotX,RobotY,RobotTheta,RobotVelocity,NumPedestrians,PedestrianPositions,BRNE_Used\n";
      }
    }

    // Initialize metrics
    trial_start_time_ = this->get_clock()->now().seconds();
    trial_path_length_ = 0.0;
    trial_closest_distance_ = 9999.0;
    last_robot_x_ = 0.0;
    last_robot_y_ = 0.0;
    trial_used_brne = false;
  }

private:
  double replan_freq, kernel_a1, kernel_a2, cost_a1, cost_a2, cost_a3, y_min, y_max, dt,
         max_ang_vel, max_lin_vel, people_timeout, goal_threshold, brne_activate_threshold,
         nominal_lin_vel, close_stop_threshold;
  int maximum_agents, n_samples, n_steps;

  brne::BRNE brne{};
  brne::TrajGen trajgen{};

  rclcpp::TimerBase::SharedPtr timer_;

  crowd_nav_interfaces::msg::PedestrianArray ped_buffer;
  crowd_nav_interfaces::msg::PedestrianArray selected_peds;

  crowd_nav_interfaces::msg::TwistArray robot_cmds;

  nav_msgs::msg::Path optimal_path;

  RobotPose robot_pose;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<crowd_nav_interfaces::msg::PedestrianArray>::SharedPtr pedestrian_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  rclcpp::Publisher<crowd_nav_interfaces::msg::TwistArray>::SharedPtr cmd_buf_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr walls_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ped_prediction_pub_;

  visualization_msgs::msg::MarkerArray wall_markers;

  bool goal_set;
  bool walls_generated;
  geometry_msgs::msg::PoseStamped goal;

  // trial information
  double trial_closest_dst_to_ped;
  double trial_path_length;
  RobotPose trial_start_pose;
  double trial_straight_line_length;
  double trial_path_ratio;
  rclcpp::Time trial_start;
  int trial_n_estops;

  bool offset_unitree_vel;

  // Existing CSV logfile (navigation_metrics.csv)
  std::ofstream logfile_;

  // NEW CSV Logging members (same format as second code)
  std::mutex step_csv_mutex;
  std::string step_csv_filename;
  double trial_start_time_;
  double trial_path_length_;
  double trial_closest_distance_;
  double last_robot_x_;
  double last_robot_y_;
  bool trial_used_brne;

  void pub_walls()
  {
    if (!walls_generated) {
      double x_min = -2.0;
      double x_max = 7.0;
      double y_min = -1.5;
      double y_max = 1.5;
      double room_size = 2.0;
      double thickness = 0.01;
      double height = 1.0;
      double transparency = 0.2;

      double center_x = (x_min + x_max) / 2.0;
      double gap_start_x = center_x - room_size / 2.0;
      double gap_end_x = center_x + room_size / 2.0;

      // Create horizontal corridor walls with central gap
      visualization_msgs::msg::Marker lower_left_wall;
      lower_left_wall.header.frame_id = "brne_odom";
      lower_left_wall.header.stamp = this->now();
      lower_left_wall.ns = "walls";
      lower_left_wall.id = 0;
      lower_left_wall.type = visualization_msgs::msg::Marker::CUBE;
      lower_left_wall.action = visualization_msgs::msg::Marker::ADD;
      lower_left_wall.pose.position.x = (x_min + gap_start_x) / 2.0;
      lower_left_wall.pose.position.y = y_min;
      lower_left_wall.pose.position.z = height / 2.0;
      lower_left_wall.scale.x = gap_start_x - x_min;
      lower_left_wall.scale.y = thickness;
      lower_left_wall.scale.z = height;
      lower_left_wall.color.a = transparency;
      lower_left_wall.color.b = 1.0;
      wall_markers.markers.push_back(lower_left_wall);

      visualization_msgs::msg::Marker lower_right_wall;
      lower_right_wall.header.frame_id = "brne_odom";
      lower_right_wall.header.stamp = this->now();
      lower_right_wall.ns = "walls";
      lower_right_wall.id = 1;
      lower_right_wall.type = visualization_msgs::msg::Marker::CUBE;
      lower_right_wall.action = visualization_msgs::msg::Marker::ADD;
      lower_right_wall.pose.position.x = (gap_end_x + x_max) / 2.0;
      lower_right_wall.pose.position.y = y_min;
      lower_right_wall.pose.position.z = height / 2.0;
      lower_right_wall.scale.x = x_max - gap_end_x;
      lower_right_wall.scale.y = thickness;
      lower_right_wall.scale.z = height;
      lower_right_wall.color.a = transparency;
      lower_right_wall.color.b = 1.0;
      wall_markers.markers.push_back(lower_right_wall);

      visualization_msgs::msg::Marker upper_left_wall;
      upper_left_wall.header.frame_id = "brne_odom";
      upper_left_wall.header.stamp = this->now();
      upper_left_wall.ns = "walls";
      upper_left_wall.id = 2;
      upper_left_wall.type = visualization_msgs::msg::Marker::CUBE;
      upper_left_wall.action = visualization_msgs::msg::Marker::ADD;
      upper_left_wall.pose.position.x = (x_min + gap_start_x) / 2.0;
      upper_left_wall.pose.position.y = y_max;
      upper_left_wall.pose.position.z = height / 2.0;
      upper_left_wall.scale.x = gap_start_x - x_min;
      upper_left_wall.scale.y = thickness;
      upper_left_wall.scale.z = height;
      upper_left_wall.color.a = transparency;
      upper_left_wall.color.b = 1.0;
      wall_markers.markers.push_back(upper_left_wall);

      visualization_msgs::msg::Marker upper_right_wall;
      upper_right_wall.header.frame_id = "brne_odom";
      upper_right_wall.header.stamp = this->now();
      upper_right_wall.ns = "walls";
      upper_right_wall.id = 3;
      upper_right_wall.type = visualization_msgs::msg::Marker::CUBE;
      upper_right_wall.action = visualization_msgs::msg::Marker::ADD;
      upper_right_wall.pose.position.x = (gap_end_x + x_max) / 2.0;
      upper_right_wall.pose.position.y = y_max;
      upper_right_wall.pose.position.z = height / 2.0;
      upper_right_wall.scale.x = x_max - gap_end_x;
      upper_right_wall.scale.y = thickness;
      upper_right_wall.scale.z = height;
      upper_right_wall.color.a = transparency;
      upper_right_wall.color.b = 1.0;
      wall_markers.markers.push_back(upper_right_wall);

      visualization_msgs::msg::Marker coffee;
      coffee.header.frame_id = "brne_odom";
      coffee.header.stamp = this->now();
      coffee.ns = "obstacles";
      coffee.id = 4;
      coffee.type = visualization_msgs::msg::Marker::CUBE;
      coffee.action = visualization_msgs::msg::Marker::ADD;
      coffee.pose.position.x = 5.00189;
      coffee.pose.position.y = 1.46894;
      coffee.pose.position.z = 0.5;
      coffee.scale.x = 0.5;
      coffee.scale.y = 0.5;
      coffee.scale.z = 1.0;
      coffee.color.a = 0.7;
      coffee.color.r = 0.5;
      coffee.color.g = 0.3;
      coffee.color.b = 0.1;
      wall_markers.markers.push_back(coffee);

      visualization_msgs::msg::Marker lower_room_left;
      lower_room_left.header.frame_id = "brne_odom";
      lower_room_left.header.stamp = this->now();
      lower_room_left.ns = "rooms";
      lower_room_left.id = 5;
      lower_room_left.type = visualization_msgs::msg::Marker::CUBE;
      lower_room_left.action = visualization_msgs::msg::Marker::ADD;
      lower_room_left.pose.position.x = center_x - room_size / 2.0;
      lower_room_left.pose.position.y = y_min - room_size / 2.0;
      lower_room_left.pose.position.z = height / 2.0;
      lower_room_left.scale.x = thickness;
      lower_room_left.scale.y = room_size;
      lower_room_left.scale.z = height;
      lower_room_left.color.a = transparency;
      lower_room_left.color.b = 1.0;
      wall_markers.markers.push_back(lower_room_left);

      visualization_msgs::msg::Marker lower_room_right;
      lower_room_right.header.frame_id = "brne_odom";
      lower_room_right.header.stamp = this->now();
      lower_room_right.ns = "rooms";
      lower_room_right.id = 6;
      lower_room_right.type = visualization_msgs::msg::Marker::CUBE;
      lower_room_right.action = visualization_msgs::msg::Marker::ADD;
      lower_room_right.pose.position.x = center_x + room_size / 2.0;
      lower_room_right.pose.position.y = y_min - room_size / 2.0;
      lower_room_right.pose.position.z = height / 2.0;
      lower_room_right.scale.x = thickness;
      lower_room_right.scale.y = room_size;
      lower_room_right.scale.z = height;
      lower_room_right.color.a = transparency;
      lower_room_right.color.b = 1.0;
      wall_markers.markers.push_back(lower_room_right);

      visualization_msgs::msg::Marker lower_room_back;
      lower_room_back.header.frame_id = "brne_odom";
      lower_room_back.header.stamp = this->now();
      lower_room_back.ns = "rooms";
      lower_room_back.id = 7;
      lower_room_back.type = visualization_msgs::msg::Marker::CUBE;
      lower_room_back.action = visualization_msgs::msg::Marker::ADD;
      lower_room_back.pose.position.x = center_x;
      lower_room_back.pose.position.y = y_min - room_size;
      lower_room_back.pose.position.z = height / 2.0;
      lower_room_back.scale.x = room_size;
      lower_room_back.scale.y = thickness;
      lower_room_back.scale.z = height;
      lower_room_back.color.a = transparency;
      lower_room_back.color.b = 1.0;
      wall_markers.markers.push_back(lower_room_back);

      visualization_msgs::msg::Marker upper_room_left;
      upper_room_left.header.frame_id = "brne_odom";
      upper_room_left.header.stamp = this->now();
      upper_room_left.ns = "rooms";
      upper_room_left.id = 8;
      upper_room_left.type = visualization_msgs::msg::Marker::CUBE;
      upper_room_left.action = visualization_msgs::msg::Marker::ADD;
      upper_room_left.pose.position.x = center_x - room_size / 2.0;
      upper_room_left.pose.position.y = y_max + room_size / 2.0;
      upper_room_left.pose.position.z = height / 2.0;
      upper_room_left.scale.x = thickness;
      upper_room_left.scale.y = room_size;
      upper_room_left.scale.z = height;
      upper_room_left.color.a = transparency;
      upper_room_left.color.b = 1.0;
      wall_markers.markers.push_back(upper_room_left);

      visualization_msgs::msg::Marker upper_room_right;
      upper_room_right.header.frame_id = "brne_odom";
      upper_room_right.header.stamp = this->now();
      upper_room_right.ns = "rooms";
      upper_room_right.id = 9;
      upper_room_right.type = visualization_msgs::msg::Marker::CUBE;
      upper_room_right.action = visualization_msgs::msg::Marker::ADD;
      upper_room_right.pose.position.x = center_x + room_size / 2.0;
      upper_room_right.pose.position.y = y_max + room_size / 2.0;
      upper_room_right.pose.position.z = height / 2.0;
      upper_room_right.scale.x = thickness;
      upper_room_right.scale.y = room_size;
      upper_room_right.scale.z = height;
      upper_room_right.color.a = transparency;
      upper_room_right.color.b = 1.0;
      wall_markers.markers.push_back(upper_room_right);

      visualization_msgs::msg::Marker upper_room_back;
      upper_room_back.header.frame_id = "brne_odom";
      upper_room_back.header.stamp = this->now();
      upper_room_back.ns = "rooms";
      upper_room_back.id = 10;
      upper_room_back.type = visualization_msgs::msg::Marker::CUBE;
      upper_room_back.action = visualization_msgs::msg::Marker::ADD;
      upper_room_back.pose.position.x = center_x;
      upper_room_back.pose.position.y = y_max + room_size;
      upper_room_back.pose.position.z = height / 2.0;
      upper_room_back.scale.x = room_size;
      upper_room_back.scale.y = thickness;
      upper_room_back.scale.z = height;
      upper_room_back.color.a = transparency;
      upper_room_back.color.b = 1.0;
      wall_markers.markers.push_back(upper_room_back);

      walls_generated = true;
    }
    walls_pub_->publish(wall_markers);
  }

  void goal_cb(const geometry_msgs::msg::PoseStamped & msg)
  {
    RCLCPP_INFO_STREAM(get_logger(), "Goal Received: " << msg.pose.position.x << ", " << msg.pose.position.y);
    goal_set = true;
    goal = msg;
    check_goal();
    trial_start = this->get_clock()->now();
    trial_start_pose = robot_pose;
    trial_path_length = 0;
    trial_closest_dst_to_ped = 10000;
    trial_n_estops = 0;
  }

  void odom_cb(const nav_msgs::msg::Odometry & msg)
  {
    if (goal_set){
      trial_path_length += dist(robot_pose.x, robot_pose.y, msg.pose.pose.position.x, msg.pose.pose.position.y);
    }
    tf2::Quaternion q(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
      msg.pose.pose.orientation.z, msg.pose.pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    robot_pose.x = msg.pose.pose.position.x;
    robot_pose.y = msg.pose.pose.position.y;
    robot_pose.theta = yaw;
    if (goal_set) {
      check_goal();
    }
  }

  void check_goal()
  {
    const auto dist_to_goal = dist(robot_pose.x, robot_pose.y, goal.pose.position.x, goal.pose.position.y);
    if (dist_to_goal < goal_threshold) {
      const auto trial_end = this->get_clock()->now();
      RCLCPP_INFO_STREAM(get_logger(), "Goal Reached!");
      const auto trial_dt = trial_end - trial_start;
      trial_straight_line_length = dist(trial_start_pose.x, trial_start_pose.y, robot_pose.x, robot_pose.y);
      RCLCPP_INFO_STREAM(get_logger(), "=========================================================");
      RCLCPP_INFO_STREAM(get_logger(), "Time: " << trial_dt.seconds() << " s");
      RCLCPP_INFO_STREAM(get_logger(), "Straight Line Path: " << trial_straight_line_length << " m");
      RCLCPP_INFO_STREAM(get_logger(), "Trial Path: " << trial_path_length << " m");
      RCLCPP_INFO_STREAM(get_logger(), "Path Ratio: " << trial_path_length/trial_straight_line_length);
      RCLCPP_INFO_STREAM(get_logger(), "Closest Dist to Ped: " << trial_closest_dst_to_ped << " m");
      RCLCPP_INFO_STREAM(get_logger(), "Number of E-STOPs: " << trial_n_estops);
      RCLCPP_INFO_STREAM(get_logger(), "=========================================================");
      goal_set = false;
    }
  }

  void pedestrians_cb(const crowd_nav_interfaces::msg::PedestrianArray & msg)
  {
    const auto curr_ped_stamp = this->get_clock()->now();
    const int n_peds = msg.pedestrians.size();
    for (int p = 0; p < n_peds; p++) {
      auto ped = msg.pedestrians.at(p);
      ped.header.stamp = curr_ped_stamp;
      while (static_cast<int>(ped_buffer.pedestrians.size()) < static_cast<int>(ped.id + 1)) {
        crowd_nav_interfaces::msg::Pedestrian blank_ped;
        blank_ped.id = ped_buffer.pedestrians.size();
        ped_buffer.pedestrians.push_back(blank_ped);
      }
      ped_buffer.pedestrians.at(ped.id) = ped;
    }
  }

  void timer_callback()
  {
    const auto start = this->get_clock()->now();
    robot_cmds.twists.clear();
    selected_peds.pedestrians.clear();

    const builtin_interfaces::msg::Time current_timestamp = this->get_clock()->now();
    const auto current_time_sec = current_timestamp.sec + 1e-9 * current_timestamp.nanosec;

    std::vector<double> dists_to_peds;
    for (auto p:ped_buffer.pedestrians) {
      auto ped_time_sec = p.header.stamp.sec + 1e-9 * p.header.stamp.nanosec;
      auto dt = current_time_sec - ped_time_sec;
      if (dt > people_timeout) continue;
      auto dist_to_ped = dist(robot_pose.x, robot_pose.y, p.pose.position.x, p.pose.position.y);
      if (dist_to_ped > brne_activate_threshold) continue;
      dists_to_peds.push_back(dist_to_ped);
      selected_peds.pedestrians.push_back(p);
    }

    const auto n_peds = static_cast<int>(selected_peds.pedestrians.size());
    const auto n_agents = std::min(maximum_agents, n_peds + 1);

    arma::rowvec goal_vec;
    if (goal_set) {
      goal_vec = arma::rowvec({goal.pose.position.x, goal.pose.position.y});
    } else {
      goal_vec = arma::rowvec({10.0, 0.0});
    }

    auto theta_a = robot_pose.theta;
    if (robot_pose.theta > 0.0) {
      theta_a -= M_PI_2;
    } else {
      theta_a += M_PI_2;
    }
    const arma::rowvec axis_vec(std::vector<double>{cos(theta_a), sin(theta_a)});
    const arma::rowvec pose_vec(std::vector<double>{robot_pose.x, robot_pose.y});
    const arma::rowvec vec_to_goal = goal_vec - pose_vec;
    const auto dist_to_goal = arma::norm(vec_to_goal);
    const auto proj_len = arma::dot(axis_vec, vec_to_goal) / arma::dot(vec_to_goal, vec_to_goal) * dist_to_goal;
    const auto radius = 0.5 * dist_to_goal / proj_len;
    double nominal_ang_vel = 0;
    if (robot_pose.theta > 0.0) {
      nominal_ang_vel = -nominal_lin_vel / radius;
    } else {
      nominal_ang_vel = nominal_lin_vel / radius;
    }

    const auto traj_samples = trajgen.traj_sample(nominal_lin_vel, nominal_ang_vel, robot_pose.toVec());

    if (n_agents > 1) {
      const auto x_pts = brne.mvn_sample_normal(n_agents - 1);
      const auto y_pts = brne.mvn_sample_normal(n_agents - 1);
      arma::mat xtraj_samples(n_agents * n_samples, n_steps, arma::fill::zeros);
      arma::mat ytraj_samples(n_agents * n_samples, n_steps, arma::fill::zeros);

      const auto closest_idxs = arma::conv_to<arma::vec>::from(arma::sort_index(arma::vec(dists_to_peds)));
      for (int p = 0; p < (n_agents - 1); p++) {
        auto ped = selected_peds.pedestrians.at(closest_idxs.at(p));
        arma::vec ped_vel(std::vector<double>{ped.velocity.linear.x, ped.velocity.linear.y});
        auto speed_factor = arma::norm(ped_vel);
        arma::rowvec ped_xmean = arma::rowvec(n_steps, arma::fill::value(ped.pose.position.x)) +
          arma::linspace<arma::rowvec>(0, (n_steps - 1), n_steps) * dt * ped.velocity.linear.x;
        arma::rowvec ped_ymean = arma::rowvec(n_steps, arma::fill::value(ped.pose.position.y)) +
          arma::linspace<arma::rowvec>(0, (n_steps - 1), n_steps) * dt * ped.velocity.linear.y;
        arma::mat ped_xmean_mat(n_samples, n_steps, arma::fill::zeros);
        arma::mat ped_ymean_mat(n_samples, n_steps, arma::fill::zeros);
        ped_xmean_mat.each_row() = ped_xmean;
        ped_ymean_mat.each_row() = ped_ymean;
        xtraj_samples.submat((p + 1) * n_samples, 0, (p + 2) * n_samples - 1, n_steps - 1) =
          x_pts.submat(p * n_samples, 0, (p + 1) * n_samples - 1, n_steps - 1) * speed_factor + ped_xmean_mat;
        ytraj_samples.submat((p + 1) * n_samples, 0, (p + 2) * n_samples - 1, n_steps - 1) =
          y_pts.submat(p * n_samples, 0, (p + 1) * n_samples - 1, n_steps - 1) * speed_factor + ped_ymean_mat;
      }
      auto robot_xtraj_samples = trajgen.get_xtraj_samples();
      auto robot_ytraj_samples = trajgen.get_ytraj_samples();
      xtraj_samples.submat(0, 0, n_samples - 1, n_steps - 1) = robot_xtraj_samples;
      ytraj_samples.submat(0, 0, n_samples - 1, n_steps - 1) = robot_ytraj_samples;

      const auto closest_ped = selected_peds.pedestrians.at(closest_idxs.at(0));
      const arma::mat robot_samples_to_ped = arma::sqrt(
        arma::pow(robot_xtraj_samples - closest_ped.pose.position.x, 2) +
        arma::pow(robot_ytraj_samples - closest_ped.pose.position.y, 2));
      const auto closest_to_ped = arma::conv_to<arma::vec>::from(arma::min(robot_samples_to_ped, 1));
      auto safety_mask = arma::conv_to<arma::rowvec>::from(closest_to_ped > close_stop_threshold);

      if (goal_set){
        const auto dst_to_closest_ped = dist(robot_pose.x, robot_pose.y, closest_ped.pose.position.x, closest_ped.pose.position.y);
        if (dst_to_closest_ped < trial_closest_dst_to_ped){
          trial_closest_dst_to_ped = dst_to_closest_ped;
        }
      }

      // Additional safety: coffee machine avoidance
      const double coffee_x = 5.00189;
      const double coffee_y = 1.46894;
      const double coffee_radius = 0.3;
      const arma::mat robot_samples_to_coffee = arma::sqrt(
          arma::pow(robot_xtraj_samples - coffee_x, 2) +
          arma::pow(robot_ytraj_samples - coffee_y, 2)
      );
      const auto coffee_mask = arma::conv_to<arma::rowvec>::from(arma::min(robot_samples_to_coffee, 1) > coffee_radius);
      safety_mask %= coffee_mask;

      auto weights = brne.brne_nav(xtraj_samples, ytraj_samples);
      if (weights.row(0).is_zero()){
        if (goal_set){
          RCLCPP_WARN_STREAM(get_logger(), "No path found -- stopping navigation to this goal.");
          goal_set = false;
        }
      } else {
        weights.row(0) %= safety_mask;
        const double mean_weights = arma::mean(weights.row(0));
        if (mean_weights != 0) {
          weights.row(0) /= mean_weights;
        } else {
          if (goal_set) {
            RCLCPP_WARN_STREAM(get_logger(), "E-STOP: Pedestrian too close!");
            trial_n_estops += 1;
          }
        }
      }

      const auto ulist = trajgen.get_ulist();
      const auto ulist_lin = arma::conv_to<arma::rowvec>::from(ulist.col(0));
      const auto ulist_ang = arma::conv_to<arma::rowvec>::from(ulist.col(1));
      const auto opt_cmds_lin = arma::mean(ulist_lin % weights.row(0));
      const auto opt_cmds_ang = arma::mean(ulist_ang % weights.row(0));

      if (goal_set) {
        for (int i = 0; i < n_steps; i++) {
          geometry_msgs::msg::Twist tw;
          tw.linear.x = opt_cmds_lin;
          tw.angular.z = opt_cmds_ang;
          if (offset_unitree_vel){
            if ((opt_cmds_lin > 0.1) && (opt_cmds_lin < 0.3)){
              tw.angular.z -= 0.04;
            } else if ((opt_cmds_lin >= 0.3) && (opt_cmds_lin < 0.5)){
              tw.angular.z -= 0.05;
            } else if (opt_cmds_lin >= 0.5){
              tw.angular.z -= 0.06;
            }
          }
          robot_cmds.twists.push_back(tw);
        }
      }

      arma::mat opt_cmds(n_steps, 2, arma::fill::zeros);
      opt_cmds.col(0) = arma::vec(n_steps, arma::fill::value(opt_cmds_lin));
      opt_cmds.col(1) = arma::vec(n_steps, arma::fill::value(opt_cmds_ang));

      const auto opt_traj = trajgen.sim_traj(robot_pose.toVec(), opt_cmds);
      optimal_path.header.stamp = current_timestamp;
      optimal_path.poses.clear();
      for (int i = 0; i < n_steps; i++) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = current_timestamp;
        ps.header.frame_id = "brne_odom";
        ps.pose.position.x = opt_traj.at(i, 0);
        ps.pose.position.y = opt_traj.at(i, 1);
        optimal_path.poses.push_back(ps);
      }

      // Create pedestrian prediction markers (for visualization)
      visualization_msgs::msg::MarkerArray ped_prediction_markers;
      int marker_id = 0;
      for (int p = 0; p < (n_agents - 1); p++) {
        auto ped = selected_peds.pedestrians.at(closest_idxs.at(p));
        for (int s = 0; s < n_samples; s++) {
          visualization_msgs::msg::Marker line;
          line.header.frame_id = "brne_odom";
          line.header.stamp = this->get_clock()->now();
          line.ns = "ped_" + std::to_string(ped.id);
          line.id = marker_id++;
          line.type = visualization_msgs::msg::Marker::LINE_STRIP;
          line.scale.x = 0.05;
          line.color.r = 1.0;
          line.color.g = 0.0;
          line.color.b = 0.0;
          line.color.a = 0.3;
          for (int t = 0; t < n_steps; t++) {
            geometry_msgs::msg::Point pt;
            pt.x = xtraj_samples((p + 1) * n_samples + s, t);
            pt.y = ytraj_samples((p + 1) * n_samples + s, t);
            line.points.push_back(pt);
          }
          ped_prediction_markers.markers.push_back(line);
        }
        visualization_msgs::msg::Marker link_line;
        link_line.header.frame_id = "brne_odom";
        link_line.header.stamp = this->get_clock()->now();
        link_line.ns = "robot_ped_link_" + std::to_string(ped.id);
        link_line.id = marker_id++;
        link_line.type = visualization_msgs::msg::Marker::LINE_STRIP;
        link_line.scale.x = 0.1;
        link_line.color.r = 0.0;
        link_line.color.g = 1.0;
        link_line.color.b = 0.0;
        link_line.color.a = 0.8;
        geometry_msgs::msg::Point start_pt;
        start_pt.x = robot_pose.x;
        start_pt.y = robot_pose.y;
        link_line.points.push_back(start_pt);
        for (int t = 0; t < n_steps; t++) {
          double sum_x = 0.0;
          double sum_y = 0.0;
          for (int s = 0; s < n_samples; s++) {
            sum_x += xtraj_samples((p + 1) * n_samples + s, t);
            sum_y += ytraj_samples((p + 1) * n_samples + s, t);
          }
          geometry_msgs::msg::Point mean_pt;
          mean_pt.x = sum_x / n_samples;
          mean_pt.y = sum_y / n_samples;
          link_line.points.push_back(mean_pt);
        }
        ped_prediction_markers.markers.push_back(link_line);
      }
      ped_prediction_pub_->publish(ped_prediction_markers);

      trial_used_brne = true;
    }
    else {
      const auto opt_cmds = trajgen.opt_controls(goal_vec);
      if (goal_set) {
        for (int i = 0; i < n_steps; i++) {
          geometry_msgs::msg::Twist tw;
          const auto opt_cmds_lin = opt_cmds.at(i, 0);
          const auto opt_cmds_ang = opt_cmds.at(i, 1);
          tw.linear.x = opt_cmds_lin;
          tw.angular.z = opt_cmds_ang;
          if (offset_unitree_vel){
            if ((opt_cmds_lin > 0.1) && (opt_cmds_lin < 0.3)){
              tw.angular.z -= 0.04;
            } else if ((opt_cmds_lin >= 0.3) && (opt_cmds_lin < 0.5)){
              tw.angular.z -= 0.05;
            } else if (opt_cmds_lin >= 0.5){
              tw.angular.z -= 0.06;
            }
          }
          robot_cmds.twists.push_back(tw);
        }
      }
      const auto opt_traj = trajgen.sim_traj(robot_pose.toVec(), opt_cmds);
      optimal_path.header.stamp = current_timestamp;
      optimal_path.poses.clear();
      for (int i = 0; i < n_steps; i++) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = current_timestamp;
        ps.header.frame_id = "brne_odom";
        ps.pose.position.x = opt_traj.at(i, 0);
        ps.pose.position.y = opt_traj.at(i, 1);
        optimal_path.poses.push_back(ps);
      }
    }
    cmd_buf_pub_->publish(robot_cmds);
    path_pub_->publish(optimal_path);
    pub_walls();

    // --- Existing CSV logging to "navigation_metrics.csv" ---
    double now_sec = this->get_clock()->now().seconds();
    double rx = robot_pose.x;
    double ry = robot_pose.y;
    double delta = dist(last_robot_x_, last_robot_y_, rx, ry);
    trial_path_length_ += delta;
    last_robot_x_ = rx;
    last_robot_y_ = ry;
    for (auto &p : selected_peds.pedestrians) {
      double px = p.pose.position.x;
      double py = p.pose.position.y;
      double d = dist(rx, ry, px, py);
      logfile_ << now_sec << "," << rx << "," << ry << "," << px << "," << py << "," << d << "\n";
    }
    double timestamp = this->get_clock()->now().seconds();
    double robot_x = robot_pose.x;
    double robot_y = robot_pose.y;
    double current_linear_vel = 0.0;
    double current_angular_vel = 0.0;
    for (auto &ped : selected_peds.pedestrians) {
      double ped_x = ped.pose.position.x;
      double ped_y = ped.pose.position.y;
      double distance = dist(robot_x, robot_y, ped_x, ped_y);
      std::string zone = "social";
      if (distance < 0.5) zone = "intimate";
      else if (distance < 1.5) zone = "personal";
      logfile_ << timestamp << ","
               << robot_x << "," << robot_y << ","
               << ped_x << "," << ped_y << ","
               << distance << "," << zone << ","
               << current_linear_vel << "," << current_angular_vel << "\n";
    }
    // --- NEW CSV logging (same format as second code) ---
    double cmd_velocity = 0.0;
    if (!robot_cmds.twists.empty())
      cmd_velocity = robot_cmds.twists.front().linear.x;
    double ros_time = this->get_clock()->now().seconds();
    std::ostringstream ped_positions;
    for (const auto &ped : selected_peds.pedestrians) {
      double d = dist(robot_pose.x, robot_pose.y, ped.pose.position.x, ped.pose.position.y);
      std::string area = (d < close_stop_threshold) ? "personal" : "social";
      ped_positions << ped.id << ":" 
                    << ped.pose.position.x << "," 
                    << ped.pose.position.y << ",d=" 
                    << d << ",area=" << area << ";";
    }
    {
      std::lock_guard<std::mutex> lock(step_csv_mutex);
      std::ofstream step_file(step_csv_filename, std::ios::app);
      if (step_file.is_open()) {
        step_file << std::fixed << std::setprecision(6);
        step_file << ros_time << ","
                  << robot_pose.x << ","
                  << robot_pose.y << ","
                  << robot_pose.theta << ","
                  << cmd_velocity << ","
                  << selected_peds.pedestrians.size() << ","
                  << ped_positions.str() << ","
                  << (trial_used_brne ? "true" : "false") << "\n";
      }
    }

    const auto end = this->get_clock()->now();
    const auto diff = end - start;
    RCLCPP_DEBUG_STREAM(get_logger(), "Agents: " << n_agents << " Timer: " << diff.seconds() << " s");
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathPlan>());
  rclcpp::shutdown();
  return 0;
}