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
    declare_parameter("ped_sample_scale", 1.0);
    declare_parameter("open_space_velocity", nominal_lin_vel);  // new open-space param
    declare_parameter("tree_x", 2.0);
    declare_parameter("tree_y", 3.0);
    declare_parameter("tree_radius", 0.5);

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
    ped_sample_scale = get_parameter("ped_sample_scale").as_double();
    open_space_velocity = get_parameter("open_space_velocity").as_double();
    tree_x = get_parameter("tree_x").as_double();
    tree_y = get_parameter("tree_y").as_double();
    tree_radius = get_parameter("tree_radius").as_double();

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
    RCLCPP_INFO_STREAM(get_logger(), "Pedestrian Sample Scale: " << ped_sample_scale);

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
    std::chrono::milliseconds rate = (std::chrono::milliseconds)((int)(1000. / replan_freq));
    timer_ = create_wall_timer(rate, std::bind(&PathPlan::timer_callback, this));

    optimal_path.header.frame_id = "brne_odom";

    // --- NEW CSV Logging: Open CSV file for step data logging ---
    step_csv_filename = "output_data_event.csv";
    {
      std::lock_guard<std::mutex> lock(step_csv_mutex);
      std::ofstream step_file(step_csv_filename);
      if (step_file.is_open()) {
        step_file << "RosTime,RobotX,RobotY,RobotTheta,RobotVelocity,NumPedestrians,PedestrianPositions,BRNE_Used\n";
      }
    }
    // Initialize trial metrics for CSV logging
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
         nominal_lin_vel, close_stop_threshold, ped_sample_scale, open_space_velocity;
  int maximum_agents, n_samples, n_steps;

  double tree_x, tree_y, tree_radius;

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

  // NEW CSV Logging members (same as second code)
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
      const double height = 2.0;
      const double thickness = 1.0;
      const double building_length = 5.0;

      visualization_msgs::msg::Marker building;
      building.header.frame_id = "brne_odom";
      building.id = 0;
      building.type = visualization_msgs::msg::Marker::CUBE;
      building.action = visualization_msgs::msg::Marker::ADD;

      building.pose.position.x = 5;
      building.pose.position.y = 5.03631;
      building.pose.position.z = height / 2.0;

      tf2::Quaternion q;
      q.setRPY(0, 0, M_PI / 2);  // 90 degrees in radians
      building.pose.orientation.x = q.x();
      building.pose.orientation.y = q.y();
      building.pose.orientation.z = q.z();
      building.pose.orientation.w = q.w();

      building.color.a = 1.0;
      building.color.r = 1.0;
      building.color.g = 1.0;
      building.color.b = 1.0;

      building.scale.x = thickness;
      building.scale.y = building_length;
      building.scale.z = height;

      wall_markers.markers.push_back(building);
      walls_generated = true;
    }

    const auto now = this->get_clock()->now();
    for (auto & marker : wall_markers.markers) {
      marker.header.stamp = now;
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
      trial_path_length += dist(robot_pose.x, robot_pose.y, 
                                msg.pose.pose.position.x, msg.pose.pose.position.y);
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

    for (const auto & incoming : msg.pedestrians) {
      auto ped = incoming; // make a non-const copy
      ped.header.stamp = curr_ped_stamp;

      while ((int)ped_buffer.pedestrians.size() <= ped.id) {
        crowd_nav_interfaces::msg::Pedestrian blank_ped;
        blank_ped.id = ped_buffer.pedestrians.size();
        ped_buffer.pedestrians.push_back(blank_ped);
      }

      ped_buffer.pedestrians[ped.id] = std::move(ped);
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
    for (auto p: ped_buffer.pedestrians) {
      auto ped_time_sec = p.header.stamp.sec + 1e-9 * p.header.stamp.nanosec;
      auto dt = current_time_sec - ped_time_sec;
      if (dt > people_timeout) {
        continue;
      }
      auto dist_to_ped = dist(robot_pose.x, robot_pose.y, p.pose.position.x, p.pose.position.y);
      if (dist_to_ped > brne_activate_threshold) {
        continue;
      }
      dists_to_peds.push_back(dist_to_ped);
      selected_peds.pedestrians.push_back(p);
    }

    const auto n_peds = static_cast<int>(selected_peds.pedestrians.size());
    const auto n_agents = std::min(maximum_agents, n_peds + 1);

    arma::rowvec goal_vec;
    if (goal_set) {
      goal_vec = arma::rowvec(std::vector<double>{goal.pose.position.x, goal.pose.position.y});
    } else {
      goal_vec = arma::rowvec(std::vector<double>{10.0, 0.0});
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
          x_pts.submat(p * n_samples, 0, (p + 1) * n_samples - 1, n_steps - 1) * speed_factor * ped_sample_scale + ped_xmean_mat;
        ytraj_samples.submat((p + 1) * n_samples, 0, (p + 2) * n_samples - 1, n_steps - 1) =
          y_pts.submat(p * n_samples, 0, (p + 1) * n_samples - 1, n_steps - 1) * speed_factor * ped_sample_scale + ped_ymean_mat;
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
      const auto safety_mask = arma::conv_to<arma::rowvec>::from(closest_to_ped > close_stop_threshold);

      if (goal_set){
        const auto dst_to_closest_ped = dist(robot_pose.x, robot_pose.y, 
          closest_ped.pose.position.x, closest_ped.pose.position.y);
        if (dst_to_closest_ped < trial_closest_dst_to_ped){
          trial_closest_dst_to_ped = dst_to_closest_ped;
        }
      }

      // Build a mask for all pedestrians
      arma::rowvec ped_mask(n_samples, arma::fill::ones);
      for (const auto &ped : selected_peds.pedestrians) {
        arma::mat dist_to_ped = arma::sqrt(
          arma::pow(robot_xtraj_samples - ped.pose.position.x, 2) +
          arma::pow(robot_ytraj_samples - ped.pose.position.y, 2));
        arma::rowvec this_mask = arma::conv_to<arma::rowvec>::from(
          arma::min(dist_to_ped, 1) > close_stop_threshold);
        ped_mask %= this_mask;
      }

      // Coffee mask (unchanged)
      const double coffee_x = 5.00189;
      const double coffee_y = 1.46894;
      const double coffee_radius = 0.3;
      arma::mat dist_to_coffee = arma::sqrt(
        arma::pow(robot_xtraj_samples - coffee_x, 2) +
        arma::pow(robot_ytraj_samples - coffee_y, 2));
      arma::rowvec coffee_mask = arma::conv_to<arma::rowvec>::from(
        arma::min(dist_to_coffee, 1) > coffee_radius);

      // Tree mask (unchanged)
      arma::mat dist_to_tree = arma::sqrt(
        arma::pow(robot_xtraj_samples - tree_x, 2) +
        arma::pow(robot_ytraj_samples - tree_y, 2));
      arma::rowvec tree_mask = arma::conv_to<arma::rowvec>::from(
        arma::min(dist_to_tree, 1) > tree_radius);

      // Combine all
      arma::rowvec combined_mask = ped_mask % coffee_mask % tree_mask;

      // Fallback if tree blocks every sample
      if (arma::all(combined_mask == 0)) {
        RCLCPP_WARN_STREAM(get_logger(), "Tree blocks all samples; ignoring tree avoidance this cycle");
        combined_mask = ped_mask % coffee_mask;
      }

      // Now apply it to weights
      auto weights = brne.brne_nav(xtraj_samples, ytraj_samples);
      if (!weights.row(0).is_zero()) {
        weights.row(0) %= combined_mask;
        double mean_w = arma::mean(weights.row(0));
        if (mean_w != 0) {
          weights.row(0) /= mean_w;
        } else if (goal_set) {
          RCLCPP_WARN_STREAM(get_logger(), "E-STOP: too close!");
          trial_n_estops++;
        }
      } else if (goal_set) {
        RCLCPP_WARN_STREAM(get_logger(), "No path found -- stopping navigation to this goal.");
        goal_set = false;
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
      // No pedestrians (n_agents <= 1) => constant open-space velocity
      const auto opt_cmds = trajgen.opt_controls(goal_vec);
      if (goal_set) {
        for (int i = 0; i < n_steps; i++) {
          geometry_msgs::msg::Twist tw;
          tw.linear.x = open_space_velocity;
          tw.angular.z = opt_cmds.at(i, 1);
          if (offset_unitree_vel) {
            if      (tw.linear.x > 0.1 && tw.linear.x < 0.3) tw.angular.z -= 0.04;
            else if (tw.linear.x >= 0.3 && tw.linear.x < 0.5) tw.angular.z -= 0.05;
            else if (tw.linear.x >= 0.5)                     tw.angular.z -= 0.06;
          }
          robot_cmds.twists.push_back(tw);
        }
      }
      auto opt_traj = trajgen.sim_traj(robot_pose.toVec(), opt_cmds);
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

    const auto end = this->get_clock()->now();
    const auto diff = end - start;
    RCLCPP_DEBUG_STREAM(get_logger(), "Agents: " << n_agents << " Timer: " << diff.seconds() << " s");

    // --- NEW CSV Logging (same format as second code) ---
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
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathPlan>());
  rclcpp::shutdown();
  return 0;
}


