#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <armadillo>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <mutex>
#include <sstream>
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
  : Node("brne"), goal_set{false}, crosswalk_generated{false}
  {
    // --- Parameter declarations ---
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

    // --- Get parameters ---
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

    RCLCPP_INFO_STREAM(get_logger(), "Replan frequency: " << replan_freq << " Hz");
    RCLCPP_INFO_STREAM(get_logger(), "dt: " << dt);
    RCLCPP_INFO_STREAM(get_logger(), "Maximum agents: " << maximum_agents);
    RCLCPP_INFO_STREAM(get_logger(), "Number of samples: " << n_samples);
    RCLCPP_INFO_STREAM(get_logger(), "Number of timesteps: " << n_steps);
    RCLCPP_INFO_STREAM(get_logger(), "Costs: " << cost_a1 << " " << cost_a2 << " " << cost_a3);
    RCLCPP_INFO_STREAM(get_logger(), "Kernels: " << kernel_a1 << " " << kernel_a2);
    RCLCPP_INFO_STREAM(get_logger(), "Hallway: " << y_min << " -> " << y_max);
    RCLCPP_INFO_STREAM(get_logger(), "People timeout after " << people_timeout << " s");
    RCLCPP_INFO_STREAM(get_logger(), "Goal Threshold " << goal_threshold << " m");
    RCLCPP_INFO_STREAM(get_logger(), "Close stop threshold " << close_stop_threshold << " m");
    RCLCPP_INFO_STREAM(get_logger(), "Brne Activate Threshold " << brne_activate_threshold << " m");
    RCLCPP_INFO_STREAM(get_logger(),
      "Max Lin: " << max_lin_vel << " nominal lin: " << nominal_lin_vel << " max ang: " << max_ang_vel);
    RCLCPP_INFO_STREAM(get_logger(), "Offset Unitree Velocity? " << offset_unitree_vel);

    // --- Initialize BRNE objects ---
    brne = brne::BRNE{kernel_a1, kernel_a2,
      cost_a1, cost_a2, cost_a3,
      dt, n_steps, n_samples,
      y_min, y_max};

    trajgen = brne::TrajGen{max_lin_vel, max_ang_vel, n_samples, n_steps, dt};

    RCLCPP_INFO_STREAM(get_logger(), brne.param_string());

    // --- Define publishers and subscribers ---
    pedestrian_sub_ = create_subscription<crowd_nav_interfaces::msg::PedestrianArray>(
      "pedestrians", 10,
      std::bind(&PathPlan::pedestrians_cb, this, std::placeholders::_1));
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "goal_pose", 10,
      std::bind(&PathPlan::goal_cb, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "brne_odom", 10,
      std::bind(&PathPlan::odom_cb, this, std::placeholders::_1));
    cmd_buf_pub_ = create_publisher<crowd_nav_interfaces::msg::TwistArray>("cmd_buf", 10);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/optimal_path", 10);
    // Publish crosswalk markers on topic "/crosswalk"
    walls_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/crosswalk", 10);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    std::chrono::milliseconds rate(static_cast<int>(1000.0 / replan_freq));
    timer_ = create_wall_timer(rate, std::bind(&PathPlan::timer_callback, this));

    optimal_path.header.frame_id = "brne_odom";

    // Navigation data file has been cancelled. (Removed csv_filename initialization and CSV file setup)
  }

private:
  // Parameters
  double replan_freq, kernel_a1, kernel_a2, cost_a1, cost_a2, cost_a3, y_min, y_max, dt,
         max_ang_vel, max_lin_vel, people_timeout, goal_threshold, brne_activate_threshold,
         nominal_lin_vel, close_stop_threshold;
  int maximum_agents, n_samples, n_steps;
  bool offset_unitree_vel;
  double robot_velocity; 

  // BRNE objects
  brne::BRNE brne{};
  brne::TrajGen trajgen{};

  // Timer and publishers/subscribers
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<crowd_nav_interfaces::msg::PedestrianArray>::SharedPtr pedestrian_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<crowd_nav_interfaces::msg::TwistArray>::SharedPtr cmd_buf_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr walls_pub_;

  // Buffers and state
  crowd_nav_interfaces::msg::PedestrianArray ped_buffer;
  crowd_nav_interfaces::msg::PedestrianArray selected_peds;
  crowd_nav_interfaces::msg::TwistArray robot_cmds;
  nav_msgs::msg::Path optimal_path;
  RobotPose robot_pose;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  visualization_msgs::msg::MarkerArray wall_markers;
  bool goal_set;
  bool crosswalk_generated;
  geometry_msgs::msg::PoseStamped goal;

  // Trial information
  double trial_closest_dst_to_ped;
  double trial_path_length;
  RobotPose trial_start_pose;
  double trial_straight_line_length;
  double trial_path_ratio;
  rclcpp::Time trial_start;
  int trial_n_estops;

  // CSV-related members for step data only
  std::string step_csv_filename;
  std::mutex step_csv_mutex;
  int trial_max_people;
  bool trial_used_brne;

  void goal_cb(const geometry_msgs::msg::PoseStamped & msg)
  {
    RCLCPP_INFO_STREAM(get_logger(),
      "Goal Received: " << msg.pose.position.x << ", " << msg.pose.position.y);
    goal_set = true;
    goal = msg;
    check_goal();
    trial_start = this->get_clock()->now();
    trial_start_pose = robot_pose;
    trial_path_length = 0;
    trial_closest_dst_to_ped = 10000;
    trial_n_estops = 0;
    trial_max_people = 0;
    trial_used_brne = false;
    
    // Always use the fixed name "output_data_shibuya.csv" for the step data file.
    step_csv_filename = "output_data_shibuya.csv";
    std::lock_guard<std::mutex> lock(step_csv_mutex);
    std::ofstream step_file(step_csv_filename);
    if (step_file.is_open()) {
      step_file << "RosTime,RobotX,RobotY,RobotTheta,RobotVelocity,NumPedestrians,PedestrianPositions,BRNE_Used\n";
    }
  }

  void odom_cb(const nav_msgs::msg::Odometry & msg)
  {
    if (goal_set) {
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
    robot_velocity = std::hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y);
    if (goal_set) {
      check_goal();
    }
  }

  void check_goal()
  {
    const auto dist_to_goal = dist(robot_pose.x, robot_pose.y,
                                    goal.pose.position.x, goal.pose.position.y);
    if (dist_to_goal < goal_threshold) {
      const auto trial_end = this->get_clock()->now();
      RCLCPP_INFO_STREAM(get_logger(), "Goal Reached!");
      const auto trial_dt = trial_end - trial_start;
      trial_straight_line_length = dist(trial_start_pose.x, trial_start_pose.y,
                                         robot_pose.x, robot_pose.y);
      trial_path_ratio = trial_path_length / trial_straight_line_length;
      RCLCPP_INFO_STREAM(get_logger(), "=========================================================");
      RCLCPP_INFO_STREAM(get_logger(), "Time: " << trial_dt.seconds() << " s");
      RCLCPP_INFO_STREAM(get_logger(), "Straight Line Path: " << trial_straight_line_length << " m");
      RCLCPP_INFO_STREAM(get_logger(), "Trial Path: " << trial_path_length << " m");
      RCLCPP_INFO_STREAM(get_logger(), "Path Ratio: " << trial_path_ratio);
      RCLCPP_INFO_STREAM(get_logger(), "Closest Dist to Ped: " << trial_closest_dst_to_ped << " m");
      RCLCPP_INFO_STREAM(get_logger(), "Number of E-STOPs: " << trial_n_estops);
      RCLCPP_INFO_STREAM(get_logger(), "=========================================================");
      goal_set = false;

      // Navigation data recording has been cancelled.
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
    for (auto p : ped_buffer.pedestrians) {
      auto ped_time_sec = p.header.stamp.sec + 1e-9 * p.header.stamp.nanosec;
      auto dt = current_time_sec - ped_time_sec;
      if (dt > people_timeout) continue;

      // Check for moving pedestrians by speed
      double speed = std::hypot(p.velocity.linear.x, p.velocity.linear.y);
      if (speed < 0.1) {  // Adjust threshold as needed
        continue;
      }

      double dist_to_ped = dist(robot_pose.x, robot_pose.y,
                                p.pose.position.x, p.pose.position.y);
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
    const arma::rowvec axis_vec({std::cos(theta_a), std::sin(theta_a)});
    const arma::rowvec pose_vec({robot_pose.x, robot_pose.y});
    const arma::rowvec vec_to_goal = goal_vec - pose_vec;
    const auto dist_to_goal = arma::norm(vec_to_goal);
    const auto proj_len = arma::dot(axis_vec, vec_to_goal) / arma::dot(vec_to_goal, vec_to_goal) * dist_to_goal;
    const auto radius = 0.5 * dist_to_goal / proj_len;
    double nominal_ang_vel = (robot_pose.theta > 0.0) ? -nominal_lin_vel / radius : nominal_lin_vel / radius;

    const auto traj_samples = trajgen.traj_sample(nominal_lin_vel, nominal_ang_vel, robot_pose.toVec());

    if (n_agents > 1) {
      const auto x_pts = brne.mvn_sample_normal(n_agents - 1);
      const auto y_pts = brne.mvn_sample_normal(n_agents - 1);
      arma::mat xtraj_samples(n_agents * n_samples, n_steps, arma::fill::zeros);
      arma::mat ytraj_samples(n_agents * n_samples, n_steps, arma::fill::zeros);

      const auto closest_idxs = arma::conv_to<arma::vec>::from(arma::sort_index(arma::vec(dists_to_peds)));
      for (int p = 0; p < (n_agents - 1); p++) {
        auto ped = selected_peds.pedestrians.at(closest_idxs.at(p));
        arma::vec ped_vel({ped.velocity.linear.x, ped.velocity.linear.y});
        auto speed_factor = arma::norm(ped_vel);
        arma::rowvec ped_xmean = arma::rowvec(n_steps, arma::fill::value(ped.pose.position.x)) +
          arma::linspace<arma::rowvec>(0, n_steps - 1, n_steps) * dt * ped.velocity.linear.x;
        arma::rowvec ped_ymean = arma::rowvec(n_steps, arma::fill::value(ped.pose.position.y)) +
          arma::linspace<arma::rowvec>(0, n_steps - 1, n_steps) * dt * ped.velocity.linear.y;
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
      const auto safety_mask = arma::conv_to<arma::rowvec>::from(closest_to_ped > close_stop_threshold);

      if (goal_set) {
        const auto dst_to_closest_ped = dist(robot_pose.x, robot_pose.y,
                                             closest_ped.pose.position.x, closest_ped.pose.position.y);
        if (dst_to_closest_ped < trial_closest_dst_to_ped) {
          trial_closest_dst_to_ped = dst_to_closest_ped;
        }
      }

      auto weights = brne.brne_nav(xtraj_samples, ytraj_samples);
      const double E_STOP_EPSILON = 1e-3;
      if (weights.row(0).is_zero()) {
        if (goal_set) {
          RCLCPP_WARN_STREAM(get_logger(), "No path found -- stopping navigation to this goal.");
          goal_set = false;
        }
      } else {
        weights.row(0) %= safety_mask;
        const double mean_weights = arma::mean(weights.row(0));
        if (mean_weights > E_STOP_EPSILON) {
          weights.row(0) /= mean_weights;
        } else {
          if (goal_set) {
            RCLCPP_WARN_STREAM(get_logger(), "E-STOP: Pedestrian too close!");
            trial_n_estops += 1;
            robot_cmds.twists.clear();
            for (int i = 0; i < n_steps; i++) {
              geometry_msgs::msg::Twist tw;
              tw.linear.x = 0.0;
              tw.angular.z = 0.0;
              robot_cmds.twists.push_back(tw);
            }
            cmd_buf_pub_->publish(robot_cmds);
            optimal_path.header.stamp = current_timestamp;
            optimal_path.poses.clear();
            geometry_msgs::msg::PoseStamped ps;
            ps.header.stamp = current_timestamp;
            ps.header.frame_id = "brne_odom";
            ps.pose.position.x = robot_pose.x;
            ps.pose.position.y = robot_pose.y;
            optimal_path.poses.push_back(ps);
            path_pub_->publish(optimal_path);
            return;
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
            if ((opt_cmds_lin > 0.1) && (opt_cmds_lin < 0.3))
              tw.angular.z -= 0.04;
            else if ((opt_cmds_lin >= 0.3) && (opt_cmds_lin < 0.5))
              tw.angular.z -= 0.05;
            else if (opt_cmds_lin >= 0.5)
              tw.angular.z -= 0.06;
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
            if ((opt_cmds_lin > 0.1) && (opt_cmds_lin < 0.3))
              tw.angular.z -= 0.04;
            else if ((opt_cmds_lin >= 0.3) && (opt_cmds_lin < 0.5))
              tw.angular.z -= 0.05;
            else if (opt_cmds_lin >= 0.5)
              tw.angular.z -= 0.06;
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

    if (n_agents > 1)
      trial_used_brne = true;
    if (n_peds > trial_max_people)
      trial_max_people = n_peds;

    double cmd_velocity = 0.0;
    if (!robot_cmds.twists.empty()) {
        cmd_velocity = robot_cmds.twists.front().linear.x;
    }

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
                  << n_peds << ","
                  << ped_positions.str() << ","
                  << (trial_used_brne ? "true" : "false") << "\n";
      }
    }

    cmd_buf_pub_->publish(robot_cmds);
    path_pub_->publish(optimal_path);
    pub_walls();

    const auto end = this->get_clock()->now();
    const auto diff = end - start;
    RCLCPP_DEBUG_STREAM(get_logger(), "Agents: " << n_agents << " Timer: " << diff.seconds() << " s");
  }

  void pub_walls()
  {
    if (!crosswalk_generated) {
      const double CROSSWALK_X = 5.0;
      const double STRIPE_LENGTH = 1.0;
      const double STRIPE_HEIGHT = 0.2;
      const double GAP = 0.2;
      int n_stripes = static_cast<int>((y_max - y_min) / (STRIPE_HEIGHT + GAP));
      visualization_msgs::msg::Marker stripe;
      stripe.header.frame_id = "brne_odom";
      stripe.type = visualization_msgs::msg::Marker::CUBE;
      stripe.action = visualization_msgs::msg::Marker::ADD;
      stripe.color.r = 1.0;
      stripe.color.g = 1.0;
      stripe.color.b = 0.0;
      stripe.color.a = 1.0;
      stripe.scale.x = STRIPE_LENGTH;
      stripe.scale.y = STRIPE_HEIGHT;
      stripe.scale.z = 0.05;
      for (int i = 0; i < n_stripes; i++) {
        stripe.id = i;
        stripe.pose.position.x = CROSSWALK_X;
        stripe.pose.position.y = y_min + STRIPE_HEIGHT/2 + i * (STRIPE_HEIGHT + GAP);
        stripe.pose.position.z = 0.025;
        wall_markers.markers.push_back(stripe);
      }
      crosswalk_generated = true;
    }
    const auto now = this->get_clock()->now();
    for (auto &m : wall_markers.markers) {
      m.header.stamp = now;
    }
    walls_pub_->publish(wall_markers);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathPlan>());
  rclcpp::shutdown();
  return 0;
}
