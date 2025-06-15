#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "crowd_nav_interfaces/msg/pedestrian.hpp"
#include "crowd_nav_interfaces/msg/pedestrian_array.hpp"
#include "std_srvs/srv/empty.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"

using namespace std::chrono_literals;

class SimulatePedestrians : public rclcpp::Node
{
public:
  SimulatePedestrians()
  : Node("simulate_pedestrians")
  {
    // --- Basic parameters and group configuration:
    // Total n_peds = 14:
    //   0–1: WALL, 2–4: MEETING, 5–8: STATIC, 9–11: CYCLIC, 12–13: TOPDOWN
    declare_parameter("rate", 15.0);
    declare_parameter("n_peds", 14);
    declare_parameter("moving", true);
    declare_parameter("ped_vel", -1.0);  // Base speed (absolute)
    declare_parameter("meeting_wait_time", 10.0); // Seconds for MEETING group

    // --- For WALL and MEETING groups (indices 0–4)
    declare_parameter("ped_x", std::vector<double>{2.0, 3.0, 4.0, 5.0, 6.0});
    declare_parameter("ped_start_y", std::vector<double>{-1.5, 1.5, -1.5, 1.5, -1.5});
    declare_parameter("ped_end_y", std::vector<double>{1.5, -1.5, 1.5, -1.5, 1.5});

    // --- Collision/obstacle avoidance parameters:
    declare_parameter("desired_distance", 1.2);
    declare_parameter("damping_factor", 0.8);
    declare_parameter("obstacle_gain", 4.0);
    declare_parameter("obstacle_range", 2.5);

    // --- Social force gain (controls how strongly pedestrians repulse each other).
    declare_parameter("social_force_gain", 0.5);

    // --- Corridor boundaries:
    declare_parameter("x_min", -2.0);
    declare_parameter("x_max", 7.0);
    declare_parameter("y_min", -1.5);
    declare_parameter("y_max", 1.5);

    // --- STATIC group clustering parameters:
    declare_parameter("static_cluster_x", 1.0);
    declare_parameter("static_cluster_y", 0.0);

    // --- CYCLIC group parameters:
    declare_parameter("cyclic_wait_time", 2.0);
    declare_parameter("cyclic_speed", 1.0);
    declare_parameter("cyclic_stop_offset", 0.5);

    // --- TOPDOWN group parameters:
    declare_parameter("topdown_wait_time", 3.0);
    declare_parameter("coffee_stop_distance", 0.4);

    // --- Room parameters (new):
    // room_size: length of each side of the square room.
    // room_thickness: thickness of each wall.
    // room_gap: gap between the corridor wall and the room.
    declare_parameter("room_size", 4.0);
    declare_parameter("room_thickness", 0.1);
    declare_parameter("room_gap", 0.01);

    // --- Retrieve parameters:
    rate_hz = get_parameter("rate").as_double();
    dt = 1.0 / rate_hz;
    n_peds = get_parameter("n_peds").as_int();
    moving_global = get_parameter("moving").as_bool();
    double ped_vel_param = get_parameter("ped_vel").as_double();
    speed_val = std::fabs(ped_vel_param);
    meeting_wait_time = get_parameter("meeting_wait_time").as_double();
    desired_distance = get_parameter("desired_distance").as_double();
    damping_factor = get_parameter("damping_factor").as_double();
    social_force_gain = get_parameter("social_force_gain").as_double();

    corridor_y_min = get_parameter("y_min").as_double();
    corridor_y_max = get_parameter("y_max").as_double();
    corridor_x_min = get_parameter("x_min").as_double();
    corridor_x_max = get_parameter("x_max").as_double();

    static_cluster_x = get_parameter("static_cluster_x").as_double();
    static_cluster_y = get_parameter("static_cluster_y").as_double();

    cyclic_wait_time = get_parameter("cyclic_wait_time").as_double();
    cyclic_speed = get_parameter("cyclic_speed").as_double();
    cyclic_stop_offset = get_parameter("cyclic_stop_offset").as_double();

    topdown_wait_time = get_parameter("topdown_wait_time").as_double();
    coffee_stop_distance = get_parameter("coffee_stop_distance").as_double();

    room_size = get_parameter("room_size").as_double();
    room_thickness = get_parameter("room_thickness").as_double();
    room_gap = get_parameter("room_gap").as_double();

    ped_x_base = get_parameter("ped_x").as_double_array();
    ped_start_y_base = get_parameter("ped_start_y").as_double_array();
    ped_end_y_base = get_parameter("ped_end_y").as_double_array();

    // --- Publishers and timer:
    pedestrian_pub_ = create_publisher<crowd_nav_interfaces::msg::PedestrianArray>("pedestrians", 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("coffee_machine", 10);
    rooms_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("rooms", 10);

    std::chrono::milliseconds period((int)(1000.0 / rate_hz));
    timer_ = create_wall_timer(period, std::bind(&SimulatePedestrians::timer_callback, this));

    publishCoffeeMachine();
    initPedestrians();

    move_srv_ = create_service<std_srvs::srv::Empty>(
      "move_ped",
      std::bind(&SimulatePedestrians::move_cb, this, std::placeholders::_1, std::placeholders::_2));
    reset_srv_ = create_service<std_srvs::srv::Empty>(
      "reset_ped",
      std::bind(&SimulatePedestrians::reset_cb, this, std::placeholders::_1, std::placeholders::_2));
  }

private:
  double rate_hz, dt;
  int n_peds;
  bool moving_global;
  double speed_val;
  double meeting_wait_time;
  double desired_distance;
  double damping_factor;
  double social_force_gain;
  double corridor_y_min, corridor_y_max, corridor_x_min, corridor_x_max;
  double static_cluster_x, static_cluster_y;
  double cyclic_wait_time, cyclic_speed, cyclic_stop_offset;
  double topdown_wait_time, coffee_stop_distance;
  double room_size, room_thickness, room_gap;

  std::vector<double> ped_x_base, ped_start_y_base, ped_end_y_base;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<crowd_nav_interfaces::msg::PedestrianArray>::SharedPtr pedestrian_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr rooms_pub_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr move_srv_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv_;

  // --- Obstacle (coffee machine) parameters:
  const double obst_x = 5.00189;
  const double obst_y = 1.46894;
  const double obst_half_width = 0.25;
  const double obst_half_height = 0.25;
  const double obst_margin = 0.1;

  // --- Meeting point for MEETING group:
  const double meeting_point_x = 5.0;
  const double meeting_point_y = 0.0;

  // --- Pedestrian types and phases:
  enum PedType { WALL, MEETING, STATIC, CYCLIC, TOPDOWN };
  enum PedPhase { TO_MID, MOVING, TO_MEETING, WAITING };

  struct PedState {
    int id;
    PedType type;
    PedPhase phase;
    double current_x;
    double current_y;
    double start_x;
    double start_y;
    double final_y;   // For WALL and MEETING groups
    double dest_x;
    double dest_y;
    double speed;
    int wait_counter;
    bool active;
    double cyclic_direction;  // Used by CYCLIC and TOPDOWN groups
  };

  std::vector<PedState> ped_states;

  // --- Initialize pedestrian states.
  // For STATIC group, arrange 4 pedestrians in a 2×2 square with 0.5 m spacing.
  void initPedestrians() {
    ped_states.clear();
    ped_states.resize(n_peds);
    for (int i = 0; i < n_peds; ++i) {
      PedState &p = ped_states[i];
      p.id = i + 1;
      p.wait_counter = 0;
      p.active = true;
      if (i < 2) { // WALL group (unchanged)
        p.type = WALL;
        p.start_x = ped_x_base[i];
        p.start_y = ped_start_y_base[i];
        p.current_x = p.start_x;
        p.current_y = p.start_y;
        p.final_y = ped_end_y_base[i];
        p.phase = TO_MID;
        p.dest_x = p.start_x;
        p.dest_y = (p.start_y + p.final_y) / 2.0;
        p.speed = speed_val;
      }
      else if (i < 5) { // MEETING group
        p.type = MEETING;
        // --- Modification: force meeting group x positions to be in [5,7]
        p.start_x = 5.0 + (i - 2) * 1.0;  // i=2 -> 5.0, i=3 -> 6.0, i=4 -> 7.0
        p.start_y = ped_start_y_base[i];  // keep original y (or adjust if desired)
        p.current_x = p.start_x;
        p.current_y = p.start_y;
        p.final_y = ped_end_y_base[i];
        p.phase = TO_MEETING;
        p.dest_x = meeting_point_x;
        p.dest_y = meeting_point_y;
        p.speed = speed_val;
      }
      else if (i < 9) { // STATIC group in a 2×2 square with 0.5 m spacing.
        p.type = STATIC;
        int j = i - 5;
        int row = j / 2;
        int col = j % 2;
        p.start_x = static_cluster_x + (col - 0.5) * 0.5;
        p.start_y = static_cluster_y + (row - 0.5) * 0.5+ 0.7;
        p.current_x = p.start_x;
        p.current_y = p.start_y;
        p.dest_x = p.start_x;
        p.dest_y = p.start_y;
        p.speed = 0.0;
      }
      else if (i < 12) { // CYCLIC group
        p.type = CYCLIC;
        p.phase = MOVING;
        if (i == 9) {
          // Vertical movement: start at x = 4.0.
          p.start_x = 4.0;
          p.start_y = corridor_y_max;
          p.current_x = p.start_x;
          p.current_y = p.start_y;
          p.dest_x = p.start_x;
          p.dest_y = corridor_y_min;
          p.cyclic_direction = -1;
          p.speed = cyclic_speed * 0.5;
        } else if (i == 10) {
          // Horizontal movement: start from the left boundary.
          p.start_x = corridor_x_min;
          p.start_y = 0.0;
          p.current_x = p.start_x;
          p.current_y = p.start_y;
          p.dest_x = corridor_x_max;
          p.dest_y = p.start_y;
          p.cyclic_direction = 1;
          p.speed = cyclic_speed * 0.7;
        } else if (i == 11) {
          // Vertical movement: start at x = 6.0.
          p.start_x = 6.0;
          p.start_y = corridor_y_max;
          p.current_x = p.start_x;
          p.current_y = p.start_y;
          p.dest_x = p.start_x;
          p.dest_y = corridor_y_min;
          p.cyclic_direction = -1;
          p.speed = cyclic_speed * 0.9;
        }
      }
      else { // TOPDOWN group (indices 12–13)
        p.type = TOPDOWN;
        p.phase = MOVING;
        p.start_x = (i == 12 ? 3.5 : 4.5);
        // --- Modification: adjust y so that the pedestrian coming from the back starts a bit lower.
        p.start_y = corridor_y_max - 0.5;
        p.current_x = p.start_x;
        p.current_y = p.start_y;
        p.dest_x = p.start_x;
        p.dest_y = corridor_y_min;
        p.cyclic_direction = -1;
        p.speed = cyclic_speed * 0.5;
      }
    }
  }

  // --- Move and reset service callbacks.
  void move_cb(
    std::shared_ptr<std_srvs::srv::Empty::Request>,
    std::shared_ptr<std_srvs::srv::Empty::Response>)
  {
    RCLCPP_INFO_STREAM(get_logger(), "Start Moving pedestrians");
    moving_global = true;
  }

  void reset_cb(
    std::shared_ptr<std_srvs::srv::Empty::Request>,
    std::shared_ptr<std_srvs::srv::Empty::Response>)
  {
    RCLCPP_INFO_STREAM(get_logger(), "Reset Pedestrians");
    initPedestrians();
    moving_global = false;
  }

  // --- Publish coffee machine marker.
  void publishCoffeeMachine() {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "brne_odom";
    marker.header.stamp = this->now();
    marker.ns = "obstacles";
    marker.id = 100;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = obst_x;
    marker.pose.position.y = obst_y;
    marker.pose.position.z = 0.5;
    marker.scale.x = 0.5;
    marker.scale.y = 0.5;
    marker.scale.z = 1.0;
    marker.color.r = 0.6;
    marker.color.g = 0.3;
    marker.color.b = 0.0;
    marker.color.a = 1.0;
    marker_pub_->publish(marker);
  }

  // --- Euclidean distance.
  double distance(double x1, double y1, double x2, double y2)
  {
    return std::sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
  }

  // --- Update position with obstacle and social avoidance.
  void updatePosition(PedState &p)
  {
    if (p.type == STATIC)
      return;
    double diff_x = p.dest_x - p.current_x;
    double diff_y = p.dest_y - p.current_y;
    double dist_val = std::sqrt(diff_x * diff_x + diff_y * diff_y);
    if (dist_val < 1e-3)
      return;
    double dx = (diff_x / dist_val) * p.speed * dt;
    double dy = (diff_y / dist_val) * p.speed * dt;
    double new_x = p.current_x + dx;
    double new_y = p.current_y + dy;
    // Personal space avoidance:
    double effective_threshold = 0.3;
    for (const auto &other : ped_states) {
      if (&other == &p) continue;
      if (!other.active || other.type == STATIC) continue;
      double sep = distance(new_x, new_y, other.current_x, other.current_y);
      if (sep < effective_threshold) {
        dx *= damping_factor * social_force_gain;
        dy *= damping_factor * social_force_gain;
        new_x = p.current_x + dx;
        new_y = p.current_y + dy;
      }
    }
    // Obstacle avoidance:
    double dx_obst = new_x - obst_x;
    double dy_obst = new_y - obst_y;
    double dist_obst = std::hypot(dx_obst, dy_obst);
    double obst_range = get_parameter("obstacle_range").as_double();
    if (dist_obst < obst_range) {
      double repulsive = 0.5 * get_parameter("obstacle_gain").as_double() * (obst_range - dist_obst);
      new_x += (dx_obst / dist_obst) * repulsive * dt;
      new_y += (dy_obst / dist_obst) * repulsive * dt;
    }
    // Optional fixed offset if extremely close:
    if ((std::fabs(new_x - obst_x) < (obst_half_width + obst_margin)) &&
        (std::fabs(new_y - obst_y) < (obst_half_height + obst_margin)))
    {
      if (new_x >= obst_x)
        new_x += 0.05;
      else
        new_x -= 0.05;
    }
    p.current_x = new_x;
    p.current_y = new_y;
  }

  // --- Publish room markers.
  // Two rooms are added: one attached to the lower wall (y_min) and one to the upper wall (y_max).
  // Each room is a square with four walls (room_thickness) and side length room_size.
  // They are positioned outside the corridor (with a gap room_gap from the wall).
  void pub_rooms() {
    visualization_msgs::msg::MarkerArray rooms;
    double center_x = (corridor_x_min + corridor_x_max) / 2.0;

    // For Room 1 (lower room): attach its north wall flush with the corridor.
    double room1_north = corridor_y_min;   // no gap now
    double room1_south = room1_north - room_size;
    double room1_west  = center_x - room_size / 2.0;
    double room1_east  = center_x + room_size / 2.0;
    int marker_id = 0;
    // Skip publishing the north wall (since the corridor wall is there) and publish the other three walls:
    for (int i = 1; i < 4; i++) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "brne_odom";
      marker.header.stamp = this->now();
      marker.ns = "rooms";
      marker.id = marker_id++;
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.scale.z = 1.0;
      marker.color.r = 0.0;
      marker.color.g = 0.0;
      marker.color.b = 1.0;
      marker.color.a = 1.0;
      if (i == 1) { // south wall
        marker.scale.x = room_size;
        marker.scale.y = room_thickness;
        marker.pose.position.x = center_x;
        marker.pose.position.y = room1_south + room_thickness / 2.0;
      } else if (i == 2) { // west wall
        marker.scale.x = room_thickness;
        marker.scale.y = room_size;
        marker.pose.position.x = room1_west + room_thickness / 2.0;
        marker.pose.position.y = (room1_north + room1_south) / 2.0;
      } else if (i == 3) { // east wall
        marker.scale.x = room_thickness;
        marker.scale.y = room_size;
        marker.pose.position.x = room1_east - room_thickness / 2.0;
        marker.pose.position.y = (room1_north + room1_south) / 2.0;
      }
      marker.pose.position.z = 0.5;
      rooms.markers.push_back(marker);
    }

    // For Room 2 (upper room): attach its south wall flush with the corridor.
    double room2_south = corridor_y_max;   // no gap now
    double room2_north = room2_south + room_size;
    double room2_west  = center_x - room_size / 2.0;
    double room2_east  = center_x + room_size / 2.0;
    for (int i = 1; i < 4; i++) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "brne_odom";
      marker.header.stamp = this->now();
      marker.ns = "rooms";
      marker.id = marker_id++;
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.scale.z = 1.0;
      marker.color.r = 0.0;
      marker.color.g = 0.0;
      marker.color.b = 1.0;
      marker.color.a = 1.0;
      if (i == 1) { // north wall
        marker.scale.x = room_size;
        marker.scale.y = room_thickness;
        marker.pose.position.x = center_x;
        marker.pose.position.y = room2_north - room_thickness / 2.0;
      } else if (i == 2) { // west wall
        marker.scale.x = room_thickness;
        marker.scale.y = room_size;
        marker.pose.position.x = room2_west + room_thickness / 2.0;
        marker.pose.position.y = (room2_north + room2_south) / 2.0;
      } else if (i == 3) { // east wall
        marker.scale.x = room_thickness;
        marker.scale.y = room_size;
        marker.pose.position.x = room2_east - room_thickness / 2.0;
        marker.pose.position.y = (room2_north + room2_south) / 2.0;
      }
      marker.pose.position.z = 0.5;
      rooms.markers.push_back(marker);
    }

    rooms_pub_->publish(rooms);
  }

  // --- Update pedestrian positions.
  void timer_callback() {
    auto current_time = this->get_clock()->now();
    crowd_nav_interfaces::msg::PedestrianArray peds_msg;
    peds_msg.header.stamp = current_time;
    peds_msg.header.frame_id = "brne_odom";

    // Process each pedestrian as before...
    for (auto &p : ped_states) {
      if (!p.active)
        continue;
      if (moving_global && p.type != STATIC) {
        if (p.type == WALL) {
          if (p.phase == TO_MID) {
            updatePosition(p);
            if (distance(p.current_x, p.current_y, p.start_x, (p.start_y + p.final_y)/2.0) < 0.1) {
              p.phase = MOVING;
              p.dest_y = p.final_y;
            }
          } else {
            updatePosition(p);
            if ((p.start_y < p.final_y && p.current_y >= p.final_y) ||
                (p.start_y > p.final_y && p.current_y <= p.final_y))
            {
              p.active = false;
              continue;
            }
          }
        }
        else if (p.type == MEETING) {
          if (p.phase == TO_MEETING) {
            updatePosition(p);
            if (distance(p.current_x, p.current_y, meeting_point_x, meeting_point_y) < 0.1) {
              p.phase = WAITING;
              p.wait_counter = static_cast<int>(meeting_wait_time * rate_hz);
            }
          } else if (p.phase == WAITING) {
            if (p.wait_counter > 0)
              p.wait_counter--;
            else {
              p.phase = MOVING;
              p.dest_x = p.start_x;
              p.dest_y = p.final_y;
            }
          } else if (p.phase == MOVING) {
            updatePosition(p);
            if (distance(p.current_x, p.current_y, p.dest_x, p.dest_y) < 0.1) {
              p.active = false;
              continue;
            }
          }
        }
        else if (p.type == CYCLIC) {
          if (p.start_y != 0.0) { // Vertical movement.
            if (p.phase == MOVING) {
              updatePosition(p);
              if ((std::fabs(p.current_y - corridor_y_min) < 0.1) ||
                  (std::fabs(p.current_y - p.start_y) < 0.1))
              {
                p.phase = WAITING;
                p.wait_counter = static_cast<int>(cyclic_wait_time * rate_hz);
              }
            } else if (p.phase == WAITING) {
              p.current_x = p.start_x;
              if (p.wait_counter > 0)
                p.wait_counter--;
              else {
                p.phase = MOVING;
                if (std::fabs(p.current_y - corridor_y_min) < 0.1)
                  p.dest_y = p.start_y;
                else if (std::fabs(p.current_y - p.start_y) < 0.1)
                  p.dest_y = corridor_y_min;
              }
            }
          } else { // Horizontal movement.
            if (p.phase == MOVING) {
              updatePosition(p);
              if ((std::fabs(p.current_x - corridor_x_min) < 0.1) ||
                  (std::fabs(p.current_x - corridor_x_max) < 0.1))
              {
                p.phase = WAITING;
                p.wait_counter = static_cast<int>(cyclic_wait_time * rate_hz);
              }
            } else if (p.phase == WAITING) {
              p.current_y = p.start_y;
              if (p.wait_counter > 0)
                p.wait_counter--;
              else {
                p.phase = MOVING;
                if (std::fabs(p.current_x - corridor_x_min) < 0.1)
                  p.dest_x = corridor_x_max;
                else if (std::fabs(p.current_x - corridor_x_max) < 0.1)
                  p.dest_x = corridor_x_min;
              }
            }
          }
        }
        else if (p.type == TOPDOWN) {
          if (p.phase == MOVING) {
            if (distance(p.current_x, p.current_y, obst_x, obst_y) < coffee_stop_distance) {
              p.phase = WAITING;
              p.wait_counter = static_cast<int>(topdown_wait_time * rate_hz);
            } else {
              updatePosition(p);
              if ((std::fabs(p.current_y - corridor_y_min) < 0.1) ||
                  (std::fabs(p.current_y - p.start_y) < 0.1))
              {
                p.phase = WAITING;
                p.wait_counter = static_cast<int>(topdown_wait_time * rate_hz);
              }
            }
          } else if (p.phase == WAITING) {
            if (p.wait_counter > 0)
              p.wait_counter--;
            else {
              p.phase = MOVING;
              if (std::fabs(p.current_y - corridor_y_min) < 0.1)
                p.dest_y = p.start_y;
              else if (std::fabs(p.current_y - p.start_y) < 0.1)
                p.dest_y = corridor_y_min;
            }
          }
        }
      }
      // Build pedestrian message.
      crowd_nav_interfaces::msg::Pedestrian ped_msg;
      ped_msg.header.stamp = current_time;
      ped_msg.id = p.id;
      ped_msg.pose.position.x = p.current_x;
      ped_msg.pose.position.y = p.current_y;
      
      tf2::Quaternion q;
      double yaw = 0.0;
      if (p.type != STATIC) {
        double vx = p.dest_x - p.current_x;
        double vy = p.dest_y - p.current_y;
        yaw = std::atan2(vy, vx);
      }
      q.setRPY(0, 0, yaw);
      ped_msg.pose.orientation.x = q.x();
      ped_msg.pose.orientation.y = q.y();
      ped_msg.pose.orientation.z = q.z();
      ped_msg.pose.orientation.w = q.w();
      if (p.type != STATIC) {
        if ((p.type == MEETING || p.type == CYCLIC || p.type == TOPDOWN) && p.phase == WAITING) {
          ped_msg.velocity.linear.x = 0.0;
          ped_msg.velocity.linear.y = 0.0;
        } else {
          ped_msg.velocity.linear.x = std::cos(yaw) * p.speed;
          ped_msg.velocity.linear.y = std::sin(yaw) * p.speed;
        }
      }
      peds_msg.pedestrians.push_back(ped_msg);
    }
    pedestrian_pub_->publish(peds_msg);
    
    // Publish the room markers.
    pub_rooms();
  }
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimulatePedestrians>());
  rclcpp::shutdown();
  return 0;
}