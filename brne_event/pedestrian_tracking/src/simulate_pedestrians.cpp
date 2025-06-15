#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <cmath> // for std::hypot, etc.

#include "rclcpp/rclcpp.hpp"
#include "crowd_nav_interfaces/msg/pedestrian.hpp"
#include "crowd_nav_interfaces/msg/pedestrian_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_srvs/srv/empty.hpp"

using namespace std::chrono_literals;

struct PedestrianPath {
  double start_x;
  double start_y;
  double end_x;
  double end_y;
  double current_x = 0.0;
  double current_y = 0.0;
  double current_vx = 0.0;
  double current_vy = 0.0;
};

struct PedState {
  enum { MOVING, JOINING, IN_GROUP, LEAVING } state;
  rclcpp::Time state_start;
  double original_speed;
};

class SimulatePedestrians : public rclcpp::Node
{
public:
  SimulatePedestrians()
  : Node("simulate_pedestrians"), gen(std::random_device{}())
  {
    declare_parameter("rate", 15.0);
    declare_parameter("n_static_peds", 40);
    declare_parameter("n_moving_peds", 10);
    declare_parameter("world_width", 10.0);
    declare_parameter("world_height", 5.0);
    declare_parameter("personal_space", 0.8);
    declare_parameter("max_speed", 2.5);
    // Removed respawn_time parameter

    initialize_parameters();
    setup_publishers();
    setup_services();
    initialize_pedestrians();
    start_update_timer();
  }

private:
  // Configuration parameters
  double rate_hz;
  int n_static_peds;
  int n_moving_peds;
  double world_width;
  double world_height;
  double personal_space;
  double max_speed;

  // Pedestrian data
  std::vector<crowd_nav_interfaces::msg::Pedestrian> static_peds;
  std::map<int, PedestrianPath> moving_peds;
  std::map<int, PedState> ped_states;
  std::mt19937 gen;

  // ROS components
  rclcpp::Publisher<crowd_nav_interfaces::msg::PedestrianArray>::SharedPtr ped_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr move_srv_, reset_srv_;
  rclcpp::TimerBase::SharedPtr update_timer_;
  bool simulation_active = false;

  void initialize_parameters() {
    rate_hz = get_parameter("rate").as_double();
    n_static_peds = get_parameter("n_static_peds").as_int();
    n_moving_peds = get_parameter("n_moving_peds").as_int();
    world_width = get_parameter("world_width").as_double();
    world_height = get_parameter("world_height").as_double();
    personal_space = get_parameter("personal_space").as_double();
    max_speed = get_parameter("max_speed").as_double();
    // Removed respawn_time initialization
  }

  void setup_publishers() {
    ped_pub_ = create_publisher<crowd_nav_interfaces::msg::PedestrianArray>("pedestrians", 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("tree_markers", 10);
  }

  void setup_services() {
    move_srv_ = create_service<std_srvs::srv::Empty>(
      "move_ped",
      [this](
        const std::shared_ptr<rmw_request_id_t> /*header*/,
        const std::shared_ptr<std_srvs::srv::Empty::Request> /*request*/,
        std::shared_ptr<std_srvs::srv::Empty::Response> /*response*/
      ) {
        RCLCPP_INFO(get_logger(), "Start Moving pedestrians");
        simulation_active = true;
      }
    );

    reset_srv_ = create_service<std_srvs::srv::Empty>(
      "reset_ped",
      [this](
        const std::shared_ptr<rmw_request_id_t> /*header*/,
        const std::shared_ptr<std_srvs::srv::Empty::Request> /*request*/,
        std::shared_ptr<std_srvs::srv::Empty::Response> /*response*/
      ) {
        RCLCPP_INFO(get_logger(), "Reset Pedestrians");
        initialize_pedestrians();
      }
    );
  }

  void initialize_pedestrians() {
    static_peds.clear();
    initialize_static_formation();

    moving_peds.clear();
    std::uniform_real_distribution<double> x_dist(0, world_width);
    std::uniform_real_distribution<double> y_dist(-world_height, world_height);

    for (int i = 0; i < n_moving_peds; i++) {
      PedestrianPath path;
      path.start_x = x_dist(gen);
      path.start_y = y_dist(gen);
      path.end_x = x_dist(gen);
      path.end_y = y_dist(gen);
      path.current_x = path.start_x;
      path.current_y = path.start_y;
      moving_peds[i] = path;

      PedState state;
      state.state = PedState::MOVING;
      state.state_start = this->get_clock()->now();
      state.original_speed = 0.0;
      ped_states[i] = state;
    }
  }

  void initialize_static_formation() {
    const double formation_start_x = 3.0;
    const double formation_end_x   = 7.0;
    const double formation_width   = formation_end_x - formation_start_x;
    const double formation_height  = 4.0; // y ranges from -2 to 2

    const double grid_spacing = personal_space * 0.8; // Denser than personal space
    int columns = static_cast<int>(formation_width / grid_spacing);
    int rows    = static_cast<int>(formation_height / grid_spacing);

    std::uniform_real_distribution<double> pert(-0.1, 0.1);

    for (int col = 0; col < columns; ++col) {
      for (int row = 0; row < rows; ++row) {
        double x = formation_start_x + col * grid_spacing + pert(gen);
        double y = -formation_height / 2.0 + row * grid_spacing + pert(gen);
        // Leave a gap near x=5.0 for realism
        if (std::hypot(x - 5.0, y) > 1.0) {
          add_static_pedestrian(x, y);
        }
      }
    }

    // Optional: Add moving pedestrians circling the formation
    const int perimeter_peds = n_moving_peds / 2;
    std::uniform_real_distribution<double> angle_dist(0, 2 * M_PI);
    for (int i = 0; i < perimeter_peds; ++i) {
      double angle = angle_dist(gen);
      double radius = 3.0 + pert(gen);

      PedestrianPath path;
      path.start_x = 5.0 + radius * cos(angle);
      path.start_y = radius * sin(angle);
      path.end_x = 5.0 + radius * cos(angle + M_PI / 2.0);
      path.end_y = radius * sin(angle + M_PI / 2.0);
      path.current_x = path.start_x;
      path.current_y = path.start_y;
      moving_peds[i] = path;

      // Initial velocity tangential to the circle
      double tangent_speed = 1.2;
      path.current_vx = -tangent_speed * sin(angle);
      path.current_vy = tangent_speed * cos(angle);
    }
  }

  void add_static_pedestrian(double x, double y) {
    crowd_nav_interfaces::msg::Pedestrian ped;
    ped.pose.position.x = x;
    ped.pose.position.y = y;
    ped.velocity.linear.x = 0.0;
    ped.velocity.linear.y = 0.0;
    static_peds.push_back(ped);
  }

  void start_update_timer() {
    auto interval = std::chrono::duration<double>(1.0 / rate_hz);
    update_timer_ = create_wall_timer(interval, [this]() {
      if (simulation_active) {
        update_moving_pedestrians();
        publish_pedestrians();
      }
      // Always publish the tree marker for visualization.
      publish_tree_marker();
    });
  }

  void update_moving_pedestrians() {
    for (auto& [id, path] : moving_peds) {
      move_to_goal(path, max_speed);
      updatePedestrianRepulsion(path);

      // Apply velocity limits
      double current_speed = std::hypot(path.current_vx, path.current_vy);
      if (current_speed > max_speed) {
        path.current_vx *= (max_speed / current_speed);
        path.current_vy *= (max_speed / current_speed);
      }

      // Update position
      double dt = 1.0 / rate_hz;
      path.current_x += path.current_vx * dt;
      path.current_y += path.current_vy * dt;

      // Keep within bounds
      path.current_x = std::clamp(path.current_x, 0.0, world_width);
      path.current_y = std::clamp(path.current_y, -world_height, world_height);

      // Assign new goal when current is reached
      if (reached_goal(path)) {
        assign_new_goal(id);
      }
    }
  }

  void assign_new_goal(int id) {
    auto& path = moving_peds[id];
    std::uniform_real_distribution<double> x_dist(0, world_width);
    std::uniform_real_distribution<double> y_dist(-world_height, world_height);
    
    // Keep trying until we get a goal that's far enough away
    double new_end_x, new_end_y, dist;
    do {
      new_end_x = x_dist(gen);
      new_end_y = y_dist(gen);
      double dx = new_end_x - path.current_x;
      double dy = new_end_y - path.current_y;
      dist = std::hypot(dx, dy);
    } while (dist < 2.0); // Ensure new goal is at least 2m away

    path.end_x = new_end_x;
    path.end_y = new_end_y;
  }

  void publish_pedestrians() {
    crowd_nav_interfaces::msg::PedestrianArray msg;
    msg.header.stamp = get_clock()->now();

    // Add static pedestrians
    for (size_t i = 0; i < static_peds.size(); i++) {
      auto ped = static_peds[i];
      ped.id = i + 1;
      msg.pedestrians.push_back(ped);
    }

    // Add all moving pedestrians
    size_t move_id_offset = static_peds.size();
    for (const auto& [id, path] : moving_peds) {
      crowd_nav_interfaces::msg::Pedestrian ped;
      ped.id = move_id_offset + id + 1;
      ped.pose.position.x = path.current_x;
      ped.pose.position.y = path.current_y;
      ped.velocity.linear.x = path.current_vx * 0.3;
      ped.velocity.linear.y = path.current_vy * 0.3;
      msg.pedestrians.push_back(ped);
    }

    ped_pub_->publish(msg);
  }

  // Publishes a MarkerArray representing the tree (trunk + foliage)
  void publish_tree_marker() {
    visualization_msgs::msg::MarkerArray marker_array;

    // --- Tree Trunk ---
    visualization_msgs::msg::Marker trunk;
    trunk.header.frame_id = "brne_odom"; // Changed from "map"
    trunk.header.stamp = this->get_clock()->now();
    trunk.ns = "tree";
    trunk.id = 0;
    trunk.type = visualization_msgs::msg::Marker::CYLINDER;
    trunk.action = visualization_msgs::msg::Marker::ADD;
    trunk.pose.position.x = 4.85646;
    trunk.pose.position.y = 0.0368093;
    trunk.pose.position.z = 0.5; // half the trunk height
    trunk.pose.orientation.w = 1.0;
    trunk.scale.x = 0.2;
    trunk.scale.y = 0.2;
    trunk.scale.z = 1.0;
    trunk.color.a = 1.0;
    trunk.color.r = 0.55;
    trunk.color.g = 0.27;
    trunk.color.b = 0.07;

    // --- Tree Foliage ---
    visualization_msgs::msg::Marker foliage;
    foliage.header.frame_id = "brne_odom"; // Changed from "map"
    foliage.header.stamp = this->get_clock()->now();
    foliage.ns = "tree";
    foliage.id = 1;
    foliage.type = visualization_msgs::msg::Marker::SPHERE;
    foliage.action = visualization_msgs::msg::Marker::ADD;
    foliage.pose.position.x = 4.85646;
    foliage.pose.position.y = 0.0368093;
    foliage.pose.position.z = 1.5; // above the trunk
    foliage.pose.orientation.w = 1.0;
    foliage.scale.x = 1.0;
    foliage.scale.y = 1.0;
    foliage.scale.z = 1.0;
    foliage.color.a = 1.0;
    foliage.color.r = 0.0;
    foliage.color.g = 1.0;
    foliage.color.b = 0.0;

    marker_array.markers.push_back(trunk);
    marker_array.markers.push_back(foliage);
    marker_pub_->publish(marker_array);
  }

  bool reached_goal(const PedestrianPath& path) {
    return std::hypot(path.current_x - path.end_x, path.current_y - path.end_y) < 0.5;
  }

  void move_to_goal(PedestrianPath& path, double speed) {
    double dx = path.end_x - path.current_x;
    double dy = path.end_y - path.current_y;
    double dist = std::hypot(dx, dy);

    if (dist > 0.1) {
      path.current_vx = (dx / dist) * speed;
      path.current_vy = (dy / dist) * speed;
    } else {
      path.current_vx = 0.0;
      path.current_vy = 0.0;
    }
  }

  void updatePedestrianRepulsion(PedestrianPath& path) {
    const double collision_threshold = 0.5; // Fixed collision distance

    // Repel from other moving peds
    for (auto& [other_id, other_path] : moving_peds) {
      if (&other_path == &path) continue;
      double dx = path.current_x - other_path.current_x;
      double dy = path.current_y - other_path.current_y;
      double dist = std::hypot(dx, dy);
      if (dist < collision_threshold && dist > 0.01) {
        double force = 2.0 * (1.0 - dist / collision_threshold);
        path.current_vx += force * (dx / dist);
        path.current_vy += force * (dy / dist);
      }
    }

    // Repel from static peds
    for (const auto& ped : static_peds) {
      double dx = path.current_x - ped.pose.position.x;
      double dy = path.current_y - ped.pose.position.y;
      double dist = std::hypot(dx, dy);
      if (dist < collision_threshold && dist > 0.01) {
        double force = 2.0 * (1.0 - dist / collision_threshold);
        path.current_vx += force * (dx / dist);
        path.current_vy += force * (dy / dist);
      }
    }

    // Tree avoidance remains the same
    const double tree_avoid_radius = 1.2;
    const double dx_tree = path.current_x - 4.85646;
    const double dy_tree = path.current_y - 0.0368093;
    const double dist_to_tree = std::hypot(dx_tree, dy_tree);
    if (dist_to_tree < tree_avoid_radius && dist_to_tree > 0.01) {
      double force = 3.0 * (1.0 - dist_to_tree / tree_avoid_radius);
      path.current_vx += force * (dx_tree / dist_to_tree);
      path.current_vy += force * (dy_tree / dist_to_tree);
    }
  }

  void updateObstacleForces(PedestrianPath& path) {
    // Pedestrian-pedestrian avoidance only
    for (auto& [other_id, other_path] : moving_peds) {
      if (&other_path == &path) continue;
      double dx = path.current_x - other_path.current_x;
      double dy = path.current_y - other_path.current_y;
      double dist = std::hypot(dx, dy);
      if (dist < personal_space && dist > 0.01) {
        double force = 2.0 * (1.0 - dist / personal_space);
        path.current_vx += force * (dx / dist);
        path.current_vy += force * (dy / dist);
      }
    }
  }
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimulatePedestrians>());
  rclcpp::shutdown();
  return 0;
}