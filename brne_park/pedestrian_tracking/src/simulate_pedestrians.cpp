#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "image_geometry/stereo_camera_model.h"
#include "sensor_msgs/msg/camera_info.hpp"
#include "crowd_nav_interfaces/msg/pedestrian.hpp"
#include "crowd_nav_interfaces/msg/pedestrian_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"
#include "std_srvs/srv/empty.hpp"
#include "crowd_nav_interfaces/msg/line_obstacle.hpp"
#include "crowd_nav_interfaces/msg/line_obstacles.hpp"

#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <random>

using namespace std::chrono_literals;

// Environment boundaries (adjust according to your setup)
constexpr double MIN_X = 0.0;
constexpr double MAX_X = 10.0;
constexpr double MIN_Y = -5.0;
constexpr double MAX_Y = 5.0;

static std::map<int, std::vector<std::pair<double,double>>> pedestrian_waypoints = {
  {0, {{2.0, 0.0}, {1.0, 0.5}, {0.0, 0.0}}},
  {1, {{2.0, -0.5}, {1.0, -1.0}, {0.0, -0.5}}}
};
static std::vector<crowd_nav_interfaces::msg::LineObstacle> obstacles_;

struct PedestrianGroup {
  std::vector<int> member_ids;
  double cohesion_force = 0.5;
  double repulsion_force = 1.0;
};
static std::vector<PedestrianGroup> pedestrian_groups;

class SimulatePedestrians : public rclcpp::Node
{
public:
  SimulatePedestrians()
  : Node("simulate_pedestrians"),
    go(false),
    gen(std::random_device{}()),
    start_x_dist(),
    end_x_dist(),
    y_dist(),
    speed_dist(),
    angle_dist(0.0, 2*M_PI)
  {
    // Existing parameters
    declare_parameter("rate", 15.0);
    declare_parameter("n_peds", 20);  // Ensure this is enough to include circle pedestrians
    declare_parameter("moving", true);
    declare_parameter("ped_start_x", std::vector<double>{5.0, 5.5, 6.0, 6.5});
    declare_parameter("ped_end_x", std::vector<double>{0.0, -0.5, -1.0, -1.5});
    declare_parameter("ped_y", std::vector<double>{0.05, -0.05, 0.10, -0.10});
    declare_parameter("ped_vel", std::vector<double>{-1.0, -0.8, -0.9, -0.7});

    declare_parameter("static_peds/x", std::vector<double>{1.0, 1.5, 2.0, 2.5});
    declare_parameter("static_peds/y", std::vector<double>{0.0, -0.5, 0.5, 1.0});

    declare_parameter("start_x_range", std::vector<double>{4.5, 6.5});
    declare_parameter("end_x_range", std::vector<double>{-2.0, 2.0});
    declare_parameter("y_range", std::vector<double>{-1.0, 1.0});
    declare_parameter("speed_range", std::vector<double>{0.5, 1.5});

    declare_parameter("cohesion_gain", 0.5);
    declare_parameter("obstacle_gain", 5.0);
    declare_parameter("obstacle_range", 2.0);

    declare_parameter("personal_space_radius", 1.0);  // Minimum comfortable distance
    declare_parameter("repulsion_gain", 3.0);         // Strength of avoidance

    // ----- Trees-related parameter and publisher added -----
    declare_parameter("n_trees", 5);
    n_trees_ = get_parameter("n_trees").as_int();
    tree_markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("trees", 10);
    trees_generated_ = false;
    // --------------------------------------------------------

    // Circle pedestrians parameters
    declare_parameter("circle_center_x", 5.0);
    declare_parameter("circle_center_y", 0.0);
    declare_parameter("circle_radius", 2.0);
    declare_parameter("circle_num_peds", 8);

    double circle_center_x = get_parameter("circle_center_x").as_double();
    double circle_center_y = get_parameter("circle_center_y").as_double();
    double circle_radius = get_parameter("circle_radius").as_double();
    circle_num_peds_ = get_parameter("circle_num_peds").as_int();

    for (int i = 0; i < circle_num_peds_; ++i) {
      double angle = 2.0 * M_PI * i / circle_num_peds_;
      static_ped_x.push_back(circle_center_x + circle_radius * std::cos(angle));
      static_ped_y.push_back(circle_center_y + circle_radius * std::sin(angle));
    }

    rate_hz = get_parameter("rate").as_double();
    auto rate = std::chrono::milliseconds((int)(1000. / rate_hz));

    n_peds = get_parameter("n_peds").as_int();
    moving = get_parameter("moving").as_bool();

    ped_start_x = get_parameter("ped_start_x").as_double_array();
    ped_end_x = get_parameter("ped_end_x").as_double_array();
    ped_y = get_parameter("ped_y").as_double_array();
    ped_vel = get_parameter("ped_vel").as_double_array();

    static_ped_x = get_parameter("static_peds/x").as_double_array();
    static_ped_y = get_parameter("static_peds/y").as_double_array();

    start_x_range = get_parameter("start_x_range").as_double_array();
    end_x_range = get_parameter("end_x_range").as_double_array();
    y_range = get_parameter("y_range").as_double_array();
    speed_range = get_parameter("speed_range").as_double_array();

    start_x_dist  = std::uniform_real_distribution<>(start_x_range[0], start_x_range[1]);
    end_x_dist    = std::uniform_real_distribution<>(end_x_range[0], end_x_range[1]);
    y_dist        = std::uniform_real_distribution<>(y_range[0], y_range[1]);
    speed_dist    = std::uniform_real_distribution<>(speed_range[0], speed_range[1]);

    velocity_reduction_factor = 0.5;

    personal_space_radius = get_parameter("personal_space_radius").as_double();
    repulsion_gain = get_parameter("repulsion_gain").as_double();

    pedestrian_pub_ =
      create_publisher<crowd_nav_interfaces::msg::PedestrianArray>("pedestrians", 10);

    timer_ = create_wall_timer(rate, std::bind(&SimulatePedestrians::timer_callback, this));

    move_srv_ = create_service<std_srvs::srv::Empty>(
      "move_ped",
      std::bind(&SimulatePedestrians::move_cb, this, std::placeholders::_1, std::placeholders::_2));
    reset_srv_ = create_service<std_srvs::srv::Empty>(
      "reset_ped",
      std::bind(&SimulatePedestrians::reset_cb, this, std::placeholders::_1, std::placeholders::_2));

    // Initialize arrays
    ped_positions.resize(n_peds);
    ped_positions_initialized.resize(n_peds, false);
    ped_velocities.resize(n_peds);
    ped_directions.resize(n_peds);
    ped_vel.resize(n_peds);

    // Stratified random start
    for (int i = 0; i < n_peds; ++i) {
      double section = (start_x_range[1] - start_x_range[0]) / 3.0;
      std::vector<double> sections = {
        start_x_range[0] + section * 0.5,
        start_x_range[0] + section * 1.5,
        start_x_range[0] + section * 2.5
      };
      std::uniform_real_distribution<> section_dist(-0.5, 0.5);
      ped_positions[i] = sections[i % 3] + section_dist(gen);

      std::uniform_real_distribution<> corridor_y(-1.5, 1.5);
      ped_y[i] = corridor_y(gen);

      double angle = angle_dist(gen);
      ped_directions[i].x = std::cos(angle);
      ped_directions[i].y = std::sin(angle);

      ped_vel[i] = speed_dist(gen);

      ped_velocities[i].linear.x = ped_directions[i].x * ped_vel[i];
      ped_velocities[i].linear.y = ped_directions[i].y * ped_vel[i];
    }

    pedestrian_groups.resize(1);
    pedestrian_groups[0].member_ids = {1, 2};
    pedestrian_groups[0].cohesion_force = 0.5;

    obstacles_sub_ = create_subscription<crowd_nav_interfaces::msg::LineObstacles>(
      "obstacles", 10, std::bind(&SimulatePedestrians::obstaclesCallback, this, std::placeholders::_1));

    declare_parameter("tree_gain", 0.5);
    declare_parameter("tree_range", 1.0);
  }

private:
  double rate_hz;
  int n_peds;
  bool moving;
  double velocity_reduction_factor;
  double personal_space_radius;
  double repulsion_gain;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<crowd_nav_interfaces::msg::PedestrianArray>::SharedPtr pedestrian_pub_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr move_srv_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv_;

  std::vector<double> ped_start_x, ped_end_x, ped_y, ped_vel;
  std::vector<double> static_ped_x, static_ped_y;
  std::vector<double> ped_positions;
  std::vector<bool> ped_positions_initialized;
  std::vector<geometry_msgs::msg::Twist> ped_velocities;

  std::vector<double> start_x_range, end_x_range, y_range, speed_range;
  std::vector<geometry_msgs::msg::Vector3> ped_directions;

  bool go;
  int idx = 0;

  rclcpp::Subscription<crowd_nav_interfaces::msg::LineObstacles>::SharedPtr obstacles_sub_;

  std::mt19937 gen;
  std::uniform_real_distribution<> start_x_dist;
  std::uniform_real_distribution<> end_x_dist;
  std::uniform_real_distribution<> y_dist;
  std::uniform_real_distribution<> speed_dist;
  std::uniform_real_distribution<> angle_dist;

  // ----- Trees-related members added -----
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr tree_markers_pub_;
  bool trees_generated_;
  int n_trees_;
  visualization_msgs::msg::MarkerArray tree_markers_;
  // ----------------------------------------

  // Circle pedestrians member
  int circle_num_peds_;

  void move_cb(std::shared_ptr<std_srvs::srv::Empty::Request>,
               std::shared_ptr<std_srvs::srv::Empty::Response>)
  {
    RCLCPP_INFO_STREAM(get_logger(), "Start Moving pedestrians");
    go = true;
  }

  void reset_cb(std::shared_ptr<std_srvs::srv::Empty::Request>,
                std::shared_ptr<std_srvs::srv::Empty::Response>)
  {
    RCLCPP_INFO_STREAM(get_logger(), "Reset Pedestrians");
    idx = 0;
    for (int i = 0; i < n_peds; ++i) {
      if (i < circle_num_peds_) {
        ped_positions[i] = static_ped_x[i];
        ped_y[i] = static_ped_y[i];
      } else {
        ped_positions[i] = start_x_dist(gen);
        ped_y[i] = y_dist(gen);
      }

      double angle = angle_dist(gen);
      ped_directions[i].x = std::cos(angle);
      ped_directions[i].y = std::sin(angle);

      ped_vel[i] = (i < circle_num_peds_) ? 0.0 : speed_dist(gen);
      ped_velocities[i].linear.x = ped_directions[i].x * ped_vel[i];
      ped_velocities[i].linear.y = ped_directions[i].y * ped_vel[i];

      ped_positions_initialized[i] = false;
    }
  }

  void obstaclesCallback(const crowd_nav_interfaces::msg::LineObstacles::SharedPtr msg)
  {
    obstacles_.clear();
    for (auto &obs : msg->obstacles) {
      obstacles_.push_back(obs);
    }
  }

  void timer_callback()
  {
    crowd_nav_interfaces::msg::PedestrianArray peds;
    auto current_time = this->get_clock()->now();

    // Build outgoing PedestrianArray
    for (int i = 0; i < n_peds; i++) {
      crowd_nav_interfaces::msg::Pedestrian ped;
      ped.header.stamp = current_time;
      ped.id = i;  // Use 0-indexing
      ped.pose.position.x = ped_positions[i];
      ped.pose.position.y = ped_y[i];
      ped.velocity = ped_velocities[i];
      peds.pedestrians.push_back(ped);
    }

    if(go) {
      for(int i=0; i<n_peds; i++) {
        if (i < circle_num_peds_) {
          ped_velocities[i].linear.x = 0.0;
          ped_velocities[i].linear.y = 0.0;
          continue;
        }
        
        auto& ped = peds.pedestrians[i];
        
        // Update forces
        updatePedestrianRepulsion(ped, peds);  // New repulsion
        updateObstacleForces(ped);
        updateTreeForces(ped);
        updateRandomMotion(ped);
        updateGroupForces(ped, peds);

        const double MAX_SPEED = 1.2;
        double cur_speed = std::hypot(ped.velocity.linear.x, ped.velocity.linear.y);
        if(cur_speed > MAX_SPEED) {
          ped.velocity.linear.x *= (MAX_SPEED / cur_speed);
          ped.velocity.linear.y *= (MAX_SPEED / cur_speed);
        }

        ped_positions[i] += ped.velocity.linear.x * (1.0 / rate_hz);
        ped_y[i] += ped.velocity.linear.y * (1.0 / rate_hz);

        // Apply boundary constraints
        if(ped_positions[i] <= MIN_X || ped_positions[i] >= MAX_X) {
          ped.velocity.linear.x *= -1;
          ped_positions[i] = std::clamp(ped_positions[i], MIN_X, MAX_X);
        }
        if(ped_y[i] <= MIN_Y || ped_y[i] >= MAX_Y) {
          ped.velocity.linear.y *= -1;
          ped_y[i] = std::clamp(ped_y[i], MIN_Y, MAX_Y);
        }

        ped_positions[i] = std::clamp(ped_positions[i], MIN_X, MAX_X);
        ped_y[i] = std::clamp(ped_y[i], MIN_Y, MAX_Y);

        ped_velocities[i] = ped.velocity;
      }
    }

    pedestrian_pub_->publish(peds);

    // ----- Publish trees -----
    pub_trees();
    // -------------------------
  }

  void updateGroupForces(
      crowd_nav_interfaces::msg::Pedestrian& ped,
      const crowd_nav_interfaces::msg::PedestrianArray& all_peds)
  {
    for (const auto& group : pedestrian_groups) {
      if (std::find(group.member_ids.begin(), group.member_ids.end(), ped.id) != group.member_ids.end()) {
        double com_x = 0.0, com_y = 0.0;
        for (auto id : group.member_ids) {
          auto it = std::find_if(all_peds.pedestrians.begin(), all_peds.pedestrians.end(),
            [id](const auto& p){ return p.id == id; });
          if (it != all_peds.pedestrians.end()) {
            com_x += it->pose.position.x;
            com_y += it->pose.position.y;
          }
        }
        com_x /= group.member_ids.size();
        com_y /= group.member_ids.size();

        double dx = com_x - ped.pose.position.x;
        double dy = com_y - ped.pose.position.y;
        ped.velocity.linear.x += group.cohesion_force * dx;
        ped.velocity.linear.y += group.cohesion_force * dy;
      }
    }
  }

  void updateRandomMotion(crowd_nav_interfaces::msg::Pedestrian& ped)
  {
    const double DIRECTION_CHANGE_PROB = 0.02;
    const double MAX_SPEED = 1.2;
    const double MAX_ANGLE_CHANGE = M_PI / 4;

    if ((double)rand() / RAND_MAX < DIRECTION_CHANGE_PROB) {
      double current_angle = std::atan2(ped.velocity.linear.y, ped.velocity.linear.x);
      double angle_change = ((double)rand() / RAND_MAX - 0.5) * 2.0 * MAX_ANGLE_CHANGE;
      current_angle += angle_change;

      double speed = std::hypot(ped.velocity.linear.x, ped.velocity.linear.y);
      speed += ((double)rand() / RAND_MAX - 0.5) * 0.2;
      speed = std::max(0.1, std::min(speed, MAX_SPEED));

      ped.velocity.linear.x = std::cos(current_angle) * speed;
      ped.velocity.linear.y = std::sin(current_angle) * speed;
    }

    ped.velocity.linear.x += ((double)rand() / RAND_MAX - 0.5) * 0.1;
    ped.velocity.linear.y += ((double)rand() / RAND_MAX - 0.5) * 0.1;

    double cur_speed = std::hypot(ped.velocity.linear.x, ped.velocity.linear.y);
    if (cur_speed > MAX_SPEED) {
      ped.velocity.linear.x *= (MAX_SPEED / cur_speed);
      ped.velocity.linear.y *= (MAX_SPEED / cur_speed);
    }
  }

  void updateObstacleForces(crowd_nav_interfaces::msg::Pedestrian& ped) 
  {
    const double OBSTACLE_FORCE_GAIN = get_parameter("obstacle_gain").as_double();
    const double OBSTACLE_RANGE = get_parameter("obstacle_range").as_double();

    // Wall avoidance forces
    // Left wall (MIN_X)
    double dist_left = ped.pose.position.x - MIN_X;
    if (dist_left < OBSTACLE_RANGE) {
      double force = OBSTACLE_FORCE_GAIN * (1.0 / (dist_left + 0.1));
      ped.velocity.linear.x += force;
    }

    // Right wall (MAX_X)
    double dist_right = MAX_X - ped.pose.position.x;
    if (dist_right < OBSTACLE_RANGE) {
      double force = OBSTACLE_FORCE_GAIN * (1.0 / (dist_right + 0.1));
      ped.velocity.linear.x -= force;
    }

    // Bottom wall (MIN_Y)
    double dist_bottom = ped.pose.position.y - MIN_Y;
    if (dist_bottom < OBSTACLE_RANGE) {
      double force = OBSTACLE_FORCE_GAIN * (1.0 / (dist_bottom + 0.1));
      ped.velocity.linear.y += force;
    }

    // Top wall (MAX_Y)
    double dist_top = MAX_Y - ped.pose.position.y;
    if (dist_top < OBSTACLE_RANGE) {
      double force = OBSTACLE_FORCE_GAIN * (1.0 / (dist_top + 0.1));
      ped.velocity.linear.y -= force;
    }

    // Existing obstacle avoidance
    for (const auto& obs : obstacles_) {
      double dx = ped.pose.position.x - obs.start.x;
      double dy = ped.pose.position.y - obs.start.y;
      double dist = std::hypot(dx, dy);
      if (dist < OBSTACLE_RANGE) {
        ped.velocity.linear.x += OBSTACLE_FORCE_GAIN * (dx / dist);
        ped.velocity.linear.y += OBSTACLE_FORCE_GAIN * (dy / dist);
      }
    }
  }

  void updatePedestrianRepulsion(
      crowd_nav_interfaces::msg::Pedestrian& ped,
      const crowd_nav_interfaces::msg::PedestrianArray& all_peds)
  {
      for (const auto& other : all_peds.pedestrians) {
          if (ped.id == other.id) continue;
          
          double dx = ped.pose.position.x - other.pose.position.x;
          double dy = ped.pose.position.y - other.pose.position.y;
          double dist = std::hypot(dx, dy);
          
          // Only react to nearby pedestrians
          if (dist < personal_space_radius && dist > 0.001) {
              double force = repulsion_gain * (1.0 - (dist/personal_space_radius));
              ped.velocity.linear.x += force * (dx/dist);
              ped.velocity.linear.y += force * (dy/dist);
          }
      }
  }
  
  // ----- Trees-related function added -----
  void pub_trees() {
    if (!trees_generated_) {
      std::mt19937 tree_gen(42); // fixed seed
      const double margin = 1.0;
      std::uniform_real_distribution<double> tree_x_dist(MIN_X + margin, MAX_X - margin);
      std::uniform_real_distribution<double> tree_y_dist(MIN_Y + margin, MAX_Y - margin);

      for (int i = 0; i < n_trees_; i++) {
        double tx = tree_x_dist(tree_gen);
        double ty = tree_y_dist(tree_gen);

        visualization_msgs::msg::Marker trunk;
        trunk.header.frame_id = "brne_odom";
        trunk.header.stamp = this->get_clock()->now();
        trunk.ns = "trees";
        trunk.id = i * 2;
        trunk.type = visualization_msgs::msg::Marker::CYLINDER;
        trunk.action = visualization_msgs::msg::Marker::ADD;
        trunk.pose.position.x = tx;
        trunk.pose.position.y = ty;
        trunk.pose.position.z = 0.5;
        trunk.pose.orientation.w = 1.0;
        trunk.scale.x = 0.2;
        trunk.scale.y = 0.2;
        trunk.scale.z = 1.0;
        trunk.color.r = 0.55;
        trunk.color.g = 0.27;
        trunk.color.b = 0.07;
        trunk.color.a = 1.0;
        tree_markers_.markers.push_back(trunk);

        visualization_msgs::msg::Marker foliage;
        foliage.header.frame_id = "brne_odom";
        foliage.header.stamp = this->get_clock()->now();
        foliage.ns = "trees";
        foliage.id = i * 2 + 1;
        foliage.type = visualization_msgs::msg::Marker::SPHERE;
        foliage.action = visualization_msgs::msg::Marker::ADD;
        foliage.pose.position.x = tx;
        foliage.pose.position.y = ty;
        foliage.pose.position.z = 1.5;
        foliage.pose.orientation.w = 1.0;
        foliage.scale.x = 1.0;
        foliage.scale.y = 1.0;
        foliage.scale.z = 1.0;
        foliage.color.r = 0.0;
        foliage.color.g = 1.0;
        foliage.color.b = 0.0;
        foliage.color.a = 1.0;
        tree_markers_.markers.push_back(foliage);
      }
      trees_generated_ = true;
    }
    for (auto & tree : tree_markers_.markers) {
      tree.header.stamp = this->get_clock()->now();
    }
    tree_markers_pub_->publish(tree_markers_);
  }
  // ----------------------------------------

  void updateTreeForces(crowd_nav_interfaces::msg::Pedestrian& ped)
  {
    double tree_gain = get_parameter("tree_gain").as_double();
    double tree_range = get_parameter("tree_range").as_double();

    for (const auto & marker : tree_markers_.markers) {
      if (marker.type != visualization_msgs::msg::Marker::CYLINDER)
        continue;

      double dx = ped.pose.position.x - marker.pose.position.x;
      double dy = ped.pose.position.y - marker.pose.position.y;
      double dist = std::hypot(dx, dy);

      if (dist < tree_range && dist > 0.001) {
        double force = tree_gain * (1.0 - (dist / tree_range));
        ped.velocity.linear.x += force * (dx / dist);
        ped.velocity.linear.y += force * (dy / dist);
      }
    }
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimulatePedestrians>());
  rclcpp::shutdown();
  return 0;
}