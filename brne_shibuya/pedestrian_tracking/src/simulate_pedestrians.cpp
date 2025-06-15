#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

#include "rclcpp/rclcpp.hpp"
#include "crowd_nav_interfaces/msg/pedestrian.hpp"
#include "crowd_nav_interfaces/msg/pedestrian_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_srvs/srv/empty.hpp"

using namespace std::chrono_literals;

// Updated waypoints with new D coordinates
constexpr double A_x = 7.13649, A_y = -3.96949;
constexpr double B_x = 6.97596, B_y = 2.96581;
constexpr double C_x = 2.98578, C_y = 0.0501903;
constexpr double D_x = 3.05853, D_y = 4.03012;  // Updated D coordinates

struct PedestrianPath {
    std::vector<std::pair<double, double>> waypoints;
    int current_waypoint = 0;  // Destination marker index
    int source_marker = 0;     // Marker at which the ped is waiting (its origin)
    double current_x = 0.0;
    double current_y = 0.0;
    double current_vx = 0.0;
    double current_vy = 0.0;
    double end_x = 0.0;
    double end_y = 0.0;
    bool arrived = false;      // True if reached destination and now waiting
    // Record when the pedestrian reached the goal.
    rclcpp::Time wait_start;
};

class SimulatePedestrians : public rclcpp::Node
{
public:
    SimulatePedestrians()
    : Node("simulate_pedestrians"), gen(std::random_device{}())
    {
        // Parameters:
        declare_parameter("rate", 20.0);
        declare_parameter("n_moving_peds", 100);  // Was 80
        declare_parameter("personal_space", 0.8);
        declare_parameter("wait_time", 15.0);
        declare_parameter("max_speed", 1.4);
        declare_parameter("repulsion_gain", 8.0); // Add missing parameter

        initialize_parameters();
        setup_publishers();
        setup_services();
        initialize_pedestrians();
        start_update_timer();
    }

private:
    // Configuration parameters
    double rate_hz;
    int n_moving_peds;
    double personal_space;
    double max_speed;
    double wait_time;
    bool simulation_active = false;
    double repulsion_gain;

    // Pedestrian data
    std::map<int, PedestrianPath> moving_peds;
    std::mt19937 gen;

    // ROS publishers, services, and timer
    rclcpp::Publisher<crowd_nav_interfaces::msg::PedestrianArray>::SharedPtr ped_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr crosswalk_pub_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr move_srv_, reset_srv_;
    rclcpp::TimerBase::SharedPtr update_timer_;

    // --- Initialization functions ---
    void initialize_parameters() {
        rate_hz = get_parameter("rate").as_double();
        n_moving_peds = get_parameter("n_moving_peds").as_int();
        personal_space = get_parameter("personal_space").as_double();
        wait_time = get_parameter("wait_time").as_double();
        max_speed = get_parameter("max_speed").as_double();
        repulsion_gain = get_parameter("repulsion_gain").as_double();
    }

    void setup_publishers() {
        ped_pub_ = create_publisher<crowd_nav_interfaces::msg::PedestrianArray>("pedestrians", 10);
        crosswalk_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/crosswalks", 10);
    }

    void setup_services() {
        move_srv_ = create_service<std_srvs::srv::Empty>(
            "move_ped",
            [this](
                const std::shared_ptr<rmw_request_id_t> /*request_header*/,
                const std::shared_ptr<std_srvs::srv::Empty::Request> /*request*/,
                std::shared_ptr<std_srvs::srv::Empty::Response> /*response*/)
            {
                RCLCPP_INFO(get_logger(), "Starting pedestrian movement");
                simulation_active = true;
            });

        reset_srv_ = create_service<std_srvs::srv::Empty>(
            "reset_ped",
            [this](
                const std::shared_ptr<rmw_request_id_t> /*request_header*/,
                const std::shared_ptr<std_srvs::srv::Empty::Request> /*request*/,
                std::shared_ptr<std_srvs::srv::Empty::Response> /*response*/)
            {
                RCLCPP_INFO(get_logger(), "Resetting pedestrians");
                initialize_pedestrians();
                simulation_active = false;
            });
    }

    // Initialize pedestrians so that exactly n_moving_peds/4 are placed at each marker.
    // Here, we use the four waypoints (A, B, C, D) in any fixed order.
    void initialize_pedestrians() {
        moving_peds.clear();
        // We use the same waypoint list for all pedestrians.
        const std::vector<std::pair<double, double>> waypoints = {
            {A_x, A_y}, {B_x, B_y}, {C_x, C_y}, {D_x, D_y}
        };

        std::uniform_real_distribution<double> offset_dist(-2.5, 2.5);  // Was -1.5
        std::vector<std::pair<double, double>> occupied_positions;

        // Distribute pedestrians evenly across markers.
        for (int i = 0; i < n_moving_peds; i++) {
            int vertex = i % 4;  // ensures equal spread
            bool valid_position = false;
            int attempts = 0;
            double x, y;

            while (!valid_position && attempts < 500) {
                x = waypoints[vertex].first + offset_dist(gen);
                y = waypoints[vertex].second + offset_dist(gen);
                valid_position = true;
                for (const auto& pos : occupied_positions) {
                    if (std::hypot(x - pos.first, y - pos.second) < (personal_space * 0.7)) {
                        valid_position = false;
                        break;
                    }
                }
                attempts++;
            }

            if (valid_position) {
                occupied_positions.emplace_back(x, y);
                PedestrianPath path;
                path.current_x = x;
                path.current_y = y;
                path.waypoints = waypoints;
                // For initial placement, the marker where the ped is located is its source.
                path.source_marker = vertex;
                // Immediately assign a balanced destination from the available three markers.
                set_balanced_goal(path);
                path.arrived = false;
                moving_peds[i] = path;
            }
        }
    }

    // --- Goal Assignment Functions ---
    // When a pedestrian is ready to move (after waiting), it chooses a destination from the three
    // markers (excluding its source marker) that currently has the fewest moving pedestrians from that source.
    void set_balanced_goal(PedestrianPath &path) {
        int source = path.source_marker;
        std::map<int, int> dest_counts;
        for (int j = 0; j < 4; j++) {
            if (j == source) continue;
            dest_counts[j] = 0;
        }
        for (const auto& [id, other] : moving_peds) {
            if (!other.arrived && other.source_marker == source) {
                if (other.current_waypoint != source) {
                    dest_counts[other.current_waypoint]++;
                }
            }
        }
        int best = -1;
        int best_count = std::numeric_limits<int>::max();
        for (const auto &entry : dest_counts) {
            if (entry.second < best_count) {
                best = entry.first;
                best_count = entry.second;
            }
        }
        if (best == -1) {
            std::vector<int> candidates;
            for (int j = 0; j < 4; j++) {
                if (j != source) candidates.push_back(j);
            }
            std::uniform_int_distribution<int> dist(0, candidates.size()-1);
            best = candidates[dist(gen)];
        }
        path.current_waypoint = best;
        
        // Compute direction vector from source to destination waypoint.
        double dx_waypoint = path.waypoints[best].first - path.waypoints[source].first;
        double dy_waypoint = path.waypoints[best].second - path.waypoints[source].second;
        
        // Calculate right-hand perpendicular vector (clockwise 90°).
        double perp_x = dy_waypoint;
        double perp_y = -dx_waypoint;
        double perp_length = std::hypot(perp_x, perp_y);
        if (perp_length > 1e-6) {
            perp_x /= perp_length;
            perp_y /= perp_length;
        }
        
        // Apply consistent right-side offset.
        std::uniform_real_distribution<double> lateral_offset(0.1, 0.3);  // 0.1-0.3m offset
        path.end_x = path.waypoints[best].first + perp_x * lateral_offset(gen);
        path.end_y = path.waypoints[best].second + perp_y * lateral_offset(gen);
        path.arrived = false;
    }

    // --- Update Loop ---
    void update_moving_pedestrians() {
        if (!simulation_active) return;
        const auto now = get_clock()->now();
        const double dt = 1.0 / rate_hz;

        for (auto& [id, path] : moving_peds) {
            if (path.arrived) {
                double elapsed = (now - path.wait_start).seconds();
                if (elapsed >= wait_time) {
                    // From the marker where the pedestrian is waiting (its source), assign a new balanced destination.
                    set_balanced_goal(path);
                } else {
                    path.current_vx = 0.0;
                    path.current_vy = 0.0;
                    continue;
                }
            }
            double dx = path.end_x - path.current_x;
            double dy = path.end_y - path.current_y;
            double dist = std::hypot(dx, dy);
            double arrival_threshold = std::max(0.3, personal_space * 0.8);
            if (dist < arrival_threshold) {
                // Snap to the exact waypoint position upon arrival
                path.current_x = path.waypoints[path.current_waypoint].first;
                path.current_y = path.waypoints[path.current_waypoint].second;
                path.source_marker = path.current_waypoint;
                path.arrived = true;
                path.wait_start = now;
                path.current_vx = 0.0;
                path.current_vy = 0.0;
                continue;
            }
            double base_speed = std::max(0.5, dist / 1.2);
            double desired_speed = std::min(base_speed, max_speed);
            path.current_vx = (dx / dist) * desired_speed;
            path.current_vy = (dy / dist) * desired_speed;
            updatePedestrianRepulsion(path, dt);
            path.current_x += path.current_vx * dt;
            path.current_y += path.current_vy * dt;
        }
    }

    // --- Repulsion Function ---
    void updatePedestrianRepulsion(PedestrianPath& path, double dt) {
        const double collision_threshold = personal_space * 0.8; // React earlier

        for (auto& [other_id, other_path] : moving_peds) {
            if (&other_path == &path) continue;
            
            double dx = path.current_x - other_path.current_x;
            double dy = path.current_y - other_path.current_y;
            double dist = std::hypot(dx, dy);
            double dvx = path.current_vx - other_path.current_vx;
            double dvy = path.current_vy - other_path.current_vy;
            
            // Only react if pedestrians are moving towards each other.
            if ((dx*dvx + dy*dvy) >= 0) continue; 
            
            if (dist < collision_threshold) {
                double nx = dx / dist;
                double ny = dy / dist;
                
                // Apply strong repulsion force when within collision threshold.
                double force = repulsion_gain * (1.0 - (dist / collision_threshold));
                path.current_vx += nx * force * dt * 2.0;  // Increased multiplier
                path.current_vy += ny * force * dt * 2.0;
            }
        }
        
        // Speed clamping.
        double current_speed = std::hypot(path.current_vx, path.current_vy);
        if (current_speed > max_speed) {
            path.current_vx *= (max_speed / current_speed);
            path.current_vy *= (max_speed / current_speed);
        }
    }

    // --- Crosswalk Visualization ---
    // (The crosswalk visualization code remains unchanged.)
    void publish_crosswalks() {
        visualization_msgs::msg::MarkerArray marker_array;
        std::vector<geometry_msgs::msg::Point> corners;
        geometry_msgs::msg::Point pt;
        pt.x = A_x; pt.y = A_y; pt.z = 0.0; corners.push_back(pt);
        pt.x = B_x; pt.y = B_y; pt.z = 0.0; corners.push_back(pt);
        pt.x = D_x; pt.y = D_y; pt.z = 0.0; corners.push_back(pt);
        pt.x = C_x; pt.y = C_y; pt.z = 0.0; corners.push_back(pt);

        const double stripe_length = 0.5;
        const double stripe_line_width = 0.1;
        int marker_id = 0;
        const int n_corners = corners.size();
        for (int i = 0; i < n_corners; i++) {
            geometry_msgs::msg::Point start = corners[i];
            geometry_msgs::msg::Point end = corners[(i+1) % n_corners];
            double dx = end.x - start.x;
            double dy = end.y - start.y;
            double seg_length = std::hypot(dx, dy);
            if (seg_length < 1e-6) continue;
            double ux = dx / seg_length;
            double uy = dy / seg_length;
            double perp_x = -uy;
            double perp_y = ux;
            int num_stripes = (i % 2 == 0) ? 5 : 7;
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "brne_odom";
            marker.header.stamp = this->get_clock()->now();
            marker.ns = "crosswalk";
            marker.id = marker_id++;
            marker.type = visualization_msgs::msg::Marker::LINE_LIST;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x = stripe_line_width;
            marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 1.0;
            marker.lifetime = rclcpp::Duration(0, 0);
            for (int j = 0; j < num_stripes; j++) {
                double t = (j + 1.0) / (num_stripes + 1.0);
                geometry_msgs::msg::Point center;
                center.x = start.x + t * dx;
                center.y = start.y + t * dy;
                center.z = 0.0;
                geometry_msgs::msg::Point p1, p2;
                p1.x = center.x + (stripe_length / 2.0) * perp_x;
                p1.y = center.y + (stripe_length / 2.0) * perp_y;
                p1.z = 0.0;
                p2.x = center.x - (stripe_length / 2.0) * perp_x;
                p2.y = center.y - (stripe_length / 2.0) * perp_y;
                p2.z = 0.0;
                marker.points.push_back(p1);
                marker.points.push_back(p2);
            }
            marker_array.markers.push_back(marker);
        }
        // Extra cross lines (diagonals)
        std::vector<std::pair<geometry_msgs::msg::Point, geometry_msgs::msg::Point>> extra_lines;
        extra_lines.push_back({corners[2], corners[0]});
        extra_lines.push_back({corners[3], corners[1]});
        for (const auto &line : extra_lines) {
            geometry_msgs::msg::Point start = line.first;
            geometry_msgs::msg::Point end = line.second;
            double dx = end.x - start.x;
            double dy = end.y - start.y;
            double seg_length = std::hypot(dx, dy);
            if (seg_length < 1e-6) continue;
            double ux = dx / seg_length;
            double uy = dy / seg_length;
            double perp_x = -uy;
            double perp_y = ux;
            int num_stripes = 6;
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "brne_odom";
            marker.header.stamp = this->get_clock()->now();
            marker.ns = "crosswalk_extra";
            marker.id = marker_id++;
            marker.type = visualization_msgs::msg::Marker::LINE_LIST;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x = stripe_line_width;
            marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 1.0;
            marker.lifetime = rclcpp::Duration(0, 0);
            for (int j = 0; j < num_stripes; j++) {
                double t = (j + 1.0) / (num_stripes + 1.0);
                geometry_msgs::msg::Point center;
                center.x = start.x + t * dx;
                center.y = start.y + t * dy;
                center.z = 0.0;
                geometry_msgs::msg::Point p1, p2;
                p1.x = center.x + (stripe_length / 2.0) * perp_x;
                p1.y = center.y + (stripe_length / 2.0) * perp_y;
                p1.z = 0.0;
                p2.x = center.x - (stripe_length / 2.0) * perp_x;
                p2.y = center.y - (stripe_length / 2.0) * perp_y;
                p2.z = 0.0;
                marker.points.push_back(p1);
                marker.points.push_back(p2);
            }
            marker_array.markers.push_back(marker);
        }
        crosswalk_pub_->publish(marker_array);
    }

    void publish_pedestrians() {
        crowd_nav_interfaces::msg::PedestrianArray msg;
        msg.header.stamp = get_clock()->now();
        msg.header.frame_id = "brne_odom";  // Set the appropriate frame
        for (const auto& [id, path] : moving_peds) {
            crowd_nav_interfaces::msg::Pedestrian ped;
            ped.pose.position.x = path.current_x;
            ped.pose.position.y = path.current_y;
            ped.velocity.linear.x = path.current_vx;
            ped.velocity.linear.y = path.current_vy;
            // Optionally assign an id if required by your detection pipeline:
            ped.id = id;
            msg.pedestrians.push_back(ped);
        }
        ped_pub_->publish(msg);
    }

    void start_update_timer() {
        update_timer_ = create_wall_timer(
            std::chrono::duration<double>(1.0 / rate_hz),
            [this]() {
                update_moving_pedestrians();
                publish_pedestrians();
                publish_crosswalks();
            }
        );
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SimulatePedestrians>());
    rclcpp::shutdown();
    return 0;
}