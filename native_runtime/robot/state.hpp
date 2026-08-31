#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ncrc {

struct Vec3 { float x{}, y{}, z{}; };
struct Quat { float w{1.0f}, x{}, y{}, z{}; };

struct RootState {
    Vec3 position;
    Quat orientation;
    Vec3 linearVelocityWorld;
    Vec3 angularVelocityWorld;
    Vec3 linearVelocityBody;
    Vec3 angularVelocityBody;
};

struct JointState {
    std::string name;
    float position{};
    float velocity{};
    float acceleration{};
    float targetPosition{};
    float targetVelocity{};
    float commandTorque{};
    float appliedTorque{};
    float lowerLimit{};
    float upperLimit{};
};

struct LinkState {
    std::string name;
    Vec3 position;
    Quat orientation;
    Vec3 linearVelocity;
    Vec3 angularVelocity;
    float mass{};
    Vec3 centerOfMass;
    Vec3 inertia;
};

struct ContactPointState {
    std::string bodyA;
    std::string bodyB;
    Vec3 point;
    Vec3 normal;
    Vec3 impulse;
    float separation{};
};

struct FootState {
    std::string name;
    Vec3 position;
    Vec3 velocity;
    bool inContact{};
    Vec3 contactForce;
    Vec3 frictionForce;
    float airTime{};
    bool firstContactAfterAir{};
};

struct RobotState {
    RootState root;
    std::vector<JointState> joints;
    std::vector<LinkState> links;
    std::vector<FootState> feet;
};

struct SimulationState {
    std::uint64_t physicsStep{};
    float simulationTime{};
    RobotState robot;
    std::vector<ContactPointState> contacts;
    bool numericallyValid{true};
    std::string error;
};

}  // namespace ncrc
