#pragma once

#include "engine/physics_scene.hpp"
#include <string>

namespace ncrc {

class ArticulationProbe final {
public:
    explicit ArticulationProbe(PhysicsScene& scene);
    ~ArticulationProbe();
    std::vector<JointState> readJoints(float physicsDt);
    std::vector<LinkState> readLinks() const;
    physx::PxArticulationReducedCoordinate& articulation() const { return *articulation_; }

private:
    physx::PxArticulationReducedCoordinate* articulation_{nullptr};
    std::vector<float> previousVelocity_;
    std::vector<std::string> jointNames_;
};

}  // namespace ncrc
