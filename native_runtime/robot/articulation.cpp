#include "robot/articulation.hpp"
#include "extensions/PxRigidActorExt.h"
#include "extensions/PxRigidBodyExt.h"
#include <stdexcept>

namespace {
ncrc::Vec3 vec(const physx::PxVec3& v) { return {v.x, v.y, v.z}; }
ncrc::Quat quat(const physx::PxQuat& q) { return {q.w, q.x, q.y, q.z}; }
}

namespace ncrc {

ArticulationProbe::ArticulationProbe(PhysicsScene& scene) {
    articulation_ = scene.physics().createArticulationReducedCoordinate();
    if (!articulation_) throw std::runtime_error("createArticulationReducedCoordinate failed");
    articulation_->setSolverIterationCounts(4, 4);

    auto* root = articulation_->createLink(nullptr, physx::PxTransform(physx::PxVec3(0.0f, 0.0f, 1.1f)));
    root->setName("probe_root");
    physx::PxRigidActorExt::createExclusiveShape(*root, physx::PxBoxGeometry(0.16f, 0.12f, 0.20f), scene.material());
    physx::PxRigidBodyExt::updateMassAndInertia(*root, 5.0f);

    auto* child = articulation_->createLink(root, physx::PxTransform(physx::PxVec3(0.0f, 0.0f, 0.65f)));
    child->setName("probe_foot");
    physx::PxRigidActorExt::createExclusiveShape(*child, physx::PxBoxGeometry(0.10f, 0.08f, 0.25f), scene.material());
    physx::PxRigidBodyExt::updateMassAndInertia(*child, 1.0f);
    auto* joint = child->getInboundJoint();
    joint->setJointType(physx::PxArticulationJointType::eREVOLUTE);
    joint->setParentPose(physx::PxTransform(physx::PxVec3(0.0f, 0.0f, -0.20f)));
    joint->setChildPose(physx::PxTransform(physx::PxVec3(0.0f, 0.0f, 0.25f)));
    joint->setMotion(physx::PxArticulationAxis::eTWIST, physx::PxArticulationMotion::eLIMITED);
    joint->setLimitParams(physx::PxArticulationAxis::eTWIST, physx::PxArticulationLimit(-1.0f, 1.0f));
    joint->setDriveParams(physx::PxArticulationAxis::eTWIST, physx::PxArticulationDrive(20.0f, 2.0f, 100.0f));
    joint->setDriveTarget(physx::PxArticulationAxis::eTWIST, 0.25f);
    jointNames_.push_back("probe_revolute");
    scene.scene().addArticulation(*articulation_);
    previousVelocity_.assign(articulation_->getDofs(), 0.0f);
}
ArticulationProbe::~ArticulationProbe() { if (articulation_) articulation_->release(); }

std::vector<JointState> ArticulationProbe::readJoints(float physicsDt) {
    auto* cache = articulation_->createCache();
    if (!cache) throw std::runtime_error("PxArticulation cache creation failed");
    articulation_->copyInternalStateToCache(*cache,
        physx::PxArticulationCacheFlag::ePOSITION |
        physx::PxArticulationCacheFlag::eVELOCITY |
        physx::PxArticulationCacheFlag::eACCELERATION |
        physx::PxArticulationCacheFlag::eFORCE |
        physx::PxArticulationCacheFlag::eLINK_INCOMING_JOINT_FORCE);
    std::vector<JointState> states;
    const auto dofs = articulation_->getDofs();
    states.reserve(dofs);
    for (physx::PxU32 i = 0; i < dofs; ++i) {
        const float finiteDifference = (cache->jointVelocity[i] - previousVelocity_[i]) / physicsDt;
        states.push_back({
            i < jointNames_.size() ? jointNames_[i] : "unnamed_dof",
            cache->jointPosition[i], cache->jointVelocity[i], finiteDifference,
            0.25f, 0.0f, cache->jointForce[i], cache->linkIncomingJointForce[i + 1].torque.x, -1.0f, 1.0f,
        });
        previousVelocity_[i] = cache->jointVelocity[i];
    }
    cache->release();
    return states;
}

std::vector<LinkState> ArticulationProbe::readLinks() const {
    const auto count = articulation_->getNbLinks();
    std::vector<physx::PxArticulationLink*> links(count);
    articulation_->getLinks(links.data(), count);
    std::vector<LinkState> states; states.reserve(count);
    for (auto* link : links) {
        const auto pose = link->getGlobalPose();
        states.push_back({
            link->getName() ? link->getName() : "UNKNOWN",
            vec(pose.p), quat(pose.q.getNormalized()), vec(link->getLinearVelocity()), vec(link->getAngularVelocity()),
            link->getMass(), vec(link->getCMassLocalPose().p), vec(link->getMassSpaceInertiaTensor()),
        });
    }
    return states;
}

}  // namespace ncrc
