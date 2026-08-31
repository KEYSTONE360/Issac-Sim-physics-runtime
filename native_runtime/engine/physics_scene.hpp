#pragma once

#include "engine/physx_context.hpp"
#include "robot/state.hpp"
#include <vector>

namespace ncrc {

struct SceneConfig {
    physx::PxVec3 gravity{0.0f, 0.0f, -9.81f};
    float physicsDt{0.005f};
    physx::PxSolverType::Enum solverType{physx::PxSolverType::eTGS};
    float staticFriction{1.0f};
    float dynamicFriction{1.0f};
    float restitution{0.0f};
};

class ContactRecorder final : public physx::PxSimulationEventCallback {
public:
    void clear() { contacts.clear(); }
    std::vector<ContactPointState> contacts;
    void onContact(const physx::PxContactPairHeader&, const physx::PxContactPair*, physx::PxU32) override;
    void onConstraintBreak(physx::PxConstraintInfo*, physx::PxU32) override {}
    void onWake(physx::PxActor**, physx::PxU32) override {}
    void onSleep(physx::PxActor**, physx::PxU32) override {}
    void onTrigger(physx::PxTriggerPair*, physx::PxU32) override {}
    void onAdvance(const physx::PxRigidBody* const*, const physx::PxTransform*, const physx::PxU32) override {}
};

class PhysicsScene final {
public:
    PhysicsScene(PhysXContext& context, const SceneConfig& config);
    ~PhysicsScene();
    PhysicsScene(const PhysicsScene&) = delete;
    PhysicsScene& operator=(const PhysicsScene&) = delete;

    physx::PxScene& scene() const { return *scene_; }
    physx::PxPhysics& physics() const { return context_.physics(); }
    physx::PxMaterial& material() const { return *material_; }
    float dt() const { return config_.physicsDt; }
    void step();
    const std::vector<ContactPointState>& contacts() const { return recorder_.contacts; }
    physx::PxRigidDynamic* createDynamicBox(const char* name, const physx::PxTransform&, const physx::PxVec3& halfExtents, float mass);
    void createGround();
    static RootState extractRootState(const physx::PxRigidBody& body);

private:
    PhysXContext& context_;
    SceneConfig config_;
    ContactRecorder recorder_;
    physx::PxScene* scene_{nullptr};
    physx::PxMaterial* material_{nullptr};
};

}  // namespace ncrc
