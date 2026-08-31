#include "engine/physics_scene.hpp"
#include <stdexcept>

namespace {
physx::PxFilterFlags filterShader(physx::PxFilterObjectAttributes, physx::PxFilterData,
    physx::PxFilterObjectAttributes, physx::PxFilterData,
    physx::PxPairFlags& pairFlags, const void*, physx::PxU32) {
    pairFlags = physx::PxPairFlag::eCONTACT_DEFAULT |
                physx::PxPairFlag::eNOTIFY_TOUCH_FOUND |
                physx::PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
                physx::PxPairFlag::eNOTIFY_CONTACT_POINTS;
    return physx::PxFilterFlag::eDEFAULT;
}
ncrc::Vec3 vec(const physx::PxVec3& v) { return {v.x, v.y, v.z}; }
ncrc::Quat quat(const physx::PxQuat& q) { return {q.w, q.x, q.y, q.z}; }
}

namespace ncrc {

void ContactRecorder::onContact(const physx::PxContactPairHeader& header, const physx::PxContactPair* pairs, physx::PxU32 count) {
    for (physx::PxU32 i = 0; i < count; ++i) {
        physx::PxContactPairPoint points[64];
        const physx::PxU32 pointCount = pairs[i].extractContacts(points, 64);
        for (physx::PxU32 p = 0; p < pointCount; ++p) {
            ContactPointState state;
            state.bodyA = header.actors[0] && header.actors[0]->getName() ? header.actors[0]->getName() : "UNKNOWN";
            state.bodyB = header.actors[1] && header.actors[1]->getName() ? header.actors[1]->getName() : "UNKNOWN";
            state.point = vec(points[p].position);
            state.normal = vec(points[p].normal);
            state.impulse = vec(points[p].impulse);
            state.separation = points[p].separation;
            contacts.push_back(std::move(state));
        }
    }
}

PhysicsScene::PhysicsScene(PhysXContext& context, const SceneConfig& config) : context_(context), config_(config) {
    physx::PxSceneDesc desc(context.physics().getTolerancesScale());
    desc.gravity = config.gravity;
    desc.cpuDispatcher = &context.dispatcher();
    desc.filterShader = filterShader;
    desc.simulationEventCallback = &recorder_;
    desc.solverType = config.solverType;
    scene_ = context.physics().createScene(desc);
    if (!scene_) throw std::runtime_error("PxPhysics::createScene failed");
    material_ = context.physics().createMaterial(config.staticFriction, config.dynamicFriction, config.restitution);
    if (!material_) throw std::runtime_error("PxPhysics::createMaterial failed");
}

PhysicsScene::~PhysicsScene() {
    if (material_) material_->release();
    if (scene_) scene_->release();
}

void PhysicsScene::createGround() {
    auto* ground = context_.physics().createRigidStatic(physx::PxTransform(physx::PxVec3(0, 0, -0.05f)));
    ground->setName("terrain");
    auto* shape = context_.physics().createShape(physx::PxBoxGeometry(100, 100, 0.05f), *material_);
    ground->attachShape(*shape); shape->release(); scene_->addActor(*ground);
}

physx::PxRigidDynamic* PhysicsScene::createDynamicBox(const char* name, const physx::PxTransform& pose, const physx::PxVec3& halfExtents, float mass) {
    auto* body = context_.physics().createRigidDynamic(pose);
    body->setName(name);
    auto* shape = context_.physics().createShape(physx::PxBoxGeometry(halfExtents), *material_);
    body->attachShape(*shape); shape->release();
    body->setMass(mass);
    const physx::PxVec3 size = halfExtents * 2.0f;
    body->setMassSpaceInertiaTensor(physx::PxVec3(
        mass * (size.y * size.y + size.z * size.z) / 12.0f,
        mass * (size.x * size.x + size.z * size.z) / 12.0f,
        mass * (size.x * size.x + size.y * size.y) / 12.0f));
    scene_->addActor(*body);
    return body;
}

void PhysicsScene::step() {
    recorder_.clear();
    scene_->simulate(config_.physicsDt);
    scene_->fetchResults(true);
}

RootState PhysicsScene::extractRootState(const physx::PxRigidBody& body) {
    const auto pose = body.getGlobalPose();
    const auto linear = body.getLinearVelocity();
    const auto angular = body.getAngularVelocity();
    const auto inverse = pose.q.getConjugate();
    return {vec(pose.p), quat(pose.q.getNormalized()), vec(linear), vec(angular), vec(inverse.rotate(linear)), vec(inverse.rotate(angular))};
}

}  // namespace ncrc
