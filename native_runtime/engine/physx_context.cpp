#include "engine/physx_context.hpp"
#include "extensions/PxDefaultCpuDispatcher.h"

namespace ncrc {

PhysXContext::PhysXContext(physx::PxU32 cpuThreads) {
    foundation_ = PxCreateFoundation(PX_PHYSICS_VERSION, allocator_, errorCallback_);
    if (!foundation_) throw std::runtime_error("PxCreateFoundation failed");
    physx::PxTolerancesScale scale;
    physics_ = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation_, scale, false, nullptr);
    if (!physics_) throw std::runtime_error("PxCreatePhysics failed");
    dispatcher_ = physx::PxDefaultCpuDispatcherCreate(cpuThreads == 0 ? 1 : cpuThreads);
    if (!dispatcher_) throw std::runtime_error("PxDefaultCpuDispatcherCreate failed");
}

PhysXContext::~PhysXContext() {
    if (dispatcher_) dispatcher_->release();
    if (physics_) physics_->release();
    if (foundation_) foundation_->release();
}

}  // namespace ncrc
