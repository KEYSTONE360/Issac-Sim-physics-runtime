#pragma once

#include "PxPhysicsAPI.h"
#include "extensions/PxDefaultCpuDispatcher.h"
#include <memory>
#include <stdexcept>

namespace ncrc {

class PhysXContext final {
public:
    explicit PhysXContext(physx::PxU32 cpuThreads);
    ~PhysXContext();
    PhysXContext(const PhysXContext&) = delete;
    PhysXContext& operator=(const PhysXContext&) = delete;

    physx::PxPhysics& physics() const { return *physics_; }
    physx::PxFoundation& foundation() const { return *foundation_; }
    physx::PxCpuDispatcher& dispatcher() const { return *dispatcher_; }

private:
    physx::PxDefaultAllocator allocator_;
    physx::PxDefaultErrorCallback errorCallback_;
    physx::PxFoundation* foundation_{nullptr};
    physx::PxPhysics* physics_{nullptr};
    physx::PxDefaultCpuDispatcher* dispatcher_{nullptr};
};

}  // namespace ncrc
