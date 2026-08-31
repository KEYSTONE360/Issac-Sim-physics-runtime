#include "engine/physics_scene.hpp"
#include "robot/articulation.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>

int main(int argc, char**) {
    try {
        ncrc::PhysXContext context(std::max(1u, std::thread::hardware_concurrency()));
        ncrc::SceneConfig config;
        ncrc::RootState initial, freeFall, final;
        std::size_t contactCount = 0;
        {
            ncrc::PhysicsScene rigidScene(context, config);
            rigidScene.createGround();
            auto* box = rigidScene.createDynamicBox("free_fall_probe", physx::PxTransform(physx::PxVec3(0, 0, 2)), physx::PxVec3(0.1f), 1.0f);
            initial = ncrc::PhysicsScene::extractRootState(*box);
            for (int i = 0; i < 200; ++i) {
                rigidScene.step();
                contactCount += rigidScene.contacts().size();
                if (i == 99) freeFall = ncrc::PhysicsScene::extractRootState(*box);
            }
            final = ncrc::PhysicsScene::extractRootState(*box);
        }
        std::vector<ncrc::JointState> joints;
        std::vector<ncrc::LinkState> links;
        {
            ncrc::PhysicsScene articulationScene(context, config);
            articulationScene.createGround();
            ncrc::ArticulationProbe articulation(articulationScene);
            for (int i = 0; i < 200; ++i) { articulationScene.step(); joints = articulation.readJoints(config.physicsDt); }
            links = articulation.readLinks();
        }
        const bool valid = std::isfinite(final.position.z) && final.orientation.w <= 1.0001f && final.orientation.w >= -1.0001f && contactCount > 0 && joints.size() == 1 && links.size() == 2 && std::isfinite(joints[0].position);
        std::cout << std::fixed << std::setprecision(7)
                  << "{\n  \"runtime\": \"NCRC PhysX 5.6.1 CPU\",\n"
                  << "  \"renderer\": false,\n  \"cuda_required\": false,\n"
                  << "  \"physics_dt\": " << config.physicsDt << ",\n"
                  << "  \"initial_z\": " << initial.position.z << ",\n"
                  << "  \"free_fall_step\": 100,\n"
                  << "  \"free_fall_time\": " << 100 * config.physicsDt << ",\n"
                  << "  \"free_fall_z\": " << freeFall.position.z << ",\n"
                  << "  \"free_fall_vz\": " << freeFall.linearVelocityWorld.z << ",\n"
                  << "  \"final_z\": " << final.position.z << ",\n"
                  << "  \"final_vz\": " << final.linearVelocityWorld.z << ",\n"
                  << "  \"contact_points_observed\": " << contactCount << ",\n"
                  << "  \"articulation_links\": " << links.size() << ",\n"
                  << "  \"articulation_dofs\": " << joints.size() << ",\n"
                  << "  \"joint_position\": " << (joints.empty() ? 0.0f : joints[0].position) << ",\n"
                  << "  \"joint_velocity\": " << (joints.empty() ? 0.0f : joints[0].velocity) << ",\n"
                  << "  \"joint_acceleration\": " << (joints.empty() ? 0.0f : joints[0].acceleration) << ",\n"
                  << "  \"joint_force\": " << (joints.empty() ? 0.0f : joints[0].appliedTorque) << ",\n"
                  << "  \"state_extraction_valid\": " << (valid ? "true" : "false") << "\n}\n";
        return valid ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "NCRC_PHYSX_ERROR: " << error.what() << '\n';
        return 1;
    }
}
