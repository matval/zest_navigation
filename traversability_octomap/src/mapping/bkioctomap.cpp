#include <algorithm>
#include <pcl/filters/voxel_grid.h>

#include "bkioctomap.h"
#include "bki.h"

using std::vector;

#define DEBUG true;

#ifdef DEBUG
    #include <iostream>

    #define Debug_Msg(msg) {\
        std::cout << "Debug: " << msg << std::endl; }
#endif

namespace traversability_bki
{
    TraversabilityBKIOctoMap::TraversabilityBKIOctoMap () : TraversabilityBKIOctoMap(
        0.1f, // resolution
        4,      // block_depth
        1.0,    // sf2
        1.0,    // ell
        0.5f,   // mean_prior
        4.0f,   // lambda_prior
        1.0f,   // beta_prior
        1.0f,   // prior_A
        1.0f,   // prior_B
        1.0f,   // var_thresh
        0.3f,   // free_thresh
        0.7f,   // occupied_thresh
        -3.0f,  // min_z
        3.0f    // max_z
    ) { }

    TraversabilityBKIOctoMap::TraversabilityBKIOctoMap (
        float resolution,
        unsigned short block_depth,
        float sf2,
        float ell,
        float mean_prior,
        float lambda_prior,
        float beta_prior,
        float pior_A,
        float prior_B,
        float var_thresh,
        float free_thresh,
        float occupied_thresh,
        float min_z,
        float max_z) :
            resolution(resolution),
            block_depth(block_depth),
            block_size((float) pow(2, block_depth - 1) * resolution),
            min_z(min_z),
            max_z(max_z)
    {
        Block::resolution = resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
        Block::index_map = init_index_map(Block::key_loc_map, block_depth);
        
        // Bug fixed
        Block::cell_num = static_cast<unsigned short>(round(Block::size / Block::resolution));
        std::cout << "block::resolution: " << Block::resolution << std::endl;
        std::cout << "block::size: " << Block::size << std::endl;
        std::cout << "block::cell_num: " << Block::cell_num << std::endl;
        
        TraversabilityOcTree::max_depth = block_depth;

        TraversabilityOcTreeNode::sf2 = sf2;
        TraversabilityOcTreeNode::ell = ell;
        TraversabilityOcTreeNode::mean_prior = mean_prior;
        TraversabilityOcTreeNode::lambda_prior = lambda_prior;
        TraversabilityOcTreeNode::beta_prior = beta_prior;
        TraversabilityOcTreeNode::prior_A = pior_A;
        TraversabilityOcTreeNode::prior_B = prior_B;
        TraversabilityOcTreeNode::var_thresh = var_thresh;
        TraversabilityOcTreeNode::free_thresh = free_thresh;
        TraversabilityOcTreeNode::occupied_thresh = occupied_thresh;
    }

    TraversabilityBKIOctoMap::~TraversabilityBKIOctoMap() {
        for (auto it = block_arr.begin(); it != block_arr.end(); ++it) {
            if (it->second != nullptr) {
                delete it->second;
            }
        }
    }

    void TraversabilityBKIOctoMap::set_resolution(float resolution) {
        this->resolution = resolution;
        Block::resolution = resolution;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void TraversabilityBKIOctoMap::set_block_depth(unsigned short max_depth) {
        this->block_depth = max_depth;
        TraversabilityOcTree::max_depth = max_depth;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void TraversabilityBKIOctoMap::insert_pointcloud(
        const PCLPointCloud &cloud, const point3f &origin, float ds_resolution, float free_res, float min_range, float max_range) {

#ifdef DEBUG
        Debug_Msg("Insert pointcloud: " << "cloud size: " << cloud.size() << " origin: " << origin);
#endif

        ////////// Preparation //////////////////////////
        /////////////////////////////////////////////////
        GPLineCloud xy;
        GPLineCloud rays;
        vector<int> ray_idx;
        this->get_training_data(cloud, origin, ds_resolution, free_res, min_range, max_range, min_z, max_z, xy, rays, ray_idx);
        // If pointcloud after max_range filtering is empty
        //  no need to do anything
        if (xy.size() == 0) {
            return;
        }
#ifdef DEBUG
        Debug_Msg("Training data size: " << xy.size());
#endif

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        for (auto it = xy.cbegin(); it != xy.cend(); ++it) {
            float px = std::get<0>(*it).x0();
            float py = std::get<0>(*it).y0();
            float pz = std::get<0>(*it).z0();
            float p[] = {px, py, pz};
            rtree.Insert(p, p, const_cast<GPLineType *>(&*it));
        }
        /////////////////////////////////////////////////
        ////////// Training /////////////////////////////
        /////////////////////////////////////////////////
        vector<BlockHashKey> test_blocks;
        std::unordered_map<BlockHashKey, TraversabilityBKI3f *> bgkl_arr;
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < blocks.size(); ++i) {
            BlockHashKey key = blocks[i];
            ExtendedBlock eblock = get_extended_block(key);
            if (has_gp_points_in_bbox(eblock))
#ifdef OPENMP
#pragma omp critical
#endif
            {
                test_blocks.push_back(key);
            };

            GPLineCloud block_xy;
            get_gp_points_in_bbox(key, block_xy);
            if (block_xy.size() < 1)
                continue;

            vector<float> block_x, block_y;
            for (auto it = block_xy.cbegin(); it != block_xy.cend(); ++it) {
#ifdef OPENMP
#pragma omp critical
#endif
            {
                float px = std::get<0>(*it).x0();
                float py = std::get<0>(*it).y0();
                float pz = std::get<0>(*it).z0();
                float traversability = std::get<1>(*it);
                float occupancy = std::get<2>(*it);

                // Add the point location to the training data
                block_x.push_back(px);
                block_x.push_back(py);
                block_x.push_back(pz);
                block_x.push_back(px);
                block_x.push_back(py);
                block_x.push_back(pz);

                // Add the point traversability and occupancy to the training data
                block_y.push_back(traversability);
                block_y.push_back(occupancy);
            }
            };
            TraversabilityBKI3f *bgkl = new TraversabilityBKI3f(TraversabilityOcTreeNode::sf2, TraversabilityOcTreeNode::ell);

            bgkl->train(block_x, block_y);
#ifdef OPENMP
#pragma omp critical
#endif
            {
                bgkl_arr.emplace(key, bgkl);
            };
        }
#ifdef DEBUG
        Debug_Msg("Training done");
        Debug_Msg("Prediction: block number: " << test_blocks.size());
#endif
        /////////////////////////////////////////////////

        ////////// Prediction ///////////////////////////
        /////////////////////////////////////////////////
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < test_blocks.size(); ++i)
        {
            BlockHashKey key = test_blocks[i];
#ifdef OPENMP
#pragma omp critical
#endif
            {
                if (block_arr.find(key) == block_arr.end())
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
            };
            Block *block = block_arr[key];
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it)
            {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }

            ExtendedBlock eblock = block->get_extended_block();
            for (auto block_it = eblock.cbegin(); block_it != eblock.cend(); ++block_it)
            {
                auto bgkl = bgkl_arr.find(*block_it);
                if (bgkl == bgkl_arr.end())
                    continue;

                vector<vector<float>> ybars;
                #ifdef OPENMP
                #pragma omp critical(bgkl_predict)
                #endif
                {
                    bgkl->second->predict(xs, ybars);
                }

                int j = 0;
                for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j)
                {
                    // Get current node
                    TraversabilityOcTreeNode &node = leaf_it.get_node();
                    
                    std::cout << "Before update. ybars[j][2]: " << ybars[j][2] << std::endl;
                    // Apply the update step
                    if (ybars[j][2] > 0.001f)
                        node.update(ybars[j]);
                    std::cout << "After update." << std::endl;
                }
            }
        }
#ifdef DEBUG
        Debug_Msg("Prediction done");
#endif
        /////////////////////////////////////////////////
        ////////// Pruning //////////////////////////////
        ///////////////////////////////////////////////
#ifdef OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
            auto block = block_arr.find(key);

            if (block == block_arr.end())
                continue;

            block->second->prune();
        }
#ifdef DEBUG
        Debug_Msg("Pruning done");
#endif
        /////////////////////////////////////////////////
        ////////// Cleaning /////////////////////////////
        /////////////////////////////////////////////////
        for (auto it = bgkl_arr.begin(); it != bgkl_arr.end(); ++it)
            delete it->second;

        rtree.RemoveAll();
    }

    void TraversabilityBKIOctoMap::get_bbox(point3f &lim_min, point3f &lim_max) const {
        lim_min = point3f(0, 0, 0);
        lim_max = point3f(0, 0, 0);

        GPLineCloud centers;
        for (auto it = block_arr.cbegin(); it != block_arr.cend(); ++it) {
            centers.emplace_back(point6f(it->second->get_center()), 1, 1);
        }
        if (centers.size() > 0) {
            bbox(centers, lim_min, lim_max);
            lim_min -= point3f(block_size, block_size, block_size) * 0.5;
            lim_max += point3f(block_size, block_size, block_size) * 0.5;
        }
    }

    void TraversabilityBKIOctoMap::get_training_data(
        const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
        float free_resolution, float min_range, float max_range, float min_z, float max_z,
        GPLineCloud &xy, GPLineCloud &rays, vector<int> &ray_idx) const {

        PCLPointCloud sampled_hits;
        traversabilityVoxelGrid(cloud, sampled_hits, ds_resolution);

        rays.clear();
        ray_idx.clear();
        xy.clear();
        int idx = 0;
        
        for (auto it = sampled_hits.begin(); it != sampled_hits.end(); ++it) {
            point3f p(it->x, it->y, it->z);

            if (p.z() < min_z || p.z() > max_z)
                continue;
            
            if (max_range > 0) {
                double l = (p - origin).norm();
                if (l > max_range)
                    continue;
            }

            float l = (float) (p - origin).norm();

            float nx = (p.x() - origin.x()) / l;
            float ny = (p.y() - origin.y()) / l;
            float nz = (p.z() - origin.z()) / l;
            
            point3f occ_endpt(origin.x() + nx * l, origin.y() + ny * l, origin.z() + nz * l);
            xy.emplace_back(point6f(occ_endpt), it->traversability, 1.0f);
            ray_idx.push_back(-1);

            PointCloud frees_n;
            beam_sample(occ_endpt, origin, frees_n, free_resolution, min_range);

            if (min_range <= 0) {
                xy.emplace_back(point6f(origin.x(), origin.y(), origin.z()), it->traversability, 0.0f);
                ray_idx.push_back(idx);
            }
            
            for (auto p = frees_n.begin(); p != frees_n.end(); ++p) {
                xy.emplace_back(point6f(p->x(), p->y(), p->z()), it->traversability, 0.0f);
                ray_idx.push_back(idx);
            }

            l = l - free_resolution;
            point3f free_endpt(origin.x() + nx * l, origin.y() + ny * l, origin.z() + nz * l);
            point6f line6f(origin, free_endpt);
            rays.emplace_back(line6f, it->traversability, 0.0f);

            ++idx;
        }
    }

    void TraversabilityBKIOctoMap::beam_sample(const point3f &hit, const point3f &origin, PointCloud &frees,
                                float free_resolution, float min_range) const {
        frees.clear();

        float x0 = origin.x();
        float y0 = origin.y();
        float z0 = origin.z();

        float x = hit.x();
        float y = hit.y();
        float z = hit.z();

        float l = (float) sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));

        float nx = (x - x0) / l;
        float ny = (y - y0) / l;
        float nz = (z - z0) / l;

        float d = l - free_resolution;
        float min_range2 = std::max(min_range, 0.0f);
        while (d > min_range2) {
            frees.emplace_back(x0 + nx * d, y0 + ny * d, z0 + nz * d);
            d -= free_resolution;
        }
    }

    /*
     * Compute bounding box of pointcloud
     * Precondition: cloud non-empty
     */
    void TraversabilityBKIOctoMap::bbox(const GPLineCloud &cloud, point3f &lim_min, point3f &lim_max) const {
        assert(cloud.size() > 0);
        vector<float> x, y, z;
        for (auto it = cloud.cbegin(); it != cloud.cend(); ++it) {
            float mx0 = std::get<0>(*it).x0();
            float mx1 = std::get<0>(*it).x1();
            float my0 = std::get<0>(*it).y0();
            float my1 = std::get<0>(*it).y1();
            float mz0 = std::get<0>(*it).z0();
            float mz1 = std::get<0>(*it).z1();
            x.push_back(mx0);
            x.push_back(mx1);
            y.push_back(my0);
            y.push_back(my1);
            z.push_back(mz0);
            z.push_back(mz1);
        }

        auto xlim = std::minmax_element(x.cbegin(), x.cend());
        auto ylim = std::minmax_element(y.cbegin(), y.cend());
        auto zlim = std::minmax_element(z.cbegin(), z.cend());

        lim_min.x() = *xlim.first;
        lim_min.y() = *ylim.first;
        lim_min.z() = *zlim.first;

        lim_max.x() = *xlim.second;
        lim_max.y() = *ylim.second;
        lim_max.z() = *zlim.second;
    }

    void TraversabilityBKIOctoMap::get_blocks_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                       vector<BlockHashKey> &blocks) const {
        for (float x = lim_min.x() - block_size; x <= lim_max.x() + 2 * block_size; x += block_size) {
            for (float y = lim_min.y() - block_size; y <= lim_max.y() + 2 * block_size; y += block_size) {
                for (float z = lim_min.z() - block_size; z <= lim_max.z() + 2 * block_size; z += block_size) {
                    blocks.push_back(block_to_hash_key(x, y, z));
                }
            }
        }
    }

    int TraversabilityBKIOctoMap::get_gp_points_in_bbox(const BlockHashKey &key,
                                         GPLineCloud &out) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return get_gp_points_in_bbox(lim_min, lim_max, out);
    }

    int TraversabilityBKIOctoMap::has_gp_points_in_bbox(const BlockHashKey &key) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return has_gp_points_in_bbox(lim_min, lim_max);
    }

    int TraversabilityBKIOctoMap::get_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                         GPLineCloud &out) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, TraversabilityBKIOctoMap::search_callback, static_cast<void *>(&out));
    }

    int TraversabilityBKIOctoMap::has_gp_points_in_bbox(const point3f &lim_min,
                                         const point3f &lim_max) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, TraversabilityBKIOctoMap::count_callback, NULL);
    }

    bool TraversabilityBKIOctoMap::count_callback(GPLineType *p, void *arg) {
        return false;
    }

    bool TraversabilityBKIOctoMap::search_callback(GPLineType *p, void *arg) {
        GPLineCloud *out = static_cast<GPLineCloud *>(arg);
        out->push_back(*p);
        return true;
    }


    int TraversabilityBKIOctoMap::has_gp_points_in_bbox(const ExtendedBlock &block) {
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            if (has_gp_points_in_bbox(*it) > 0)
                return 1;
        }
        return 0;
    }

    int TraversabilityBKIOctoMap::get_gp_points_in_bbox(const ExtendedBlock &block,
                                         GPLineCloud &out) {
        int n = 0;
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            n += get_gp_points_in_bbox(*it, out);
        }
        return n;
    }

    Block *TraversabilityBKIOctoMap::search(BlockHashKey key) const {
        auto block = block_arr.find(key);
        if (block == block_arr.end()) {
            return nullptr;
        } else {
            return block->second;
        }
    }

    TraversabilityOcTreeNode TraversabilityBKIOctoMap::search(point3f p) const {
        Block *block = search(block_to_hash_key(p));
        if (block == nullptr) {
          return TraversabilityOcTreeNode();
        } else {
          return TraversabilityOcTreeNode(block->search(p));
        }
    }

    TraversabilityOcTreeNode TraversabilityBKIOctoMap::search(float x, float y, float z) const {
        return search(point3f(x, y, z));
    }
}
